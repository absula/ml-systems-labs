"""
Lab 3: Architectures and their costs (Ch. 4)

MLP vs CNN vs patch-Transformer on Fashion-MNIST: parameter counts, hand vs thop FLOPs,
CPU inference latency, 5-epoch training, accuracy/cost trade-off plots.

Run (from the repo root):  python lab03/src/lab3.py
Outputs go to lab03/results/ (tradeoffs.png, results.json, log.txt).
"""
import copy
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms

SEED = 0
BATCH = 128
EPOCHS = 5
LR = 1e-3
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REPO_ROOT = Path(__file__).resolve().parents[2]
torch.manual_seed(SEED)

TRAIN_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------- 1. Dataset
def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full = datasets.FashionMNIST(REPO_ROOT / "data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(REPO_ROOT / "data", train=False, download=True, transform=tf)
    train, val = random_split(full, [55_000, 5_000], generator=torch.Generator().manual_seed(SEED))
    return (
        DataLoader(train, batch_size=BATCH, shuffle=True),
        DataLoader(val, batch_size=BATCH),
        DataLoader(test, batch_size=BATCH),
    )


# ---------------------------------------------------------------- 2. Models
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchTransformer(nn.Module):
    """28x28 -> 16 patches of 7x7 -> Linear(49, 64) -> 2 encoder layers -> mean pool -> Linear(64, 10)."""

    def __init__(self, patch=7, d_model=64, nhead=4, ff=128, layers=2):
        super().__init__()
        self.patch = patch
        n_patches = (28 // patch) ** 2
        self.proj = nn.Linear(patch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))  # learned position embedding
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        p = self.patch
        b = x.shape[0]
        # (B,1,28,28) -> (B,4,4,7,7) -> (B,16,49)
        x = x.unfold(2, p, p).unfold(3, p, p).reshape(b, -1, p * p)
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": PatchTransformer}


# ---------------------------------------------------------------- 3. Cost
def hand_flops():
    """Ch. 4 formulas, FLOPs = 2 x MACs, batch 1, bias/activation/softmax/LayerNorm ignored."""
    lin = lambda i, o, n=1: 2 * n * i * o
    conv = lambda k, ci, co, h, w: 2 * k * k * ci * co * h * w
    seq, d, ff = 16, 64, 128
    attn_layer = {
        "QKV projections (2*seq*d*3d)": 2 * seq * d * 3 * d,
        "Scores QK^T (2*seq^2*d)": 2 * seq * seq * d,
        "Scores x V (2*seq^2*d)": 2 * seq * seq * d,
        "Output projection (2*seq*d*d)": 2 * seq * d * d,
        "FFN (2*seq*d*ff*2)": 2 * 2 * seq * d * ff,
    }
    attn_one = sum(attn_layer.values())
    return {
        "MLP": {
            "layers": {"Linear 784->256": lin(784, 256), "Linear 256->128": lin(256, 128),
                       "Linear 128->10": lin(128, 10)},
        },
        "CNN": {
            "layers": {"Conv1 1->16 @28x28": conv(3, 1, 16, 28, 28),
                       "Conv2 16->32 @14x14": conv(3, 16, 32, 14, 14),
                       "Linear 1568->10": lin(1568, 10)},
        },
        "Attention": {
            "encoder_layer_breakdown": attn_layer,
            "layers": {"Patch proj 49->64 (x16)": lin(49, 64, seq),
                       "Encoder layer 1": attn_one, "Encoder layer 2": attn_one,
                       "Head 64->10": lin(64, 10)},
        },
    }


def thop_flops(model):
    model = copy.deepcopy(model).cpu().eval()  # thop leaves float64 buffers on the module
    macs, params = profile(model, inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    return int(macs), int(params)


@torch.inference_mode()
def latency_ms(model, batch, runs, warmup=10):
    model = model.cpu().eval()
    x = torch.randn(batch, 1, 28, 28)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


# ---------------------------------------------------------------- 4. Training
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(TRAIN_DEVICE), y.to(TRAIN_DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def train(model, train_dl, val_dl):
    model.to(TRAIN_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = n = 0
        for x, y in train_dl:
            x, y = x.to(TRAIN_DEVICE), y.to(TRAIN_DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            n += y.numel()
        val_acc = evaluate(model, val_dl)
        history.append({"epoch": epoch, "train_loss": running / n, "val_acc": val_acc})
        print(f"  epoch {epoch}: train_loss={running / n:.4f} val_acc={val_acc:.4f}")
    return history


# ---------------------------------------------------------------- 5. Plots
def plot(results):
    names = list(results)
    acc = [results[m]["test_acc"] * 100 for m in names]
    params = [results[m]["params"] for m in names]
    colors = {"MLP": "#1f77b4", "CNN": "#2ca02c", "Attention": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (params, "Parameters", False),
        ([results[m]["latency_ms_b1"] for m in names], "Latency, batch 1 (ms, median of 100, CPU)", False),
        ([results[m]["latency_ms_b128"] for m in names], "Latency, batch 128 (ms, median of 20, CPU)", False),
    ]
    for ax, (xs, label, _) in zip(axes, panels):
        for name, xv, yv in zip(names, xs, acc):
            ax.scatter(xv, yv, s=120, color=colors[name], zorder=3)
            ax.annotate(name, (xv, yv), textcoords="offset points", xytext=(8, -4))
        ax.set_xlabel(label)
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0].set_title("Accuracy vs parameters")
    axes[1].set_title("Accuracy vs latency (batch 1)")
    axes[2].set_title("Accuracy vs latency (batch 128)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tradeoffs.png", dpi=150)


def main():
    torch.set_num_threads(4)
    RESULTS_DIR.mkdir(exist_ok=True)
    train_dl, val_dl, test_dl = get_loaders()
    hand = hand_flops()
    results = {}
    for name, cls in MODELS.items():
        print(f"\n=== {name} ===")
        torch.manual_seed(SEED)
        model = cls()
        info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        print(info)
        macs, thop_params = thop_flops(model)
        r = {
            "params": info.total_params,
            "torchinfo_mult_adds": info.total_mult_adds,
            "thop_macs": macs,
            "thop_flops": 2 * macs,
            "thop_params": thop_params,
            "hand_flops": hand[name],
            "hand_flops_total": sum(hand[name]["layers"].values()),
            "latency_ms_b1": latency_ms(model, 1, 100),
            "latency_ms_b128": latency_ms(model, 128, 20),
        }
        r["history"] = train(model, train_dl, val_dl)
        r["test_acc"] = evaluate(model, test_dl)
        print(f"  test_acc={r['test_acc']:.4f}  params={r['params']}  "
              f"hand={r['hand_flops_total']}  thop_flops={r['thop_flops']}  "
              f"lat b1={r['latency_ms_b1']:.3f}ms b128={r['latency_ms_b128']:.3f}ms")
        results[name] = r
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    plot(results)


if __name__ == "__main__":
    main()
