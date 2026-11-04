"""
Lab 5: Frameworks and automatic differentiation (Ch. 7)

1-3. Scalar autograd engine (engine.py): Value node, +, *, **, relu, topological backward.
4.   Gradient check of ((a*b + c)**2).relu().sum() against PyTorch autograd and central differences.
5.   2 -> 8 -> 1 classifier on make_moons with the engine vs the same model in PyTorch (median of 5 runs).
6.   PyTorch graph inspection, eager vs torch.compile / torch.jit.trace, tensor layout and
     the cost of non-contiguous views.

Run (from the repo root):  python lab05/src/lab5.py
Outputs go to lab05/results/ (results.json, log.txt, moons_training.png).
"""
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

from engine import Value, vsum

SEED = 0
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_RUNS = 5


def timed(fn, warmup=3, reps=20):
    """Median wall time of fn() in milliseconds after warm-up."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


# ---------------------------------------------------------------- 1-3. Engine demo
def engine_demo():
    print("\n=== 1-3. Engine: graph construction and gradient accumulation ===")
    a, b, c = Value(2.0, label="a"), Value(-3.0, label="b"), Value(10.0, label="c")
    out = ((a * b + c) ** 2).relu()
    order = out.topo_order()
    print("Graph built in the forward pass (topological order):")
    for n in order:
        parents = ", ".join(f"{p.data:g}" for p in n._prev)
        print(f"  {n.label or n._op or 'const':>6}  data={n.data:<6g} parents=[{parents}]")
    out.backward()
    print(f"out = {out.data:g};  da = {a.grad:g}, db = {b.grad:g}, dc = {c.grad:g}")
    # (a*b + c) = 4, out = 16, dout/d(ab+c) = 8 -> da = 8*b = -24, db = 8*a = 16, dc = 8
    assert (a.grad, b.grad, c.grad) == (-24.0, 16.0, 8.0)

    # x is used in three places. Overwriting instead of accumulating would give 1 or 2*x.
    x = Value(3.0)
    y = x * x + x            # dy/dx = 2x + 1 = 7
    y.backward()
    print(f"accumulation check: y = x*x + x at x=3 -> dy/dx = {x.grad:g} (expected 7)")
    assert x.grad == 7.0
    return {"demo_grads": [a.grad, b.grad, c.grad], "accumulation_dx": x.grad}


# ---------------------------------------------------------------- 4. Gradient check
def f_engine(a, b, c):
    return vsum([((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c)])


def f_float(a, b, c):
    return sum(max((ai * bi + ci) ** 2, 0.0) for ai, bi, ci in zip(a, b, c))


def gradient_check(n=10, h=1e-6):
    print("\n=== 4. Gradient check: ((a*b + c)**2).relu().sum() ===")
    rng = np.random.default_rng(SEED)
    A, B, C = (rng.normal(size=n) for _ in range(3))

    # engine
    a, b, c = [[Value(v) for v in arr] for arr in (A, B, C)]
    out = f_engine(a, b, c)
    out.backward()
    g_engine = np.array([[v.grad for v in arr] for arr in (a, b, c)])

    # PyTorch autograd (float64 so the comparison is not limited by float32 rounding)
    ta, tb, tc = (torch.tensor(arr, dtype=torch.float64, requires_grad=True) for arr in (A, B, C))
    t_out = ((ta * tb + tc) ** 2).relu().sum()
    t_out.backward()
    g_torch = np.stack([ta.grad.numpy(), tb.grad.numpy(), tc.grad.numpy()])

    # central differences
    g_num = np.zeros((3, n))
    for k in range(3):
        for i in range(n):
            plus = [A.copy(), B.copy(), C.copy()]
            minus = [A.copy(), B.copy(), C.copy()]
            plus[k][i] += h
            minus[k][i] -= h
            g_num[k, i] = (f_float(*plus) - f_float(*minus)) / (2 * h)

    res = {
        "n_elements": n,
        "forward_engine": out.data,
        "forward_torch": t_out.item(),
        "max_abs_diff_engine_vs_torch": float(np.abs(g_engine - g_torch).max()),
        "max_abs_diff_engine_vs_numerical": float(np.abs(g_engine - g_num).max()),
        "max_abs_diff_torch_vs_numerical": float(np.abs(g_torch - g_num).max()),
    }
    print(f"forward: engine {res['forward_engine']:.12f}   torch {res['forward_torch']:.12f}")
    for name in ("a", "b", "c"):
        k = "abc".index(name)
        print(f"d/d{name}[:3]  engine {np.round(g_engine[k, :3], 8)}  torch {np.round(g_torch[k, :3], 8)}")
    print(f"max |engine - torch|      = {res['max_abs_diff_engine_vs_torch']:.3e}")
    print(f"max |engine - numerical|  = {res['max_abs_diff_engine_vs_numerical']:.3e}")
    print(f"max |torch  - numerical|  = {res['max_abs_diff_torch_vs_numerical']:.3e}")
    assert res["max_abs_diff_engine_vs_torch"] < 1e-6
    assert res["max_abs_diff_engine_vs_numerical"] < 1e-6
    print("PASS: both differences < 1e-6")
    return res


# ---------------------------------------------------------------- 5. make_moons
N_SAMPLES = 200
HIDDEN = 8
EPOCHS = 300
LR = 0.5


def init_params(seed):
    """Same initial weights for both implementations (PyTorch nn.Linear default init)."""
    torch.manual_seed(seed)
    l1, l2 = nn.Linear(2, HIDDEN), nn.Linear(HIDDEN, 1)
    return [p.detach().double().numpy().copy() for p in (l1.weight, l1.bias, l2.weight, l2.bias)]


class EngineMLP:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = [[Value(w) for w in row] for row in W1]   # 8 x 2
        self.b1 = [Value(v) for v in b1]
        self.W2 = [Value(w) for w in W2[0]]                 # 1 x 8
        self.b2 = Value(b2[0])

    def parameters(self):
        return [w for row in self.W1 for w in row] + self.b1 + self.W2 + [self.b2]

    def __call__(self, x):
        h = [(vsum([w * xi for w, xi in zip(row, x)]) + b).relu() for row, b in zip(self.W1, self.b1)]
        return (vsum([w * hi for w, hi in zip(self.W2, h)]) + self.b2).sigmoid()


def bce_engine(probs, ys, eps=1e-12):
    terms = [-(y * (p + eps).log() + (1 - y) * (1 - p + eps).log()) for p, y in zip(probs, ys)]
    return vsum(terms) * (1.0 / len(terms))


def train_engine(X, y, params):
    model = EngineMLP(*params)
    ps = model.parameters()
    losses = []
    for _ in range(EPOCHS):
        loss = bce_engine([model(x) for x in X.tolist()], y.tolist())
        for p in ps:
            p.grad = 0.0
        loss.backward()
        for p in ps:
            p.data -= LR * p.grad
        losses.append(loss.data)
    acc = np.mean([(model(x).data > 0.5) == t for x, t in zip(X.tolist(), y)])
    return losses, float(acc), model


def train_torch(X, y, params, dtype=torch.float64):
    torch.set_num_threads(1)
    model = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1), nn.Sigmoid()).to(dtype)
    with torch.no_grad():
        for p, v in zip(model.parameters(), params):
            p.copy_(torch.tensor(v, dtype=dtype))
    Xt = torch.tensor(X, dtype=dtype)
    yt = torch.tensor(y, dtype=dtype).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    losses = []
    for _ in range(EPOCHS):
        loss = loss_fn(model(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = ((model(Xt) > 0.5).double() == yt).double().mean().item()
    return losses, acc, model


def moons_experiment():
    print(f"\n=== 5. make_moons: 2 -> {HIDDEN} -> 1, sigmoid + BCE, full-batch SGD "
          f"(lr {LR}, {EPOCHS} epochs, {N_SAMPLES} samples) ===")
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=SEED)
    X = (X - X.mean(0)) / X.std(0)
    params = init_params(SEED)

    train_torch(X, y, params)   # untimed warm-up: first call pays PyTorch's lazy initialisation
    runs = {"engine": [], "torch": []}
    for r in range(N_RUNS):
        t0 = time.perf_counter()
        e_losses, e_acc, e_model = train_engine(X, y, params)
        runs["engine"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        t_losses, t_acc, _ = train_torch(X, y, params)
        runs["torch"].append(time.perf_counter() - t0)
        print(f"run {r + 1}: engine {runs['engine'][-1]:.3f} s   torch {runs['torch'][-1]:.3f} s")

    med_e, med_t = statistics.median(runs["engine"]), statistics.median(runs["torch"])
    loss_diff = float(np.abs(np.array(e_losses) - np.array(t_losses)).max())
    print(f"median training time: engine {med_e:.3f} s, torch {med_t:.3f} s  -> torch is {med_e / med_t:.0f}x faster")
    print(f"final loss: engine {e_losses[-1]:.6f}, torch {t_losses[-1]:.6f}; "
          f"max |loss diff| over all epochs = {loss_diff:.2e}")
    print(f"train accuracy: engine {e_acc:.3f}, torch {t_acc:.3f}")

    nodes = len(bce_engine([EngineMLP(*params)(x) for x in X.tolist()], y.tolist()).topo_order())
    print(f"engine graph size per epoch: {nodes:,} scalar nodes")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(e_losses, label="scalar engine", lw=2)
    axes[0].plot(t_losses, "--", label="PyTorch")
    axes[0].set(xlabel="epoch", ylabel="BCE loss", title="Training loss (same init)")
    axes[0].legend()
    model = e_model
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 60), np.linspace(-2.5, 2.5, 60))
    zz = np.array([model(list(pt)).data for pt in np.c_[xx.ravel(), yy.ravel()]]).reshape(xx.shape)
    axes[1].contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3, cmap="coolwarm")
    axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12, edgecolors="k", linewidths=0.3)
    axes[1].set(title=f"Engine decision boundary (acc {e_acc:.1%})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "moons_training.png", dpi=130)

    return {
        "times_engine_s": runs["engine"], "times_torch_s": runs["torch"],
        "median_engine_s": med_e, "median_torch_s": med_t, "speedup_torch": med_e / med_t,
        "final_loss_engine": e_losses[-1], "final_loss_torch": t_losses[-1],
        "max_abs_loss_diff": loss_diff, "acc_engine": e_acc, "acc_torch": t_acc,
        "engine_nodes_per_epoch": nodes,
    }


# ---------------------------------------------------------------- 6. PyTorch internals
class SmallMLP(nn.Module):
    def __init__(self, d_in=64, d_hidden=256, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def print_autograd_graph(t):
    """Walk the backward graph PyTorch recorded during the forward pass."""
    seen, stack = set(), [(t.grad_fn, 0)]
    while stack:
        fn, depth = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        extra = ""
        if hasattr(fn, "variable"):
            extra = f"  -> leaf {tuple(fn.variable.shape)}"
        print("  " + "  " * depth + type(fn).__name__ + extra)
        for nxt, _ in reversed(fn.next_functions):
            stack.append((nxt, depth + 1))


def pytorch_internals():
    print("\n=== 6. PyTorch: graph, eager vs compiled, tensor layout ===")
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = SmallMLP().eval()
    x = torch.randn(32, 64)
    res = {}

    print("\n-- 6a. Dynamic autograd graph (grad_fn chain of model(x).sum()) --")
    print_autograd_graph(model(x).sum())

    print("\n-- 6b. Static graph from torch.fx.symbolic_trace --")
    gm = torch.fx.symbolic_trace(model)
    print(gm.graph)

    print("\n-- 6c. Eager vs compiled / traced (CPU, 4 threads, inference_mode, median of 50 after warm-up) --")
    variants = {"eager": model}
    t0 = time.perf_counter()
    variants["jit.trace"] = torch.jit.trace(model, x)
    res["jit_trace_build_s"] = time.perf_counter() - t0
    try:
        compiled = torch.compile(model)
        t0 = time.perf_counter()
        with torch.inference_mode():
            compiled(x)           # first call triggers compilation
        res["compile_first_call_s"] = time.perf_counter() - t0
        variants["torch.compile"] = compiled
        print(f"torch.compile first call (compilation): {res['compile_first_call_s']:.2f} s")
    except Exception as e:  # noqa: BLE001
        print(f"torch.compile unsupported here: {type(e).__name__}: {str(e).splitlines()[0]}")
        res["compile_error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"

    timings = {}
    for bs in (1, 32, 1024):
        xb = torch.randn(bs, 64)
        timings[bs] = {}
        with torch.inference_mode():
            ref = model(xb)
            for name, m in variants.items():
                assert torch.allclose(m(xb), ref, atol=1e-5)
                timings[bs][name] = timed(lambda m=m: m(xb), warmup=10, reps=50)
    names = list(variants)
    print(f"{'batch':>6} " + " ".join(f"{n:>15}" for n in names))
    for bs, row in timings.items():
        print(f"{bs:>6} " + " ".join(f"{row[n] * 1e3:>12.1f} us" for n in names))
    res["inference_ms"] = timings

    # training step: forward + backward + update is where kernel fusion has more to fuse
    print("\ntraining step (forward + backward + SGD), batch 256:")
    xb, yb = torch.randn(256, 64), torch.randint(0, 10, (256,))
    train_t = {}
    for name in ("eager", "torch.compile"):
        if name == "torch.compile" and "torch.compile" not in variants:
            continue
        torch.manual_seed(SEED)
        m = SmallMLP().train()
        step_model = torch.compile(m) if name == "torch.compile" else m
        opt = torch.optim.SGD(m.parameters(), lr=0.01)

        def step():
            opt.zero_grad()
            nn.functional.cross_entropy(step_model(xb), yb).backward()
            opt.step()

        train_t[name] = timed(step, warmup=5, reps=50)
        print(f"  {name:>14}: {train_t[name] * 1e3:.1f} us")
    res["train_step_ms"] = train_t

    # a memory-bound chain of pointwise ops: eager runs one kernel per op and writes every
    # intermediate to memory, the compiler can fuse the whole chain into one loop
    def pointwise(t):
        return (t * torch.sigmoid(t) + 0.5 * torch.tanh(t)) * 1.7 - t.square() * 0.1

    if "torch.compile" in variants:
        print("\npointwise chain (x*sigmoid(x) + 0.5*tanh(x))*1.7 - 0.1*x^2 on 4M floats:")
        xp = torch.randn(4_000_000)
        cp = torch.compile(pointwise)
        with torch.inference_mode():
            assert torch.allclose(cp(xp), pointwise(xp), atol=1e-5)
            pw = {"eager": timed(lambda: pointwise(xp), warmup=5, reps=30),
                  "torch.compile": timed(lambda: cp(xp), warmup=5, reps=30)}
        for k, v in pw.items():
            print(f"  {k:>14}: {v:.2f} ms")
        print(f"  speed-up from fusion: {pw['eager'] / pw['torch.compile']:.1f}x")
        res["pointwise_chain_ms"] = pw

    print("\n-- 6d. Tensor representation and contiguity --")
    big = torch.randn(4096, 4096)
    views = {
        "big": big,
        "big.t()": big.t(),
        "big[:, ::2]": big[:, ::2],
        "big.t().contiguous()": big.t().contiguous(),
    }
    layout = {}
    for name, t in views.items():
        layout[name] = {"shape": list(t.shape), "dtype": str(t.dtype), "stride": list(t.stride()),
                        "contiguous": t.is_contiguous(),
                        "shares_storage": t.untyped_storage().data_ptr() == big.untyped_storage().data_ptr()}
        print(f"  {name:<22} shape={tuple(t.shape)} dtype={t.dtype} stride={t.stride()} "
              f"contiguous={t.is_contiguous()} shares_storage={layout[name]['shares_storage']}")
    res["layout"] = layout

    print("\n  operation speed, contiguous vs non-contiguous view of a 4096x4096 float32 tensor "
          "(median of 20):")
    c, nc = big.t().contiguous(), big.t()          # same values, different memory layout
    w = torch.randn(4096, 256)
    ops = {
        "x + 1 (elementwise)": lambda t: t + 1,
        "x.sum(dim=1)": lambda t: t.sum(dim=1),
        "x.exp()": lambda t: t.exp(),
        "x @ W (4096x256)": lambda t: t @ w,
        "x.reshape(-1)": lambda t: t.reshape(-1),
    }
    speed = {}
    print(f"  {'op':<26}{'contiguous':>12}{'.t() view':>12}{'ratio':>10}")
    for name, op in ops.items():
        tc = timed(lambda: op(c), warmup=3, reps=20)
        tn = timed(lambda: op(nc), warmup=3, reps=20)
        speed[name] = {"contiguous_ms": tc, "noncontiguous_ms": tn, "ratio": tn / tc}
        print(f"  {name:<26}{tc:>9.3f} ms{tn:>9.3f} ms{tn / tc:>9.1f}x")
    tcopy = timed(lambda: nc.contiguous(), warmup=3, reps=20)
    speed["cost of .contiguous() copy"] = {"ms": tcopy}
    print(f"  one-off .contiguous() copy of the view: {tcopy:.2f} ms")
    try:
        nc.view(-1)
    except RuntimeError as e:
        print(f"  nc.view(-1) fails: {str(e).splitlines()[0][:90]}...")
    res["noncontiguous_speed"] = speed
    return res


# ---------------------------------------------------------------- main
class Tee:
    """Write console output to both stdout and results/log.txt."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, open(RESULTS_DIR / "log.txt", "w"))
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"python {platform.python_version()}, torch {torch.__version__}, {platform.platform()}")
    results = {
        "engine_demo": engine_demo(),
        "gradient_check": gradient_check(),
        "moons": moons_experiment(),
        "pytorch": pytorch_internals(),
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
