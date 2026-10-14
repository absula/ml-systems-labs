"""
Lab 2: A neural network from scratch (Ch. 3)

Two-hidden-layer MLP in pure NumPy:
    input 64 -> Linear(64,W) -> ReLU -> Linear(W,W) -> ReLU -> Linear(W,10) -> softmax

Run (from the repo root):  python lab02/src/lab2.py
Outputs go to lab02/results/ (loss_curves.png, results.json, log.txt).
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42


# ---------------------------------------------------------------------------
# Logging: print to console and to results/log.txt
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ---------------------------------------------------------------------------
# Task 1: Dataset
# ---------------------------------------------------------------------------
def one_hot(y, num_classes=10):
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def load_data():
    digits = load_digits()
    X = digits.data / 16.0  # pixel values 0..16 -> 0..1
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    return X_train, X_test, one_hot(y_train), one_hot(y_test), y_train, y_test


# ---------------------------------------------------------------------------
# Task 2: Parameters
#   W1 (64,W)  b1 (W,)
#   W2 (W,W)   b2 (W,)
#   W3 (W,10)  b3 (10,)
# ---------------------------------------------------------------------------
def init_params(width, seed=SEED, n_in=64, n_out=10):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.01, size=(n_in, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0.0, 0.01, size=(width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0.0, 0.01, size=(width, n_out)),
        "b3": np.zeros(n_out),
    }


def count_params(params):
    return sum(p.size for p in params.values())


# ---------------------------------------------------------------------------
# Task 3: Forward pass, layer by layer
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]   # (B, W)
    a1 = relu(z1)                           # (B, W)
    z2 = a1 @ params["W2"] + params["b2"]  # (B, W)
    a2 = relu(z2)                           # (B, W)
    z3 = a2 @ params["W3"] + params["b3"]  # (B, 10)
    p = softmax(z3)                         # (B, 10)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "p": p}
    return p, cache


# ---------------------------------------------------------------------------
# Task 4: Cross-entropy loss (mean over the batch)
#   L = -(1/B) * sum_i sum_k y_ik * log(p_ik)
# ---------------------------------------------------------------------------
def cross_entropy(p, Y):
    return -np.sum(Y * np.log(np.clip(p, 1e-12, 1.0))) / Y.shape[0]


# ---------------------------------------------------------------------------
# Task 5: Backward pass (chain rule)
#   dL/dz3 = (p - Y) / B                (softmax + cross-entropy combined)
#   dW3 = a2^T dz3,  db3 = sum_rows(dz3),  da2 = dz3 W3^T
#   dz2 = da2 * 1[z2 > 0]
#   dW2 = a1^T dz2,  db2 = sum_rows(dz2),  da1 = dz2 W2^T
#   dz1 = da1 * 1[z1 > 0]
#   dW1 = X^T dz1,   db1 = sum_rows(dz1)
# ---------------------------------------------------------------------------
def backward(cache, Y, params):
    B = Y.shape[0]
    dz3 = (cache["p"] - Y) / B                 # (B, 10)
    dW3 = cache["a2"].T @ dz3                  # (W, 10)
    db3 = dz3.sum(axis=0)                      # (10,)
    da2 = dz3 @ params["W3"].T                 # (B, W)

    dz2 = da2 * (cache["z2"] > 0)              # (B, W)
    dW2 = cache["a1"].T @ dz2                  # (W, W)
    db2 = dz2.sum(axis=0)                      # (W,)
    da1 = dz2 @ params["W2"].T                 # (B, W)

    dz1 = da1 * (cache["z1"] > 0)              # (B, W)
    dW1 = cache["X"].T @ dz1                   # (64, W)
    db1 = dz1.sum(axis=0)                      # (W,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def loss_fn(X, Y, params):
    p, _ = forward(X, params)
    return cross_entropy(p, Y)


def gradient_check(X, Y, params, n_checks=10, eps=1e-5, seed=SEED):
    """Nudge n_checks random weights by +-eps and compare with backprop.

    Weights whose analytic gradient is exactly 0 (dead ReLU units, or W1 rows
    for border pixels that are always 0) are skipped and redrawn: 0 == 0
    would "pass" without testing anything.
    """
    rng = np.random.default_rng(seed)
    _, cache = forward(X, params)
    grads = backward(cache, Y, params)

    weight_names = ["W1", "W2", "W3"]
    rows = []
    while len(rows) < n_checks:
        name = weight_names[rng.integers(len(weight_names))]
        idx = tuple(rng.integers(s) for s in params[name].shape)
        if grads[name][idx] == 0.0:
            continue

        original = params[name][idx]
        params[name][idx] = original + eps
        loss_plus = loss_fn(X, Y, params)
        params[name][idx] = original - eps
        loss_minus = loss_fn(X, Y, params)
        params[name][idx] = original  # restore

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grads[name][idx]
        denom = max(abs(analytic) + abs(numeric), 1e-12)
        rel_err = abs(analytic - numeric) / denom
        rows.append(
            {"param": name, "index": [int(i) for i in idx],
             "analytic": float(analytic), "numeric": float(numeric),
             "rel_error": float(rel_err)}
        )
    return rows


# ---------------------------------------------------------------------------
# Task 6: Training loop (shuffle, mini-batch 32, SGD, per-epoch logging)
# ---------------------------------------------------------------------------
def accuracy(X, y, params):
    p, _ = forward(X, params)
    return float(np.mean(np.argmax(p, axis=1) == y))


def train(X_train, Y_train, X_test, Y_test, y_test, width, lr,
          epochs=50, batch_size=32, seed=SEED, verbose=True):
    params = init_params(width, seed=seed)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        X_sh, Y_sh = X_train[perm], Y_train[perm]
        batch_losses = []
        for start in range(0, n, batch_size):
            Xb = X_sh[start:start + batch_size]
            Yb = Y_sh[start:start + batch_size]
            p, cache = forward(Xb, params)
            batch_losses.append(cross_entropy(p, Yb) * Xb.shape[0])
            grads = backward(cache, Yb, params)
            for k in params:
                params[k] -= lr * grads[k]

        train_loss = sum(batch_losses) / n  # mean loss over the epoch
        test_loss = loss_fn(X_test, Y_test, params)
        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(test_loss))
        if verbose:
            print(f"  [W={width:>2}, lr={lr:<5}] epoch {epoch:>2}/{epochs}  "
                  f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}")

    history["test_acc"] = accuracy(X_test, y_test, params)
    return params, history


def plot_loss_curves(all_histories, widths, lrs, path):
    fig, axes = plt.subplots(1, len(widths), figsize=(12, 4.5), sharey=True)
    colors = {0.001: "#1f77b4", 0.01: "#ff7f0e", 0.1: "#2ca02c"}
    for ax, width in zip(axes, widths):
        for lr in lrs:
            h = all_histories[(width, lr)]
            epochs = np.arange(1, len(h["train_loss"]) + 1)
            ax.plot(epochs, h["train_loss"], color=colors[lr], lw=2,
                    label=f"lr={lr} train (acc {h['test_acc']:.3f})")
            ax.plot(epochs, h["test_loss"], color=colors[lr], lw=1.2, ls="--",
                    label=f"lr={lr} test")
        ax.axhline(np.log(10), color="gray", lw=0.8, ls=":", label="ln(10)")
        ax.annotate("lr=0.001 and lr=0.01 overlap\n(stuck at ln(10) for all 50 epochs)",
                    xy=(5, np.log(10)), xytext=(2, 1.75), fontsize=8,
                    arrowprops={"arrowstyle": "->", "color": "gray"})
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Cross-entropy loss")
    fig.suptitle("Loss curves: SGD, batch 32, 50 epochs (solid = train, dashed = test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 7: Timing forward vs backward (median of 100 runs)
# ---------------------------------------------------------------------------
def time_passes(X, Y, params, runs=100, warmup=10):
    for _ in range(warmup):
        _, cache = forward(X, params)
        backward(cache, Y, params)

    fwd, bwd = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        _, cache = forward(X, params)
        t1 = time.perf_counter()
        backward(cache, Y, params)
        t2 = time.perf_counter()
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
    return float(np.median(fwd)), float(np.median(bwd))


def time_matmuls_only(X, params, runs=100, warmup=10):
    """Time only the matrix products of each pass (same shapes as forward/backward)."""
    B, width = X.shape[0], params["W2"].shape[0]
    rng = np.random.default_rng(0)
    a1, a2 = rng.random((B, width)), rng.random((B, width))
    dz3, dz2, dz1 = rng.random((B, 10)), rng.random((B, width)), rng.random((B, width))

    def fwd():
        X @ params["W1"]; a1 @ params["W2"]; a2 @ params["W3"]

    def bwd():
        a2.T @ dz3; dz3 @ params["W3"].T
        a1.T @ dz2; dz2 @ params["W2"].T
        X.T @ dz1

    out = []
    for f in (fwd, bwd):
        for _ in range(warmup):
            f()
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        out.append(float(np.median(ts)))
    return out


def matmul_flops(batch, width, n_in=64, n_out=10):
    """Multiply-adds in the matrix products of each pass."""
    fwd = batch * (n_in * width + width * width + width * n_out)
    # backward: dW for every layer (3 matmuls) + dA for layers 3 and 2 (2 matmuls)
    bwd = (batch * (n_in * width + width * width + width * n_out)
           + batch * (width * n_out + width * width))
    return fwd, bwd


# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(RESULTS_DIR, "log.txt"))
    results = {}

    # ---- Task 1 ----
    print("=" * 70 + "\nTask 1: Dataset\n" + "=" * 70)
    X_train, X_test, Y_train, Y_test, y_train, y_test = load_data()
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Y_train {Y_train.shape}, Y_test {Y_test.shape}")
    print(f"X range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print("Train class counts:", np.bincount(y_train).tolist())
    print("Test  class counts:", np.bincount(y_test).tolist())
    results["data"] = {"train": list(X_train.shape), "test": list(X_test.shape)}

    # ---- Task 2 ----
    print("\n" + "=" * 70 + "\nTask 2: Network design\n" + "=" * 70)
    results["design"] = {}
    for width in (16, 64):
        params = init_params(width)
        shapes = {k: list(v.shape) for k, v in params.items()}
        print(f"W={width}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in params.items())
              + f"  -> total params = {count_params(params)}")
        results["design"][width] = {"shapes": shapes, "total": count_params(params)}

    # ---- Task 3 ----
    print("\n" + "=" * 70 + "\nTask 3: Forward pass\n" + "=" * 70)
    results["forward_check"] = {}
    for width in (16, 64):
        params = init_params(width)
        p, _ = forward(X_train[:32], params)
        max_dev = float(np.max(np.abs(p.sum(axis=1) - 1.0)))
        ok = p.shape == (32, 10) and max_dev < 1e-6
        print(f"W={width}: output shape {p.shape}, max |row_sum - 1| = {max_dev:.2e}  "
              f"-> {'PASS' if ok else 'FAIL'}")
        assert ok
        results["forward_check"][width] = {"shape": list(p.shape), "max_dev": max_dev}

    # ---- Task 4 ----
    print("\n" + "=" * 70 + "\nTask 4: Cross-entropy sanity check\n" + "=" * 70)
    results["initial_loss"] = {}
    for width in (16, 64):
        params = init_params(width)
        loss = loss_fn(X_train, Y_train, params)
        print(f"W={width}: untrained loss = {loss:.4f}   (ln(10) = {np.log(10):.4f}, "
              f"diff = {abs(loss - np.log(10)):.2e})")
        assert abs(loss - np.log(10)) < 0.01
        results["initial_loss"][width] = float(loss)

    # ---- Task 5 ----
    print("\n" + "=" * 70 + "\nTask 5: Gradient check (eps = 1e-5, threshold 1e-6)\n" + "=" * 70)
    print("Relative error = |analytic - numeric| / (|analytic| + |numeric|), batch of 32.\n"
          "Checked on (a) the trained network (lr=0.1, 50 epochs) where gradients are\n"
          "~1e-3, and (b) at initialisation, where many gradients are ~1e-6 and the\n"
          "check hits the float64 round-off floor of the central difference\n"
          "(~ulp(loss)/(2*eps) = 4.4e-16/2e-5 = 2.2e-11 absolute).")

    def print_check(label, rows, graded=True):
        n_pass = sum(r["rel_error"] < 1e-6 for r in rows)
        print(f"\n{label}")
        print(f"  {'param':<5} {'index':<10} {'analytic':>14} {'numeric':>14} {'rel_err':>10}")
        for r in rows:
            print(f"  {r['param']:<5} {str(tuple(r['index'])):<10} {r['analytic']:>14.6e} "
                  f"{r['numeric']:>14.6e} {r['rel_error']:>10.2e}")
        verdict = ("PASS" if n_pass >= 9 else "FAIL") if graded else "diagnostic only"
        print(f"  -> {n_pass}/10 below 1e-6: {verdict}")
        return n_pass

    results["grad_check"] = {}
    for width in (16, 64):
        trained, _ = train(X_train, Y_train, X_test, Y_test, y_test, width, 0.1, verbose=False)
        rows_t = gradient_check(X_train[:32], Y_train[:32], trained)
        n_t = print_check(f"(a) W={width}, trained network:", rows_t)

        rows_i = gradient_check(X_train[:32], Y_train[:32], init_params(width))
        n_i = print_check(f"(b) W={width}, at initialisation N(0, 0.01):", rows_i, graded=False)
        small = [r for r in rows_i if r["rel_error"] >= 1e-6]
        if small:
            print("     failures have |grad| <= "
                  f"{max(abs(r['analytic']) for r in small):.1e}: round-off floor, not a bug")
        results["grad_check"][width] = {"trained": {"rows": rows_t, "n_pass": n_t},
                                        "init": {"rows": rows_i, "n_pass": n_i}}

    # ---- Task 6 ----
    print("\n" + "=" * 70 + "\nTask 6: Training\n" + "=" * 70)
    widths, lrs = [16, 64], [0.001, 0.01, 0.1]
    all_histories = {}
    for width in widths:
        for lr in lrs:
            _, h = train(X_train, Y_train, X_test, Y_test, y_test, width, lr)
            all_histories[(width, lr)] = h
            print(f"  ==> W={width}, lr={lr}: final test accuracy = {h['test_acc']:.4f}\n")

    plot_path = os.path.join(RESULTS_DIR, "loss_curves.png")
    plot_loss_curves(all_histories, widths, lrs, plot_path)
    print(f"Saved {os.path.relpath(plot_path, REPO_ROOT)}\n")
    print(f"{'W':>4} {'lr':>7} {'final train loss':>17} {'final test loss':>16} {'test acc':>9}")
    for (width, lr), h in all_histories.items():
        print(f"{width:>4} {lr:>7} {h['train_loss'][-1]:>17.4f} "
              f"{h['test_loss'][-1]:>16.4f} {h['test_acc']:>9.4f}")
    results["training"] = {f"W={w},lr={lr}": h for (w, lr), h in all_histories.items()}

    # ---- Task 7 ----
    print("\n" + "=" * 70 + "\nTask 7: Timing (median of 100 runs)\n" + "=" * 70)
    print("Widths 16 and 64 are the lab networks; 256 and 1024 are added only to\n"
          "show the trend as matrix multiplication takes over.\n"
          "'matmul' columns time just the matrix products of each pass.\n")
    print(f"{'W':>5} {'batch':>6} {'fwd us':>9} {'bwd us':>9} {'bwd/fwd':>8} "
          f"{'fwd mm us':>10} {'bwd mm us':>10} {'MACs fwd':>14} {'MACs bwd':>14} {'MAC ratio':>9}")
    results["timing"] = []
    for width in widths + [256, 1024]:
        params = init_params(width)
        for batch in (32, X_train.shape[0]):
            Xb, Yb = X_train[:batch], Y_train[:batch]
            t_f, t_b = time_passes(Xb, Yb, params)
            mm_f, mm_b = time_matmuls_only(Xb, params)
            f_macs, b_macs = matmul_flops(batch, width)
            print(f"{width:>5} {batch:>6} {t_f * 1e6:>9.1f} {t_b * 1e6:>9.1f} {t_b / t_f:>8.2f} "
                  f"{mm_f * 1e6:>10.1f} {mm_b * 1e6:>10.1f} {f_macs:>14,} {b_macs:>14,} "
                  f"{b_macs / f_macs:>9.2f}")
            results["timing"].append(
                {"width": width, "batch": batch, "forward_s": t_f, "backward_s": t_b,
                 "ratio": t_b / t_f, "fwd_matmul_s": mm_f, "bwd_matmul_s": mm_b,
                 "fwd_macs": f_macs, "bwd_macs": b_macs})

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {os.path.relpath(os.path.join(RESULTS_DIR, 'results.json'), REPO_ROOT)}")


if __name__ == "__main__":
    main()
