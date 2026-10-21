# ML Systems Labs — Abil Sultan

Labs for *Introduction to Machine Learning Systems* (V. J. Reddi).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `pyarrow` is pinned to 15.0.2 instead of 16.1.0 because `mlflow==2.14.1` requires `pyarrow<16`.

## Labs

- [lab01](lab01/report.md) — environment and first system measurements
- [lab02](lab02/report.md) — a two-hidden-layer neural network from scratch in NumPy (forward, backprop, gradient check, training, timing)
- [lab03](lab03/report.md) — MLP vs CNN vs attention on Fashion-MNIST: parameters, hand vs thop FLOPs, latency, accuracy trade-offs
