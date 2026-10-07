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
