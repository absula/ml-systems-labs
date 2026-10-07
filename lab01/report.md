# Lab 1: Environment and first system measurements

## 1. Goal

Set up a reproducible Python environment with pinned dependencies, train two baseline classifiers
(logistic regression and random forest) on the Breast Cancer Wisconsin dataset, measure their system
cost (training time, single-sample inference latency, model size, peak memory), and decide on which
deployment targets (Cloud, Edge, Mobile, TinyML) each model could run.

## 2. Method

**Environment.** Python 3.11.8 in an isolated `.venv`, packages from the repo `requirements.txt`.
`src/print_versions.py` prints Python, the platform and every pinned package version; its output is in
`results/versions.txt`. Hardware: Apple Silicon (arm64) laptop, macOS, CPU only.

> Note on pins: the manual pins `pyarrow==16.1.0`, but `mlflow==2.14.1` requires `pyarrow<16`, so pip
> cannot resolve both. I pinned `pyarrow==15.0.2` (latest version mlflow 2.14.1 accepts); `pip check`
> reports no broken requirements. All other versions are exactly as specified.

**Data and models** (`src/common.py`, `src/baselines.py`). `load_breast_cancer(return_X_y=True)`:
569 samples, 30 features. 70/30 stratified split with `random_state=42` → 398 train / 171 test.
Seeds set for `random`, `numpy`, `torch` and `torch.cuda` (42).
Models exactly as specified: `LogisticRegression(max_iter=1000, random_state=42)` and
`RandomForestClassifier(n_estimators=100, random_state=42)`, no preprocessing.

**Measurements** (`src/measure.py`). Each model is measured in its own fresh subprocess so memory
of one model does not leak into the other's numbers.
- *Training time*: 1 warm-up fit, then 5 timed fits (`time.perf_counter`); median reported.
- *Single-sample inference*: `predict` on one test row, 1 warm-up, then 100 timed calls; median reported (p95 for reference).
- *Model size*: `joblib.dump` of the fitted model; since there are two models, files are
  `results/model_LogisticRegression.joblib` and `results/model_RandomForest.joblib`.
- *Peak memory*: two views. (a) Process peak RSS via `psutil.Process().memory_info().rss`, sampled
  every 1 ms in a background thread during one training run and during 100 inferences. (b) Peak heap
  allocated inside the same block via `tracemalloc` (NumPy buffers included) — the memory the model
  itself needs, without the interpreter and libraries.

**Deployment fit.** A model fits a target when its latency ≤ target latency, its size ≤ target model size,
and its memory ≤ target upper memory limit (Cloud has no upper limit). Memory is checked twice:
with the full Python process RSS, and with "model only" = model size + peak inference heap.

Reproduce: `python src/print_versions.py > results/versions.txt`, `python src/baselines.py`, `python src/measure.py` (run from `lab01/`).

## 3. Results

### Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| LogisticRegression | 0.9415 |
| RandomForest | 0.9357 |

Logistic regression raised a `ConvergenceWarning` (lbfgs did not converge in 1000 iterations on unscaled features).

### System cost (`results/measurements.csv`)

| Metric | LogisticRegression | RandomForest |
|---|---|---|
| Training time, median of 5 (s) | 0.1244 | 0.0726 |
| Training time, min–max (s) | 0.1145–0.1751 | 0.0723–0.0736 |
| Inference latency, median of 100 (ms) | 0.024 | 1.154 |
| Inference latency, p95 (ms) | 0.038 | 1.234 |
| Model size (bytes) | 1,103 | 290,905 |
| Model size (KB) | 1.08 | 284.09 |
| Peak process RSS, training (MB) | 280.8 | 265.7 |
| Peak process RSS, inference (MB) | 280.6 | 266.0 |
| Peak heap, training (MB, tracemalloc) | 0.073 | 0.229 |
| Peak heap, inference (MB, tracemalloc) | 0.011 | 0.259 |

For context, an empty Python process uses ~19 MB RSS, Python + NumPy + scikit-learn ~110 MB, and
adding `torch` (imported only to set the required seed) ~233 MB. So ~99% of the process RSS is the
runtime stack, not the model.

### Deployment fit (`results/deployment_fit.csv`)

| Model | Target | Latency | Size | Memory (Python process) | Memory (model only) | Verdict |
|---|---|---|---|---|---|---|
| LogReg | Cloud (≤100 ms, ≤500 MB) | ✅ 0.02 ms | ✅ 1.1 KB | ✅ 281 MB | ✅ | **Fits** |
| LogReg | Edge (≤50 ms, ≤50 MB, ≤1024 MB) | ✅ | ✅ | ✅ 281 MB | ✅ | **Fits** |
| LogReg | Mobile (≤20 ms, ≤10 MB, ≤256 MB) | ✅ | ✅ | ❌ 281 MB | ✅ ~12 KB | **Fits if exported out of Python** |
| LogReg | TinyML (≤10 ms, ≤100 KB, ≤256 KB) | ✅ | ✅ | ❌ | ✅ ~12 KB | **Fits if exported as C code** |
| RF | Cloud | ✅ 1.15 ms | ✅ 284 KB | ✅ 266 MB | ✅ | **Fits** |
| RF | Edge | ✅ | ✅ | ✅ 266 MB | ✅ | **Fits** |
| RF | Mobile | ✅ | ✅ | ❌ 266 MB | ✅ ~0.5 MB | **Fits if exported out of Python** |
| RF | TinyML | ✅ | ❌ 284 KB > 100 KB | ❌ | ❌ ~0.5 MB > 256 KB | **Does not fit** |

**Argument.** Both models are far inside every *latency* budget — even the random forest (1.15 ms) is ~9×
faster than the strictest TinyML limit. Latency is therefore not the deciding factor; size and memory are.
On Cloud and Edge both models fit as they are, with the full Python stack. On Mobile both fail only
because the Python + NumPy + scikit-learn (+ torch) process is above 256 MB; the model itself needs well under 1 MB,
so either model fits Mobile once exported to a lightweight runtime (e.g. ONNX Runtime Mobile or plain native code).
For TinyML, logistic regression is just 30 weights + 1 bias (~124 bytes as float32; the 1.1 KB joblib file is mostly
pickling overhead) and a dot product — it easily fits a microcontroller after exporting the weights to C.
The random forest cannot: its serialized size (284 KB) already exceeds the 100 KB size budget and its
inference memory exceeds 256 KB; it would need pruning (fewer/shallower trees) to be considered.

## 4. Conclusions

1. On this dataset the cheaper model is also the better one: logistic regression has slightly higher
   accuracy (0.9415 vs 0.9357), ~47× lower inference latency and a ~264× smaller file than the random forest.
   A bigger model is not automatically a better system choice.
2. For small tabular models the deployment constraint is the software stack, not the model: the Python process
   takes ~265–280 MB while the models need between ~12 KB and ~0.5 MB. Measuring only process RSS would wrongly rule out
   Mobile for both models, so it is important to separate runtime overhead from model footprint and to choose the
   serving runtime together with the target.
3. Measurement methodology matters: single-shot timings and coarse memory sampling gave noisy or meaningless numbers
   (training time varied by ~50% between runs, and the first memory_profiler attempt captured only one sample), so medians
   over repeated runs, isolated processes per model and fine-grained sampling are needed. Also, the specified
   logistic regression does not converge on unscaled features — adding feature standardization would be the first fix
   if I were allowed to change the baseline.
