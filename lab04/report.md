# Lab 4: Data pipeline and quality

Everything runs with **one command** from the repo root:

```bash
python lab04/src/lab4.py
```

All settings are in [`config.yaml`](config.yaml) (seed 42). The script downloads the data into `data/` (git-ignored) and writes to `results/`:

| File | Content |
|---|---|
| [`validation_report.md`](results/validation_report.md) / `validation.json` | every schema, range, null, consistency and duplicate check with its violation count |
| `preprocessor.joblib` | fitted transformer (feature engineering → impute → clip → scale / one-hot) |
| [`baseline.csv`](results/baseline.csv) | LogisticRegression with and without derived features |
| [`label_audit.json`](results/label_audit.json), `label_sample_50.csv` | duplicate-conflict audit and the 50-row dictionary check |
| [`run_config.yaml`](results/run_config.yaml) | config + library versions + raw and processed SHA256 hashes |
| `data_overview.png`, `log.txt` | missing-value / label-balance plot, full console output |

The data card is [`data_card.md`](data_card.md). Setup: Python 3.11, pandas 2.2.2, scikit-learn 1.5.0, PyYAML 6.0.1. The whole run takes about 20 s on an Apple Silicon Mac.

## 1. Problem definition

The task is to predict whether a person's annual income exceeds **$50K** from 1994 US Census attributes. The **inputs** are the 14 features of the UCI Adult dataset: 6 numeric (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical (workclass, education, marital-status, occupation, relationship, race, sex, native-country). The **output** is a binary label `income ∈ {<=50K, >50K}`, with `>50K` as the positive class.

Because the classes are imbalanced (23.9% positive), accuracy alone is misleading: predicting `<=50K` for everyone already scores 76.1%. The main **success metric** is therefore **F1 on the positive class**, with **accuracy** reported next to it. Both are measured once on a held-out 30% test set that is never used for any fitting decision.

**Constraints.** (1) **No leakage.** The split happens before any cleaning. Every statistic (medians, modes, percentiles, scaler means, one-hot vocabulary) is learned from the training split only, and the test split is never deduplicated or otherwise edited. (2) **Reproducibility.** One command rebuilds everything from the raw download. The seed and all parameters live in a versioned YAML config, and the processed data is identified by a SHA256 hash that must match across runs. (3) **Validation without silent repair.** Bad rows are counted and reported, not dropped. The goal at this stage is a clean, documented dataset and a simple baseline, not a tuned model. The data also carries known demographic biases (see the data card), so the model is for teaching only and must not drive decisions about real people.

## 2. Ingestion

`fetch_openml(name="adult", version=2, as_frame=True)` (OpenML id 1590) returns **48,842 rows × 15 columns**, which the script asserts. The frame is written unchanged to `data/adult/raw.csv` and hashed (SHA256 `8f970e20…adddc`). All later steps work on a copy, and at the end the script asserts that the raw frame still has its original shape and column names. The only changes on the copy are renaming OpenML's `class` column to `income` and restoring the `?` token. OpenML had already decoded `?` to NaN, so the null policy is applied explicitly instead of relying on the loader.

## 3. Validation

The schema from the manual is encoded in `SCHEMA`, `RANGES` and `CATEGORIES`. The categories are copied from UCI `adult.names`, and `?` is allowed for every categorical feature but not for the label. The checks run on all rows **before** the split. Results are in [`results/validation_report.md`](results/validation_report.md):

| Check | Violations |
|---|---:|
| Column set, integer types, allowed categories (incl. label) | 0 |
| Ranges: age 17–90, hours 1–99, capital-gain 0–99999, capital-loss 0–4356 (+ education-num 1–16, fnlwgt > 0) | 0 |
| True NaN after mapping to `?` | 0 |
| `?` counts (info): workclass / occupation / native-country | 2,799 / 2,809 / 857 |
| education ↔ education-num mapping | 0 |
| relationship ↔ sex (Husband ⇒ Male, Wife ⇒ Female) | **4** (rows 5661, 16856, 23390, 43422) |
| capital-gain = 99999 (top-code sentinel) | **244** |
| hours-per-week = 99 (top-code) | 137 |
| Exact duplicate rows | **52** |

There are **0 schema errors**: every value is within the stated types, ranges and dictionary. The real problems are semantic. Four rows contradict themselves on spouse and sex, 244 capital-gain values are a censoring code rather than an amount, and 52 rows are duplicates. Nothing is dropped at this stage. The duplicates are handled in step 4 (train only), and the 99999 code is neutralised by the percentile clip (the train 99th percentile is 15,024).

## 4. Cleaning and transformation (leakage-free)

1. **Split first:** `train_test_split(test_size=0.30, stratify=income, random_state=42)` gives 34,189 train rows and 14,653 test rows. The positive rate is 0.2394 in train and 0.2393 in test.
2. **Duplicates, train only:** 24 exact duplicates are dropped from train, leaving 34,165 rows. The test split keeps its 5 internal duplicates, and 24 test rows also occur in train. This overlap comes from the source data and is reported rather than "fixed", because editing test based on train would itself be a form of leakage.
3. **Preprocessor** (`build_preprocessor`, saved with `joblib.dump` to `results/preprocessor.joblib`), fitted on train only:
   * numeric: `SimpleImputer(median)` → `PercentileClipper(1, 99)` → `StandardScaler`
   * categorical: `SimpleImputer(missing_values="?", most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`

Fitted values read back from the saved file: the medians are age 37, hours 40 and capital-gain 0. The modes are Private / Prof-specialty / United-States. The clip bounds include age [17, 74], hours [8, 80], capital-gain [0, 15024] and capital-loss [0, 1980]. The custom transformers are in `src/pipeline_steps.py`, a separate module so the pickle loads outside the script (`sys.path.insert(0, "lab04/src")`, then `joblib.load(...)`).

## 5. Derived features

`FeatureEngineer` adds `capital_net = capital-gain − capital-loss` and `hours_per_age = hours-per-week / (age − 16)`. Both are pure row-wise formulas with no fitted state, so they cannot leak. They are computed before imputation and clipping, so their clip bounds also come from train. Since age ≥ 17, the denominator is always ≥ 1.

`LogisticRegression(max_iter=2000)` on the processed data, evaluated on the untouched test split:

| Features | Inputs | Test accuracy | Test F1 (>50K) |
|---|---:|---:|---:|
| Original 14 | 105 | 0.8540 | 0.6601 |
| + capital_net, hours_per_age | 107 | **0.8578** | **0.6696** |

The two features add **+0.38 pp accuracy and +0.95 pp F1**. A one-at-a-time ablation (same split and seed, run separately and not part of the script) shows where the gain comes from:

| Derived features added | Test accuracy | Test F1 |
|---|---:|---:|
| only `capital_net` | 0.8542 | 0.6610 |
| only `hours_per_age` | 0.8584 | 0.6713 |

Almost the whole gain comes from `hours_per_age`. `capital_net` is (up to clipping) a linear combination of two columns the model already has, so a linear model gains almost nothing from it. `hours_per_age` is a non-linear interaction between age and hours that LogisticRegression cannot build from the raw columns: 40 hours per week at 20 and at 50 mean very different things. Train accuracy (0.854) is close to test accuracy, so the baseline is not overfitting. The F1 of about 0.67 shows that the minority class is the hard part.

## 6. Label quality

**Duplicate-conflict audit** over the full data, grouping on all 14 features: 54 groups of identical feature rows, of which **5 groups (10 rows) have conflicting income labels**. For example, a 34-year-old female Masters exec-manager working 50 h appears once as `>50K` and once as `<=50K`. These are two different people with identical census answers, or a labelling error. Either way they set a ceiling on achievable accuracy. Without `fnlwgt`, which is a near-unique sampling weight, the count grows to 1,029 groups (3,873 rows). This shows how much label ambiguity the 13 "real" features cannot resolve.

**50-row check against the data dictionary** (`df.sample(50, random_state=42)` → `results/label_sample_50.csv`). Each row was checked with the `dictionary_flags` rules (ranges, categories, education ↔ education-num, spouse ↔ sex, spouse ↔ married, Never-worked ↔ occupation, gain and loss together, age vs years of education). I also read every row by hand. **0 of 50 rows contain impossible values.** 3 rows contain `?`, which is legitimate missingness. Two are unusual but possible: a 25-year-old who is married yet listed as `Own-child`, and a person working 4 hours per week. On the full data the same rules flag **5 rows**: the 4 spouse/sex contradictions from §3 and one 18-year-old with a Masters degree (education-num 14). That is about 0.01%, so label and record noise in Adult is low but not zero.

(Adult has objective labels, so the 24-hour relabel / Cohen's kappa option does not apply.)

## 7. Hashing, config and reproducibility

The processed train and test matrices (107 inputs + label) are written as CSV with a fixed float format and `\n` line endings, so the bytes are deterministic, and then hashed with SHA256:

| | Shape | SHA256 |
|---|---|---|
| train | 34,165 × 108 | `8ca18e7745f7c765b7799acbacbd32ae629a6e3dcfc8cfc29e3ae95a16090c6b` |
| test | 14,653 × 108 | `2f96451ec7c49d28a481fc4fed0ea936fec0363d40a13bba0026a4550513c098` |
| combined | | `ba64a322bed6f35b97c41eba372f05722cf4608d1bb3169c92aa37a8a70fcbeb` |

[`results/run_config.yaml`](results/run_config.yaml) stores the full config, the library versions, the raw hash, the split statistics, these hashes and the baseline metrics. On every run the script compares the new combined hash with the stored one and prints `MATCH` or `DIFFERENT`. I ran the pipeline three times from scratch, deleting `data/` and `results/` each time, and got the same three hashes every time. A fourth run on top of the saved results printed `MATCH` (see `results/log.txt`).
