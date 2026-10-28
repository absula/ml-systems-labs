"""
Lab 4: Data pipeline and quality (Ch. 5-6)

Leakage-free pipeline for the UCI Adult dataset: ingestion, schema/range validation,
split-then-clean, train-only imputation/clipping/deduplication, derived features,
LogisticRegression baseline, label-quality audit, SHA256 dataset hashing.

Run (from the repo root):  python lab04/src/lab4.py
Settings come from lab04/config.yaml. Outputs go to lab04/results/.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_steps import FeatureEngineer, PercentileClipper, to_missing_token  # noqa: E402

LAB_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_DIR.parent
RESULTS_DIR = LAB_DIR / "results"
DATA_DIR = REPO_ROOT / "data" / "adult"          # git-ignored
CONFIG = yaml.safe_load((LAB_DIR / "config.yaml").read_text())
SEED = CONFIG["seed"]
TARGET = CONFIG["dataset"]["target"]
POS = CONFIG["dataset"]["positive_label"]
MISSING = CONFIG["cleaning"]["missing_token"]

# ---------------------------------------------------------------- schema (UCI adult.names)
SCHEMA = {
    "age": "int", "workclass": "category", "fnlwgt": "int", "education": "category",
    "education-num": "int", "marital-status": "category", "occupation": "category",
    "relationship": "category", "race": "category", "sex": "category",
    "capital-gain": "int", "capital-loss": "int", "hours-per-week": "int",
    "native-country": "category", TARGET: "category",
}
RANGES = {"age": (17, 90), "hours-per-week": (1, 99),
          "capital-gain": (0, 99999), "capital-loss": (0, 4356)}
CATEGORIES = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                  "State-gov", "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                  "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th",
                  "Doctorate", "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
                       "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
                     "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
                       "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
                       "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
                       "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
                       "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
                       "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
                       "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru",
                       "Hong", "Holand-Netherlands"],
    TARGET: ["<=50K", ">50K"],
}
# '?' is an allowed value of every categorical feature (null policy); never of the label
for col in CATEGORIES:
    if col != TARGET:
        CATEGORIES[col] = CATEGORIES[col] + [MISSING]
EDU_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
           "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
           "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}
NUM_COLS = [c for c, t in SCHEMA.items() if t == "int"]
CAT_COLS = [c for c, t in SCHEMA.items() if t == "category" and c != TARGET]
FEATURES = [c for c in SCHEMA if c != TARGET]
DERIVED = CONFIG["features"]["derived"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 2. Ingestion
def ingest():
    bunch = fetch_openml(name=CONFIG["dataset"]["name"], version=CONFIG["dataset"]["version"],
                         as_frame=True, data_home=REPO_ROOT / "data", parser="auto")
    raw = bunch.frame
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "raw.csv"
    raw.to_csv(raw_path, index=False)          # untouched snapshot of what was downloaded
    print(f"Ingested: {raw.shape[0]} rows x {raw.shape[1]} columns  "
          f"(OpenML id {bunch.details['id']}, license {bunch.details.get('licence')})")
    assert raw.shape == (48842, 15), raw.shape
    # Work on a copy so the raw frame is never modified; only rename OpenML's "class".
    df = raw.copy().rename(columns={"class": TARGET})
    df = to_missing_token(df, CAT_COLS, MISSING)   # OpenML decodes '?' to NaN; restore it
    df[TARGET] = df[TARGET].astype(str)
    return raw, df, {"rows": int(raw.shape[0]), "columns": int(raw.shape[1]),
                     "openml_id": int(bunch.details["id"]),
                     "raw_sha256": sha256_file(raw_path)}


# ---------------------------------------------------------------- 3. Validation
def validate(df):
    """Run every check on the full raw data and report violations. Nothing is dropped."""
    checks = []

    def add(name, column, rule, mask, kind="error"):
        idx = df.index[mask]
        checks.append({"check": name, "column": column, "rule": rule, "severity": kind,
                       "violations": int(mask.sum()), "example_rows": idx[:5].tolist()})

    # columns and types
    missing_cols = sorted(set(SCHEMA) - set(df.columns))
    extra_cols = sorted(set(df.columns) - set(SCHEMA))
    checks.append({"check": "columns", "column": "*", "rule": "exactly the 15 schema columns",
                   "severity": "error", "violations": len(missing_cols) + len(extra_cols),
                   "example_rows": missing_cols + extra_cols})
    for col, t in SCHEMA.items():
        if t == "int":
            ok = pd.api.types.is_integer_dtype(df[col])
            checks.append({"check": "type", "column": col, "rule": "integer dtype",
                           "severity": "error", "violations": 0 if ok else len(df),
                           "example_rows": []})
        else:
            add("allowed_values", col, "value in data dictionary" +
                (" (incl. '?')" if col != TARGET else ""), ~df[col].isin(CATEGORIES[col]))

    # ranges
    for col, (lo, hi) in RANGES.items():
        add("range", col, f"{lo} <= x <= {hi}", ~df[col].between(lo, hi))
    add("range", "education-num", "1 <= x <= 16", ~df["education-num"].between(1, 16))
    add("range", "fnlwgt", "x > 0", ~(df["fnlwgt"] > 0))

    # null policy: true nulls are not allowed, '?' is counted (informational)
    for col in SCHEMA:
        add("null", col, "no NaN (missing must be '?')", df[col].isna())
    for col in CAT_COLS:
        add("missing_token", col, "count of '?'", df[col] == MISSING, kind="info")

    # cross-field consistency (data dictionary)
    add("consistency", "education/education-num", "education-num matches education level",
        df["education"].map(EDU_NUM) != df["education-num"])
    add("consistency", "relationship/sex", "Husband => Male, Wife => Female",
        ((df["relationship"] == "Husband") & (df["sex"] != "Male"))
        | ((df["relationship"] == "Wife") & (df["sex"] != "Female")), kind="warning")
    add("consistency", "relationship/marital-status", "Husband/Wife => Married-*",
        df["relationship"].isin(["Husband", "Wife"])
        & ~df["marital-status"].str.startswith("Married"), kind="warning")
    add("suspicious_value", "capital-gain", "99999 is a top-code / sentinel",
        df["capital-gain"] == 99999, kind="warning")
    add("suspicious_value", "hours-per-week", "99 hours is a top-code",
        df["hours-per-week"] == 99, kind="info")
    add("duplicates", "*", "exact duplicate rows (all 15 columns)", df.duplicated(),
        kind="warning")
    return checks


def write_validation_report(checks, ingest_info):
    (RESULTS_DIR / "validation.json").write_text(json.dumps(checks, indent=2))
    lines = ["# Validation report", "",
             f"Data: UCI Adult (OpenML id {ingest_info['openml_id']}), "
             f"{ingest_info['rows']} rows x {ingest_info['columns']} columns, "
             f"raw SHA256 `{ingest_info['raw_sha256']}`.", "",
             "Checks run on the full raw data before the split. Rows are **reported, never "
             "dropped**. Severity: `error` = schema violation, `warning` = plausible data "
             "issue, `info` = expected, counted for documentation.", "",
             "| Check | Column | Rule | Severity | Violations | Example rows |",
             "|---|---|---|---|---:|---|"]
    for c in checks:
        ex = ", ".join(map(str, c["example_rows"])) if c["violations"] else ""
        lines.append(f"| {c['check']} | {c['column']} | {c['rule']} | {c['severity']} | "
                     f"{c['violations']} | {ex} |")
    errors = sum(c["violations"] for c in checks if c["severity"] == "error")
    lines += ["", f"**Schema errors: {errors}.** "
              f"Warnings: {sum(1 for c in checks if c['severity'] == 'warning' and c['violations'])} "
              "checks with hits (see table)."]
    (RESULTS_DIR / "validation_report.md").write_text("\n".join(lines) + "\n")
    return errors


# ---------------------------------------------------------------- 4. Split and clean
def split(df):
    train, test = train_test_split(
        df, test_size=CONFIG["split"]["test_size"], random_state=SEED,
        stratify=df[TARGET] if CONFIG["split"]["stratify"] else None)
    n_before = len(train)
    if CONFIG["cleaning"]["drop_train_duplicates"]:
        train = train.drop_duplicates()        # train only; test stays untouched
    info = {"train_rows_before_dedup": n_before, "train_duplicates_dropped": n_before - len(train),
            "train_rows": len(train), "test_rows": len(test),
            "test_duplicates_kept": int(test.duplicated().sum()),
            # same row on both sides of the split (exists in the source data, not created by us)
            "test_rows_also_in_train": int(test.merge(train, how="inner").shape[0]),
            "train_pos_rate": float((train[TARGET] == POS).mean()),
            "test_pos_rate": float((test[TARGET] == POS).mean())}
    print(f"Split 70/30 stratified: train {n_before} -> {len(train)} after dropping "
          f"{info['train_duplicates_dropped']} exact duplicates; test {len(test)} (untouched, "
          f"{info['test_duplicates_kept']} duplicates kept, "
          f"{info['test_rows_also_in_train']} test rows also appear in train)")
    print(f"Positive rate: train {info['train_pos_rate']:.4f}, test {info['test_pos_rate']:.4f}")
    return train.reset_index(drop=True), test.reset_index(drop=True), info


def build_preprocessor(with_derived):
    low, high = CONFIG["cleaning"]["clip_percentiles"]
    num_cols = NUM_COLS + (DERIVED if with_derived else [])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy=CONFIG["cleaning"]["numeric_impute"])),
        ("clip", PercentileClipper(low, high)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(missing_values=MISSING,
                                 strategy=CONFIG["cleaning"]["categorical_impute"])),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("features", FeatureEngineer(enabled=with_derived)),
        ("columns", ColumnTransformer([("num", numeric, num_cols),
                                       ("cat", categorical, CAT_COLS)])),
    ])


# ---------------------------------------------------------------- 5. Baseline with/without features
def baseline(train, test):
    y_tr = (train[TARGET] == POS).astype(int)
    y_te = (test[TARGET] == POS).astype(int)
    rows, fitted = [], {}
    for with_derived in (False, True):
        pre = build_preprocessor(with_derived)
        X_tr = pre.fit_transform(train[FEATURES])      # fit on train only
        X_te = pre.transform(test[FEATURES])
        clf = LogisticRegression(max_iter=CONFIG["baseline"]["max_iter"], random_state=SEED)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        r = {"features": "with derived" if with_derived else "original 14",
             "n_inputs": X_tr.shape[1],
             "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
             "test_f1": round(float(f1_score(y_te, pred)), 4),
             "train_accuracy": round(float(accuracy_score(y_tr, clf.predict(X_tr))), 4)}
        print(f"LogReg [{r['features']:>12}]  inputs={r['n_inputs']}  "
              f"test acc={r['test_accuracy']:.4f}  F1={r['test_f1']:.4f}")
        rows.append(r)
        fitted[with_derived] = (pre, X_tr, X_te, y_tr, y_te)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "baseline.csv", index=False)
    return rows, fitted[True]


# ---------------------------------------------------------------- 6. Label quality
def dictionary_flags(row):
    flags = []
    for col, (lo, hi) in RANGES.items():
        if not lo <= row[col] <= hi:
            flags.append(f"{col} out of range")
    for col in CATEGORIES:
        if row[col] not in CATEGORIES[col]:
            flags.append(f"{col} not in dictionary")
    if EDU_NUM.get(row["education"]) != row["education-num"]:
        flags.append("education-num mismatch")
    if row["relationship"] == "Husband" and row["sex"] != "Male":
        flags.append("Husband but not Male")
    if row["relationship"] == "Wife" and row["sex"] != "Female":
        flags.append("Wife but not Female")
    if row["relationship"] in ("Husband", "Wife") and not row["marital-status"].startswith("Married"):
        flags.append("spouse relationship but not married")
    if row["workclass"] == "Never-worked" and row["occupation"] != MISSING:
        flags.append("Never-worked with an occupation")
    if row["capital-gain"] > 0 and row["capital-loss"] > 0:
        flags.append("both capital gain and loss")
    if row["age"] - 5 < row["education-num"]:
        flags.append("too young for education level")
    return "; ".join(flags)


def label_audit(df):
    # duplicate-conflict audit: identical 14-feature rows with different income labels
    def conflicts(cols):
        g = df.groupby(cols, observed=True, dropna=False)[TARGET].agg(["nunique", "size"])
        bad = g[g["nunique"] > 1]
        return {"groups": int(len(bad)), "rows": int(bad["size"].sum())}

    all14 = conflicts(FEATURES)
    no_fnlwgt = conflicts([c for c in FEATURES if c != "fnlwgt"])
    dup_groups = int((df.groupby(FEATURES, observed=True, dropna=False).size() > 1).sum())

    sample = df.sample(CONFIG["label_audit"]["sample_size"], random_state=SEED).copy()
    sample["dictionary_flags"] = sample.apply(dictionary_flags, axis=1)
    sample.to_csv(RESULTS_DIR / "label_sample_50.csv", index_label="row")
    full_flags = df.apply(dictionary_flags, axis=1)
    audit = {
        "duplicate_feature_groups": dup_groups,
        "conflicting_label_groups_14_features": all14,
        "conflicting_label_groups_without_fnlwgt": no_fnlwgt,
        "sample_rows_checked": len(sample),
        "sample_rows_flagged": int((sample["dictionary_flags"] != "").sum()),
        "sample_flags": sample.loc[sample["dictionary_flags"] != "", "dictionary_flags"].to_dict(),
        "sample_rows_with_missing_token": int((sample[CAT_COLS] == MISSING).any(axis=1).sum()),
        "full_data_rows_flagged": int((full_flags != "").sum()),
        "full_data_flag_counts": full_flags[full_flags != ""].str.split("; ").explode()
                                 .value_counts().to_dict(),
    }
    audit["sample_flags"] = {str(k): v for k, v in audit["sample_flags"].items()}
    (RESULTS_DIR / "label_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"Label audit: {dup_groups} duplicate feature groups, "
          f"{all14['groups']} with conflicting labels ({all14['rows']} rows); "
          f"without fnlwgt: {no_fnlwgt['groups']} groups ({no_fnlwgt['rows']} rows)")
    print(f"50-row dictionary check: {audit['sample_rows_flagged']} rows flagged; "
          f"full data: {audit['full_data_rows_flagged']} rows flagged")
    return audit


# ---------------------------------------------------------------- 7. Hash, config, artefacts
def save_processed(pre, X_tr, X_te, y_tr, y_te):
    names = pre.named_steps["columns"].get_feature_names_out()
    hashes = {}
    for split_name, X, y in (("train", X_tr, y_tr), ("test", X_te, y_te)):
        out = pd.DataFrame(X, columns=names)
        out[TARGET] = y.to_numpy()
        path = DATA_DIR / f"processed_{split_name}.csv"
        # fixed float format and line endings so the bytes (and the hash) are deterministic
        out.to_csv(path, index=False, float_format="%.8f", lineterminator="\n")
        hashes[split_name] = {"path": str(path.relative_to(REPO_ROOT)),
                              "shape": list(out.shape), "sha256": sha256_file(path)}
    combined = hashlib.sha256((hashes["train"]["sha256"] + hashes["test"]["sha256"]).encode())
    hashes["combined_sha256"] = combined.hexdigest()
    joblib.dump(pre, RESULTS_DIR / "preprocessor.joblib")
    print(f"Processed train {hashes['train']['shape']} sha256={hashes['train']['sha256']}")
    print(f"Processed test  {hashes['test']['shape']} sha256={hashes['test']['sha256']}")
    return hashes


def write_run_config(ingest_info, split_info, hashes, baseline_rows):
    run = {
        "config": CONFIG,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                        "joblib": joblib.__version__, "pyyaml": yaml.__version__},
        "ingestion": ingest_info,
        "split": split_info,
        "processed_data": hashes,
        "baseline": baseline_rows,
    }
    path = RESULTS_DIR / "run_config.yaml"
    previous = yaml.safe_load(path.read_text()) if path.exists() else None
    path.write_text(yaml.safe_dump(run, sort_keys=False))
    if previous:
        same = previous["processed_data"]["combined_sha256"] == hashes["combined_sha256"]
        print(f"Reproducibility check vs previous run: {'MATCH' if same else 'DIFFERENT'} "
              f"({hashes['combined_sha256']})")


def plot_missing(df):
    counts = (df[CAT_COLS] == MISSING).sum().sort_values()
    counts = counts[counts > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].barh(counts.index, counts.values, color="#4C72B0")
    axes[0].set_title("Missing ('?') values per column")
    axes[0].set_xlabel("rows")
    vc = df[TARGET].value_counts()
    axes[1].bar(vc.index, vc.values, color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Label balance")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "data_overview.png", dpi=150)


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
    np.random.seed(SEED)

    print("=== 2. Ingestion ===")
    raw, df, ingest_info = ingest()

    print("\n=== 3. Validation ===")
    checks = validate(df)
    errors = write_validation_report(checks, ingest_info)
    for c in checks:
        if c["violations"] and c["check"] != "missing_token":
            print(f"  [{c['severity']:>7}] {c['check']:<16} {c['column']:<28} {c['violations']}")
    print(f"Schema errors: {errors} (rows are reported, not dropped)")
    plot_missing(df)

    print("\n=== 4-5. Split, clean, features, baseline ===")
    train, test, split_info = split(df)
    baseline_rows, final = baseline(train, test)

    print("\n=== 6. Label quality ===")
    label_audit(df)

    print("\n=== 7. Hashing and config ===")
    hashes = save_processed(*final)
    write_run_config(ingest_info, split_info, hashes, baseline_rows)
    assert raw.shape == (48842, 15) and "class" in raw.columns   # raw frame never modified


if __name__ == "__main__":
    main()
