"""Step 2: train both baselines on a 70/30 stratified split and record test accuracy."""
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score

from common import RESULTS, load_split, make_models, set_seeds


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    print(f"train={X_train.shape} test={X_test.shape}")

    rows = []
    for name, factory in make_models().items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = factory().fit(X_train, y_train)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}", "converged": converged})
        print(f"{name}: accuracy={acc:.4f} converged={converged}")

    pd.DataFrame(rows).to_csv(RESULTS / "baseline_accuracy.csv", index=False)


if __name__ == "__main__":
    main()
