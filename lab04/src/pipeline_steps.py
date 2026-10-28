"""
Custom transformers used inside results/preprocessor.joblib.

They live in their own module (not in lab4.py) so the pickled preprocessor can be
loaded from anywhere with `lab04/src` on the path:

    import sys; sys.path.insert(0, "lab04/src")
    pre = joblib.load("lab04/results/preprocessor.joblib")
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add row-wise derived features. Stateless, so it cannot leak test information."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.enabled:
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
            # age >= 17 in the schema, so the denominator is always >= 1
            X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
        return X


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [p_low, p_high] percentiles learned in fit (train only)."""

    def __init__(self, low=1, high=99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.percentile(X, self.low, axis=0)
        self.upper_ = np.percentile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def to_missing_token(df, cat_cols, token="?"):
    """Represent missing categoricals with the '?' token, as in the original UCI files."""
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype(object).where(df[c].notna(), token).astype(str)
    return df
