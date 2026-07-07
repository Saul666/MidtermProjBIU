"""Train and compare classifiers that predict spread reversion.

Models, from simplest to strongest:
  * base_rate   - always predict the training-set majority class (sanity floor)
  * logistic    - scaled logistic regression (the interpretable baseline)
  * random_forest, gradient_boosting - nonlinear comparison models

All real models share one random seed and are trained only on the training
split produced by ``labeling.time_split``.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .config import CONFIG


def build_models(seed: int = None) -> dict:
    seed = CONFIG.seed if seed is None else seed
    return {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=20,
            class_weight="balanced", random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=seed),
    }


def train_all(train: pd.DataFrame, test: pd.DataFrame, features=None):
    """Fit every model and return (fitted_models, predictions, base_rate)."""
    features = list(features or CONFIG.model.feature_cols)
    X_tr, y_tr = train[features].values, train["reverted"].values
    X_te = test[features].values

    base_rate = float(y_tr.mean())          # P(revert) on training data
    models = build_models()
    fitted, preds = {}, {}

    # naive floor: predict the majority class for everyone
    majority = int(base_rate >= 0.5)
    preds["base_rate"] = {
        "proba": np.full(len(test), base_rate),
        "pred": np.full(len(test), majority, dtype=int),
    }

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= CONFIG.model.decision_threshold).astype(int)
        fitted[name] = model
        preds[name] = {"proba": proba, "pred": pred}

    return fitted, preds, base_rate


def feature_importance(model, features) -> pd.Series:
    """Importance for tree models, |coef| for logistic."""
    features = list(features)
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "named_steps"):
        vals = np.abs(model.named_steps["clf"].coef_[0])
    else:
        vals = np.abs(getattr(model, "coef_", np.zeros(len(features)))[0])
    return pd.Series(vals, index=features).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Tuning and calibration (the "improvement" layer)
# ---------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV


def tune_models(train, features=None, n_splits: int = 4, seed: int = None) -> dict:
    """Grid-search each model with forward-chaining TimeSeriesSplit on the
    training events only. Returns best estimator, CV score and CV std per model.
    Tuning on time-ordered folds (not a random split) is what keeps the choice
    honest for a time series.
    """
    seed = CONFIG.seed if seed is None else seed
    features = list(features or CONFIG.model.feature_cols)
    X, y = train[features].values, train["reverted"].values
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grids = {
        "logistic": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
            {"clf__C": [0.1, 0.3, 1.0, 3.0]}),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
            {"n_estimators": [300], "max_depth": [3, 4, 5], "min_samples_leaf": [10, 20, 40]}),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {"n_estimators": [150, 300], "max_depth": [2, 3], "learning_rate": [0.03, 0.05]}),
    }
    out = {}
    for name, (est, grid) in grids.items():
        gs = GridSearchCV(est, grid, cv=tscv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X, y)
        i = gs.best_index_
        out[name] = {
            "estimator": gs.best_estimator_,
            "cv_score": float(gs.best_score_),
            "cv_std": float(gs.cv_results_["std_test_score"][i]),
            "params": gs.best_params_,
        }
    return out


def calibrate(estimator, train, features=None, method: str = "sigmoid", n_splits: int = 4):
    """Wrap an estimator in time-aware probability calibration, fit on train.

    Calibrated probabilities are what make a decision threshold meaningful, which
    matters because the strategy trades on P(revert) crossing a threshold.
    """
    features = list(features or CONFIG.model.feature_cols)
    X, y = train[features].values, train["reverted"].values
    cal = CalibratedClassifierCV(estimator, method=method, cv=TimeSeriesSplit(n_splits))
    cal.fit(X, y)
    return cal
