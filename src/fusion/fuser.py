import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier

MODALITY_FEATURE_GROUPS: dict[str, list[str]] = {
    "pressure":    ["PS1_", "PS2_", "PS3_", "PS4_", "PS5_", "PS6_"],
    "flow":        ["FS1_", "FS2_"],
    "temperature": ["TS1_", "TS2_", "TS3_", "TS4_"],
    "vibration":   ["VS1_"],
    "power":       ["EPS1_"],
    "efficiency":  ["SE_", "CE_", "CP_"],
}

_LABEL_COLUMNS = ["cooler", "valve", "pump", "accumulator"]


def get_group_columns(df_columns: list[str], prefixes: list[str]) -> list[str]:
    return [c for c in df_columns if any(c.startswith(p) for p in prefixes)]


class EarlyFusionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator) -> None:
        self.base_estimator = base_estimator

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "EarlyFusionClassifier":
        self._model = MultiOutputClassifier(clone(self.base_estimator))
        self._model.fit(X, Y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> list[np.ndarray]:
        return self._model.predict_proba(X)


class LateFusionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator, n_splits: int = 5) -> None:
        self.base_estimator = base_estimator
        self.n_splits = n_splits

    def _group_proba(self, model: MultiOutputClassifier, X_group: np.ndarray) -> np.ndarray:
        # Stack each label's class probabilities, reindexed to the FULL class set so the
        # width is identical regardless of which classes a given fold's training data saw.
        proba_list = model.predict_proba(X_group)
        blocks = []
        for j, proba in enumerate(proba_list):
            full = np.zeros((proba.shape[0], len(self._classes_[j])))
            for col, cls in enumerate(model.estimators_[j].classes_):
                full[:, self._class_pos_[j][cls]] = proba[:, col]
            blocks.append(full)
        return np.hstack(blocks)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "LateFusionClassifier":
        n = len(X)
        groups = list(MODALITY_FEATURE_GROUPS.items())
        all_cols = list(X.columns)

        self._classes_ = [np.unique(Y.iloc[:, j].values) for j in range(Y.shape[1])]
        self._class_pos_ = [{c: i for i, c in enumerate(cls)} for cls in self._classes_]
        width = sum(len(c) for c in self._classes_)  # meta columns contributed per group

        meta_X = np.zeros((n, len(groups) * width))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        self._group_cols_: list[list[str]] = []
        for g_idx, (_, prefixes) in enumerate(groups):
            cols = get_group_columns(all_cols, prefixes)
            self._group_cols_.append(cols)
            X_group = X[cols].values if cols else np.zeros((n, 1))

            block = np.zeros((n, width))
            for train_idx, val_idx in kf.split(X_group):
                model = MultiOutputClassifier(clone(self.base_estimator))
                model.fit(X_group[train_idx], Y.values[train_idx])
                block[val_idx] = self._group_proba(model, X_group[val_idx])
            meta_X[:, g_idx * width : (g_idx + 1) * width] = block

        self._meta_ = MultiOutputClassifier(
            LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        )
        self._meta_.fit(meta_X, Y)

        # final group models on full data for inference
        self._group_models_: list[MultiOutputClassifier] = []
        for g_idx, (_, prefixes) in enumerate(groups):
            cols = self._group_cols_[g_idx]
            X_group = X[cols].values if cols else np.zeros((n, 1))
            model = MultiOutputClassifier(clone(self.base_estimator))
            model.fit(X_group, Y)
            self._group_models_.append(model)

        self._width_ = width
        return self

    def _build_meta_X(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        meta_X = np.zeros((n, len(self._group_models_) * self._width_))
        for g_idx, (model, cols) in enumerate(zip(self._group_models_, self._group_cols_)):
            X_group = X[cols].values if cols else np.zeros((n, 1))
            meta_X[:, g_idx * self._width_ : (g_idx + 1) * self._width_] = self._group_proba(
                model, X_group
            )
        return meta_X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._meta_.predict(self._build_meta_X(X))

    def predict_proba(self, X: pd.DataFrame) -> list[np.ndarray]:
        return self._meta_.predict_proba(self._build_meta_X(X))
