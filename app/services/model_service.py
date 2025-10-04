# app/services/model_service.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import os

# --- MacOS LightGBM fix ---
os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")

# Varsayılan model yolu
DEFAULT_MODEL_PATH = Path("app/models/model.pkl")

LABEL_CANDIDATES = {
    "disposition", "koi_disposition", "label", "target", "y", "class", "CLASS", "Disposition"
}


# === Helper: numpy tiplerini Python'a çevir ===
def _convert_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj


class ModelService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._model: Optional[BaseEstimator] = None
        self._classes: Optional[List[Any]] = None
        self._expected_cols: Optional[List[str]] = None

    # --- Load model once ---
    def load(self) -> None:
        self._model = joblib.load(self.model_path)
        self._classes = self._infer_classes(self._model)
        self._expected_cols = self._infer_expected_features(self._model)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # --- Infer feature names ---
    @staticmethod
    def _infer_expected_features(model: BaseEstimator) -> Optional[List[str]]:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "named_steps"):
            for _, step in list(model.named_steps.items())[::-1]:
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
        return None

    # --- Infer class names ---
    @staticmethod
    def _infer_classes(model: BaseEstimator) -> Optional[List[Any]]:
        if hasattr(model, "classes_"):
            return list(model.classes_)
        if hasattr(model, "named_steps"):
            for _, step in list(model.named_steps.items())[::-1]:
                if hasattr(step, "classes_"):
                    return list(step.classes_)
        return None

    # --- Main prediction logic ---
    def predict_records(self, records: List[Dict[str, Any]], strict: bool = False) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()
        assert self._model is not None

        df = pd.DataFrame.from_records(records)

        # Muhtemel label kolonlarını düş
        present_labels = [c for c in df.columns if c in LABEL_CANDIDATES]
        if present_labels:
            df = df.drop(columns=present_labels)

        # Kolon hizalama
        missing: List[str] = []
        extra: List[str] = []
        X = df
        if self._expected_cols:
            missing = [c for c in self._expected_cols if c not in df.columns]
            extra = [c for c in df.columns if c not in self._expected_cols]
            for m in missing:
                df[m] = np.nan  # imputer varsa doldurur
            X = df[self._expected_cols]

            if strict and (missing or extra):
                return {
                    "ok": False,
                    "error": "Column mismatch",
                    "missing": missing,
                    "unexpected": extra,
                    "expected_columns": self._expected_cols,
                }

        # Tahmin + olasılıklar
        try:
            y_pred = self._model.predict(X)
            y_proba = None
            if hasattr(self._model, "predict_proba"):
                y_proba = self._model.predict_proba(X)
        except Exception as e:
            return {"ok": False, "error": f"Prediction failed: {e}"}

        # Sonuçları toparla
        results: List[Dict[str, Any]] = []
        for i in range(len(X)):
            item: Dict[str, Any] = {"input_index": int(i)}
            # İnsan-okur sınıf etiketi
            if self._classes is not None:
                try:
                    item["prediction"] = self._classes[int(y_pred[i])]
                except Exception:
                    item["prediction"] = y_pred[i]
            else:
                item["prediction"] = y_pred[i]

            # Olasılıklar
            if y_proba is not None:
                proba_row = y_proba[i]
                if self._classes is not None and len(self._classes) == len(proba_row):
                    item["probabilities"] = {
                        str(self._classes[j]): float(proba_row[j]) for j in range(len(proba_row))
                    }
                    top_idx = int(np.argmax(proba_row))
                    item["top_class"] = str(self._classes[top_idx])
                    item["confidence"] = float(proba_row[top_idx])
                else:
                    item["probabilities"] = [float(p) for p in proba_row.tolist()]
                    item["confidence"] = float(np.max(proba_row))
            results.append(item)

        # === JSON'a uygun dönüşüm ===
        return _convert_numpy({
            "ok": True,
            "n": len(results),
            "classes": self._classes,
            "expected_columns": self._expected_cols,
            "missing_received": missing,
            "unexpected_received": extra,
            "results": results,
        })


# Singleton instance
model_service = ModelService()
