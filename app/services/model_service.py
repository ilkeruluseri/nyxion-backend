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
DEFAULT_MODEL_PATH = Path("app/models/xgb_koi_star_son.joblib")

LABEL_CANDIDATES = {
    "disposition", "koi_disposition", "label", "target", "y", "class", "CLASS", "Disposition"
}



# app/services/model_service.py

# === Feature engineering (train ile birebir) ===
_BASE_FEATURES = [
    "koi_period","koi_duration","koi_depth","koi_prad",
    "koi_steff","koi_slogg","koi_srad","koi_smass",
    "koi_impact","koi_kepmag",
    "koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec",
]

def _build_features_for_inference(df_raw: pd.DataFrame) -> pd.DataFrame:
    X = df_raw.copy()

    # Güvenli tip dönüşümü
    for c in _BASE_FEATURES:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Oran ve log dönüşümleri
    X["ratio_duration_period"] = (X["koi_duration"] / 24.0) / X["koi_period"]
    X["log_period"]   = np.log1p(X["koi_period"])
    X["log_duration"] = np.log1p(X["koi_duration"])
    X["log_depth"]    = np.log1p(X["koi_depth"])
    X["log_prad"]     = np.log1p(X["koi_prad"])

    # rp_over_rstar ~ sqrt(depth_fraction) (depth ppm varsayımıyla)
    frac_depth = pd.to_numeric(X.get("koi_depth", np.nan), errors="coerce") / 1e6
    X["rp_over_rstar"] = np.sqrt(np.clip(frac_depth, a_min=0, a_max=None))

    # Yıldız yoğunluk proxy
    if "koi_smass" in X.columns and "koi_srad" in X.columns:
        X["stellar_density_proxy"] = X["koi_smass"] / (X["koi_srad"] ** 3)
    else:
        X["stellar_density_proxy"] = np.nan

    # Beklenen süre proxy & anomaly
    P = pd.to_numeric(X.get("koi_period", np.nan), errors="coerce")
    Rstar = pd.to_numeric(X.get("koi_srad", np.nan), errors="coerce")
    expected = (P ** (1/3)) / (Rstar.replace(0, np.nan))
    X["duration_expected_proxy"] = expected
    X["duration_anomaly"] = (X["koi_duration"] / 24.0) / expected

    # Train'de kaydedilen kolon listesi self._expected_cols olacak;
    # yine de burada seçilebilir olması için train'deki önerilen seti oluşturuyoruz:
    use_cols = [c for c in _BASE_FEATURES if c in X.columns] + [
        "ratio_duration_period","log_period","log_duration","log_depth","log_prad",
        "rp_over_rstar","stellar_density_proxy","duration_expected_proxy","duration_anomaly",
    ]
    return X[use_cols]




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
        self._imputer: Optional[Any] = None
        self._classes: Optional[List[Any]] = None
        self._expected_cols: Optional[List[str]] = None

    # --- Load model once ---
    def load(self) -> None:
        obj = joblib.load(self.model_path)

        # 1) Model + imputer + features (bundle) ya da tek model ayrımı
        if isinstance(obj, dict) and "model" in obj:
            self._model = obj["model"]
            self._imputer = obj.get("imputer")
            self._expected_cols = obj.get("features")  # <-- ÖNEMLİ: overwrite etme
        else:
            self._model = obj
            self._imputer = None
            self._expected_cols = None  # şimdilik

        # 2) Sınıf isimleri
        self._classes = self._infer_classes(self._model)

        # 3) Eğer bundle features yoksa, modelden çıkarmayı dene
        if not self._expected_cols:
            self._expected_cols = self._infer_expected_features(self._model)
        print(f"[MODEL] path={self.model_path} "
        f"features={len(self._expected_cols) if self._expected_cols else 'None'} "
        f"classes={self._classes}")


        
      


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

        raw = pd.DataFrame.from_records(records)



        # Muhtemel label kolonlarını düş
        present_labels = [c for c in raw.columns if c in LABEL_CANDIDATES]
        if present_labels:
            raw = raw.drop(columns=present_labels)

        try:
            df = _build_features_for_inference(raw)
        except Exception as e:
            return {"ok": False, "error": f"Feature engineering failed: {e}"}

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
            
        if self._imputer is not None:
            try:
                X_arr = self._imputer.transform(X)
            except Exception as e:
                return {"ok": False, "error": f"Imputer transform failed: {e}"}
        else:
            X_arr = X.values

        # Tahmin + olasılıklar
        try:
            y_pred = self._model.predict(X_arr)
            y_proba = self._model.predict_proba(X_arr) if hasattr(self._model, "predict_proba") else None
        except Exception as e:
            return {"ok": False, "error": f"Prediction failed: {e}"}

        # Sonuçları toparla (mevcut mantığınla aynı)
        results: List[Dict[str, Any]] = []
        for i in range(len(X)):
            item: Dict[str, Any] = {"input_index": int(i)}
            if self._classes is not None:
                try:
                    idx = int(y_pred[i])
                    item["prediction_idx"] = idx
                    item["prediction"] = str(self._classes[idx])
                except Exception:
                    item["prediction_idx"] = None
                    item["prediction"] = str(y_pred[i])
            else:
                item["prediction"] = y_pred[i]

            if y_proba is not None:
                proba_row = y_proba[i]
                if self._classes is not None and len(self._classes) == len(proba_row):
                    item["probabilities"] = {str(self._classes[j]): float(proba_row[j]) for j in range(len(proba_row))}
                    top_idx = int(np.argmax(proba_row))
                    item["top_class"] = str(self._classes[top_idx])
                    item["confidence"] = float(proba_row[top_idx])
                else:
                    item["probabilities"] = [float(p) for p in proba_row.tolist()]
                    item["confidence"] = float(np.max(proba_row))
            results.append(item)

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