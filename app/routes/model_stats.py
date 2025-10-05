# app/routes/model_stats.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.model_service import model_service

router = APIRouter(prefix="/models", tags=["models"])


# ---------- Request/Response Şemaları ----------

class SelectModelRequest(BaseModel):
    model_id: str  # örn: "app/models/xgb_koi_star_son.joblib"


class ModelStatsPayload(BaseModel):
    model_id: str
    model_name: str
    version: str
    trained_at: str
    dataset: str
    metrics: Dict[str, Optional[float]]
    per_class: List[Dict[str, Any]]
    confusion_matrix: List[List[int]]


# ---------- Endpoints ----------

@router.get("")
def list_models():
    """
    Mevcut .joblib modellerini listeler.
    """
    metas = model_service.list_models()
    return {"models": [m.__dict__ for m in metas]}


@router.post("/select")
def select_model(req: SelectModelRequest):
    """
    Aktif modeli değiştirir.
    """
    p = Path(req.model_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {req.model_id}")

    try:
        model_service.set_active_model(req.model_id)
        meta = model_service.active_model_meta()
        return {"ok": True, "active": (meta.__dict__ if meta else None)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{model_id:path}/stats", response_model=ModelStatsPayload)
def model_stats(
    model_id: str,
    csv: Optional[str] = Query(default="app/data/koi_test.csv", description="Test edilecek CSV yolu"),
):
    """
    Belirtilen model ile belirtilen CSV üzerinde değerlendirme çalıştırır.
    Path parametresi 'model_id:path' şeklinde tanımlı; içeride '/' olmasını destekler.
    """
    model_path = Path(model_id)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    csv_path = Path(csv)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv}")

    try:
        # modeli yükle ve değerlendir
        model_service.set_active_model(str(model_path))
        report = model_service.evaluate_on_csv(str(csv_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Evaluation error: {e}")

    # ModelStatsPayload ile birebir uyumlu döndür
    meta = report.get("model") or {}
    payload: Dict[str, Any] = {
        "model_id": meta.get("path") or str(model_path),
        "model_name": meta.get("model_name") or model_path.stem,
        "version": meta.get("version") or "",
        "trained_at": meta.get("trained_at") or "",
        "dataset": report.get("dataset") or str(csv_path),
        "metrics": report.get("metrics") or {},
        "per_class": report.get("per_class") or [],
        "confusion_matrix": report.get("confusion_matrix") or [],
    }
    return payload
