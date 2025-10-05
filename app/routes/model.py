from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from app.services.model_service import model_service

router = APIRouter(prefix="/models", tags=["models"])

class SelectModelRequest(BaseModel):
    model_id: str

@router.get("")
def list_models():
    metas = model_service.list_models()
    return {"models": [m.__dict__ for m in metas]}

@router.post("/select")
def select_model(req: SelectModelRequest):
    p = Path(req.model_id)
    if not p.exists():
        raise HTTPException(404, f"Model not found: {req.model_id}")
    model_service.set_active_model(str(p))
    meta = model_service.active_model_meta()
    return {"ok": True, "active": (meta.__dict__ if meta else None)}

@router.get("/{model_id:path}/stats")
def model_stats(model_id: str, csv: Optional[str] = Query(default="app/data/koi_test.csv")):
    p = Path(model_id)
    if not p.exists():
        raise HTTPException(404, f"Model not found: {model_id}")
    model_service.set_active_model(str(p))
    if not Path(csv).exists():
        raise HTTPException(404, f"CSV not found: {csv}")
    rep = model_service.evaluate_on_csv(csv)
    meta = rep.get("model") or {}
    return {
        "model_id": meta.get("path") or str(p),
        "model_name": meta.get("model_name") or p.stem,
        "version": meta.get("version") or "",
        "trained_at": meta.get("trained_at") or "",
        "dataset": rep.get("dataset"),
        "metrics": rep.get("metrics"),
        "per_class": rep.get("per_class"),
        "confusion_matrix": rep.get("confusion_matrix"),
    }
