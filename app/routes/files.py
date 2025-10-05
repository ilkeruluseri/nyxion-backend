# app/routes/files.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(prefix="/files", tags=["files"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../nyxion-backend
SAFE_BASES = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "data" / "splits",
]

def _safe_resolve(rel_path: str) -> Path:
    p = (PROJECT_ROOT / rel_path).resolve()
    # path traversal engelle
    if not any(str(p).startswith(str(b.resolve())) for b in SAFE_BASES):
        raise HTTPException(status_code=403, detail="Access denied")
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return p

@router.get("/download")
def download(path: str = Query(..., description="Relative path, e.g. models/xgb...joblib")):
    p = _safe_resolve(path)
    return FileResponse(path=p, filename=p.name, media_type="application/octet-stream")
