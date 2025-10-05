# app/routes/train.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, status, HTTPException
import uuid, json, tempfile, os
from pathlib import Path

from app.routes.train_runner import run_training
from app.state.job_store import init_job, update_job, get_job

router = APIRouter(prefix="/train", tags=["train"])

def resolve_base_model(base_model_id: str) -> str:
    p = Path(base_model_id).expanduser()
    if p.exists():
        return str(p.resolve())
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "app" / "models" / f"{base_model_id}.joblib",
        project_root / "app" / "models" / f"{base_model_id}_son.joblib",
        * (project_root / "app" / "models").glob(f"{base_model_id}*.joblib"),
        * (project_root / "models").glob(f"{base_model_id}*.joblib"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError(f"Base model not found for id='{base_model_id}'")

@router.post("", response_model=None, status_code=status.HTTP_202_ACCEPTED)
async def start_train(
    base_model_id: str = Form(...),
    hparams: str       = Form(...),
    dataset_file: UploadFile = File(...),
    background: BackgroundTasks = ...,   # ✅ opsiyonel değil
):
    job_id = str(uuid.uuid4())
    init_job(job_id, status="uploading", progress=0, logs=[])

    # CSV'yi geçici dosyaya yaz
    with tempfile.NamedTemporaryFile(prefix="train_", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
        content = await dataset_file.read()
        tmp.write(content)

    update_job(job_id, status="queued", progress=15, csv_path=tmp_path)

    # Hparams parse
    try:
        hp = json.loads(hparams)
    except json.JSONDecodeError as e:
        update_job(job_id, status="failed", error=f"hparams JSON error: {e}")
        try:
            os.remove(tmp_path)  # 🧹 temizlik
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"hparams JSON error: {e}")

    # Hparamlar
    lr    = float(hp.get("learning_rate", 0.05))
    extra = int(hp.get("extra_estimators", 0))

    # Base model yolunu çöz (ID geldiyse .joblib'e çevir)
    try:
        resolved_base = resolve_base_model(base_model_id)
    except FileNotFoundError as e:
        update_job(job_id, status="failed", error=str(e))
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=str(e))

    # Arka plan işlemi başlat
    background.add_task(run_training, job_id, tmp_path, resolved_base, lr, extra, hp)
    return {"job_id": job_id, "detail": "Training scheduled"}

@router.get("/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)
    return job or {"job_id": job_id, "status": "not_found", "progress": 0}
