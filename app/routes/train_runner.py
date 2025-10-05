# app/routes/train_runner.py
from __future__ import annotations
from pathlib import Path
import os, sys, subprocess
from app.state.job_store import update_job, append_logs

def to_rel(p: str | Path) -> str:
    project_root = Path(__file__).resolve().parents[2]
    try:
        return str(Path(p).resolve().relative_to(project_root))
    except ValueError:
        return str(Path(p).name)

def run_training(job_id: str, csv_path: str, base_model_path: str, lr: float, extra: int, hp: dict):
    try:
        update_job(job_id, status="running", progress=25)
        append_logs(job_id, ["training started"])

        project_root = Path(__file__).resolve().parents[2]
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"xgb_koi_finetuned_{job_id}.joblib"

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH','')}"

        cmd = [
            sys.executable, "-m", "app.models.train_koi",
            "--train_csv", csv_path,
            "--group_split_by_kepid",
            "--base_model", base_model_path,
            "--extra_estimators", str(extra),
            "--learning_rate", str(lr),
            "--model_out", str(out_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root), env=env)

        if proc.stdout:
            append_logs(job_id, [proc.stdout.strip()])
        if proc.stderr:
            append_logs(job_id, [proc.stderr.strip()])

        if proc.returncode != 0:
            update_job(job_id, status="failed", error=f"exit={proc.returncode}")
            return

        # --- manifest ve metrics yollarını bul ---
        model_path = Path(out_path)
        manifest_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
        metrics_path  = model_path.with_suffix(model_path.suffix + ".metrics.json")

        update_job(
            job_id,
            status="completed",
            progress=100,
            model_id=to_rel(model_path),
            manifest_path=to_rel(manifest_path),
            metrics_path=to_rel(metrics_path),
        )

    finally:
        try:
            os.remove(csv_path)
        except Exception:
            pass
