# app/utils/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_DIRS = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "data" / "splits",
]

def to_rel(p: str | Path) -> str:
    p = Path(p).resolve()
    for base in ALLOWED_DIRS:
        try:
            rel = p.relative_to(base)
            return str(base.name / rel)  # "models/..." veya "data/splits/..."
        except ValueError:
            continue
    # fallback: proje tabanına göre relative
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return p.name
