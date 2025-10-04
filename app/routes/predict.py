from fastapi import APIRouter
from app.services.model_service import model_service
from app.models.predict import PredictResponse
from app.routes.table import TableData  # zaten mevcut

router = APIRouter(prefix="/predict", tags=["predict"])

@router.post("", response_model=PredictResponse)
def predict_api(req: dict):
    # Eski raw predict endpoint (records listesi) burada zaten var
    records = req.get("records", [])
    strict = req.get("strict", False)
    out = model_service.predict_records(records, strict=strict)
    return out


# --- yeni endpoint: frontend table datası ile çalışsın ---
@router.post("/from-table")
def predict_from_table(data: TableData):
    """
    Frontend'in gönderdiği table datasını (columns, rows)
    modelin beklediği formata çevirir ve tahmin yapar.
    """
    records = data.rows
    out = model_service.predict_records(records, strict=False)
    return {
        "ok": out.get("ok", False),
        "expected_columns": out.get("expected_columns"),
        "results": out.get("results"),
        "columns": data.columns + ["prediction", "confidence"],
        "rows": [
            {**row, **{
                "prediction": res.get("prediction"),
                "confidence": res.get("confidence")
            }}
            for row, res in zip(data.rows, out.get("results", []))
        ]
    }
