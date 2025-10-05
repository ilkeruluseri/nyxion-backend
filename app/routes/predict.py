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
    records = data.rows
    out = model_service.predict_records(records, strict=False)

    if not out.get("ok", False):
        # Hata ayrıntısını aynen döndür ki frontend gösterebilsin
        return {
            "ok": False,
            "error": out.get("error", "Prediction error"),
            "missing": out.get("missing_received"),
            "unexpected": out.get("unexpected_received"),
            "expected_columns": out.get("expected_columns"),
        }

    results = out.get("results", [])
    # Güvenli zip: uzunluk farkı olursa taşmasın
    rows_with_preds = []
    for i, row in enumerate(data.rows):
        pred = results[i] if i < len(results) else {}
        rows_with_preds.append({
            **row,
            "prediction": pred.get("prediction"),
            "confidence": pred.get("confidence"),
        })

    return {
        "ok": True,
        "expected_columns": out.get("expected_columns"),
        "results": results,
        "columns": [*data.columns, "prediction", "confidence"],
        "rows": rows_with_preds,
    }
