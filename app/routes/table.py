from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()

# Pydantic model for validation
class TableData(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]

@router.post("/table")
def receive_table(data: TableData):
    # run prediction and return the results
    processed_rows = [
        {**row, "processed": True}
        for row in data.rows
    ]
    return {
        "columns": data.columns + ["processed"],
        "rows": processed_rows
    }
