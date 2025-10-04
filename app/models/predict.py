# app/models/predict.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, RootModel

# İstersen tek tek alanları tipleyebilirdik ama dinamik feature seti için serbest bırakıyoruz.
class PredictItem(RootModel[Dict[str, Any]]):
    pass

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(default_factory=list)
    strict: bool = False

class PredictResponse(BaseModel):
    ok: bool
    n: Optional[int] = None
    classes: Optional[List[Any]] = None
    expected_columns: Optional[List[str]] = None
    missing_received: Optional[List[str]] = None
    unexpected_received: Optional[List[str]] = None
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
