# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import table
from app.routes import predict
from app.services.model_service import model_service

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod'da frontend URL ile sınırla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _load_model_once():
    # Uygulama ayaklanınca modeli tek sefer yükle
    # (dosya çok büyükse burada sadece path kontrol edip ilk istekte de yükleyebilirsin)
    try:
        model_service.load()
        
    except Exception as e:
        # Model yüklenemezse bile servis ilk predict çağrısında tekrar deneyecek.
        print(f"[warn] Model not loaded at startup: {e}")

@app.get("/")
def root():
    return {"message": "Hello from FastAPI!"}

app.include_router(table.router, prefix="/api")
app.include_router(predict.router, prefix="/api")

@app.get("/api/hello")
def say_hello(name: str = "World"):
    return {"message": f"Hello, {name}!"}