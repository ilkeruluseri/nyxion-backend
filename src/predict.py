# src/predict.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path("models/sonilk/model.pkl")          # dosya adın farklıysa değiştir
INPUT_PATH = Path("data/processed/exo_test.csv")       # test dosyan
OUTPUT_PATH = Path("predictions.csv")

# Eğitimde hiçbiri özellik olarak kullanılmaması gereken "muhtemel label" kolon adları:
LABEL_CANDIDATES = {
    "disposition","koi_disposition","label","target","y","class","CLASS","Disposition"
}

def infer_expected_features(pipe):
    """Pipeline/estimatordan eğitimde beklenen giriş kolonlarını çıkarmaya çalış."""
    # 1) En kolay yol: feature_names_in_ (sklearn >=1.0'da bir çok step set ediyor)
    if hasattr(pipe, "feature_names_in_"):
        return list(pipe.feature_names_in_)
    # 2) Pipeline içindeki adımlardan ilk feature_names_in_ olanı bul
    if hasattr(pipe, "named_steps"):
        # sondan başa taramak genelde daha kolay (preprocess adımı çoğu kez önce)
        for name, step in list(pipe.named_steps.items()):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    # 3) Bulunamazsa None
    return None

def main():
    # 1) Modeli yükle
    pipe = joblib.load(MODEL_PATH)
    print("[info] model loaded:", type(pipe))

    # 2) Test verisini oku
    df = pd.read_csv(INPUT_PATH)
    print("[info] test shape:", df.shape)

    # 3) Etiketi/label’ı varsa düş
    present_labels = [c for c in df.columns if c in LABEL_CANDIDATES]
    if present_labels:
        df = df.drop(columns=present_labels)
        print("[info] dropped label columns:", present_labels)

    # 4) Eğitimde beklenen giriş kolonlarını öğren ve hizala
    expected = infer_expected_features(pipe)
    if expected is not None:
        missing = [c for c in expected if c not in df.columns]
        extra   = [c for c in df.columns if c not in expected]

        # Eksik kolonları NaN olarak ekle (imputer varsa doldurur)
        for m in missing:
            df[m] = np.nan

        if missing:
            print("[warn] created missing columns (NaN):", missing[:15], "..." if len(missing) > 15 else "")
        if extra:
            print("[info] ignoring extra columns not used by model:", extra[:15], "..." if len(extra) > 15 else "")

        # Sıralamayı eğitimdeki gibi yap
        X = df[expected]
    else:
        # Beklenen kolonlar tespit edilemediyse olduğu gibi dene (genelde yeter)
        print("[warn] could not infer expected features; using all columns as-is.")
        X = df

    # 5) Tahmin + (varsa) olasılık
    y_pred = pipe.predict(X)
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X)
    except Exception:
        pass

    # 6) Çıktı DataFrame’i
    out = df.copy()
    out["prediction_idx"] = y_pred

    # sınıf isimlerini yakalayıp eklemeye çalış
    class_names = None
    if hasattr(pipe, "classes_"):
        class_names = list(pipe.classes_)
    elif hasattr(pipe, "named_steps"):
        for _, step in reversed(list(pipe.named_steps.items())):
            if hasattr(step, "classes_"):
                class_names = list(step.classes_)
                break

    if class_names is not None:
        try:
            out["prediction"] = [class_names[int(i)] for i in y_pred]
        except Exception:
            out["prediction"] = y_pred
    else:
        out["prediction"] = y_pred

    if y_proba is not None:
        for i in range(y_proba.shape[1]):
            out[f"proba_{i}"] = y_proba[:, i]

    # 7) Kaydet
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"[ok] saved: {OUTPUT_PATH} | shape={out.shape}")

if __name__ == "__main__":
    main()
