# src/evaluate.py
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse

# Sabit sınıf isimleri (senin projendeki mapping)
NAMES = ["FALSE POSITIVE","CANDIDATE","CONFIRMED"]
NAME2IDX = {n:i for i,n in enumerate(NAMES)}

def coerce_to_names(series):
    """Seriyi string etiketlere (NAMES) normalize et."""
    s = series.copy()

    # sayısal (int/float/num-string) -> isim
    def to_name(x):
        # 2 -> "CONFIRMED"; "2" -> "CONFIRMED"
        try:
            xi = int(x)
            if 0 <= xi < len(NAMES):
                return NAMES[xi]
        except Exception:
            pass
        # string varyasyonlarını normalize et
        if isinstance(x, str):
            t = x.strip().upper()
            # birkaç varyasyon/typo'ya tolerans
            if t in {"FP","FALSE_POSITIVE","FALSE-POSITIVE","FALSE POSITIVE"}:
                return "FALSE POSITIVE"
            if t in {"CAND","CANDIDATE"}:
                return "CANDIDATE"
            if t in {"CONF","CONFIRMED"}:
                return "CONFIRMED"
        return x  # olduğu gibi bırak (rapor esnasında hata verirse göreceğiz)

    return s.map(to_name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="exo_test.csv", help="Gerçek etiketlerin olduğu CSV (disposition sütunlu)")
    ap.add_argument("--pred", default="predictions.csv", help="Tahmin CSV (prediction veya prediction_idx sütunu)")
    ap.add_argument("--true-col", default="disposition", help="Test CSV’deki gerçek etiket sütunu adı")
    ap.add_argument("--pred-col", default="prediction", help="Pred CSV’deki tahmin sütunu (yoksa prediction_idx denenir)")
    args = ap.parse_args()

    test_path = Path(args.test)
    pred_path = Path(args.pred)

    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV bulunamadı: {test_path.resolve()}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Pred CSV bulunamadı: {pred_path.resolve()}")

    df_true = pd.read_csv(test_path)
    df_pred = pd.read_csv(pred_path)

    # Tahmin sütunu yoksa prediction_idx kullan
    pred_col = args.pred_col if args.pred_col in df_pred.columns else (
        "prediction_idx" if "prediction_idx" in df_pred.columns else None
    )
    if pred_col is None:
        raise ValueError("Pred CSV’de 'prediction' veya 'prediction_idx' sütunu bulunamadı.")

    # Index bazlı hizalama varsayımı: predictions.csv, exo_test.csv ile aynı sırada üretildi
    # (Eğer ayrı bir ID varsa, merge ile birleştirin.)
    if len(df_true) != len(df_pred):
        # Farklı uzunlukta ise yine de min uzunlukla hizalayalım (güvenli değil ama kurtarıcı)
        m = min(len(df_true), len(df_pred))
        df_true = df_true.iloc[:m].copy()
        df_pred = df_pred.iloc[:m].copy()

    y_true_raw = df_true[args.true_col]
    y_pred_raw = df_pred[pred_col]

    # Hepsini string etiketlere dönüştür
    y_true = coerce_to_names(y_true_raw)
    y_pred = coerce_to_names(y_pred_raw)

    # (İsteğe bağlı) Geçersiz etiketleri filtrele
    mask = y_true.isin(NAMES) & y_pred.isin(NAMES)
    if mask.sum() < len(mask):
        print(f"[warn] {len(mask) - mask.sum()} satır geçersiz etiket nedeniyle dışlandı.")
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=NAMES, target_names=NAMES))

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=NAMES),
                       index=[f"T_{n}" for n in NAMES],
                       columns=[f"P_{n}" for n in NAMES]))

if __name__ == "__main__":
    main()
