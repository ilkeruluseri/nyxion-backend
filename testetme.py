# testetme.py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import joblib

# =========================
# Label eşlemesi (eğitimdeki sıra ile aynı olsun)
# =========================
LABEL_ORDER = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
IDX2LBL = {i: lab for i, lab in enumerate(LABEL_ORDER)}
LBL2IDX = {lab: i for i, lab in enumerate(LABEL_ORDER)}

# =========================
# Yardımcılar
# =========================
def predict_aligned(model, X_df: pd.DataFrame):
    """
    Eğitimde kolon adlarıyla fit edilmiş modeller (özellikle LightGBM) için:
    - Kolonları eğitimdeki sıraya hizalar.
    - Eksik kolonları 0 ile doldurur, fazlaları atar.
    - DataFrame olarak besler ki "valid feature names" uyarısı çıkmasın.
    Döndürür: (y_pred, proba_or_None)
    """
    X_use = X_df.copy()

    # Pipeline ise sınıflandırıcıya inmeyi deneyelim
    clf = model
    if hasattr(model, "named_steps") and isinstance(model.named_steps, dict):
        # yaygın adlar: "clf", "lgbm"
        for key in ["clf", "lgbm"]:
            if key in model.named_steps:
                clf = model.named_steps[key]
                break

    # LightGBM: feature_name_ / feature_name parametreleri olabilir
    feat_names = getattr(clf, "feature_name_", None)
    if feat_names is None:
        feat_names = getattr(clf, "feature_name", None)

    if feat_names is not None:
        cols = list(feat_names)
        for c in cols:
            if c not in X_use.columns:
                X_use[c] = 0
        X_use = X_use.reindex(columns=cols)

    # predict & predict_proba
    y_pred = model.predict(X_use)
    proba = model.predict_proba(X_use) if hasattr(model, "predict_proba") else None
    return y_pred, proba

def normalize_label_types(y_true, y_pred, label_names=None):
    """
    y_true ve y_pred'i aynı tipe getirir.
    - y_true string, y_pred int ise: y_pred -> string
    - y_true int, y_pred string ise: y_pred -> int
    """
    if label_names is None:
        label_names = LABEL_ORDER

    idx2lbl = {i: lab for i, lab in enumerate(label_names)}
    lbl2idx = {lab: i for i, lab in enumerate(label_names)}

    yt = pd.Series(y_true)
    yp = pd.Series(y_pred)

    yt_is_str = yt.dtype == "object"
    yp_is_str = yp.dtype == "object"

    if yt_is_str and not yp_is_str:
        yp = yp.astype(int).map(idx2lbl)
    elif (not yt_is_str) and yp_is_str:
        yp = yp.astype(str).map(lbl2idx)

    return yt.to_numpy(), yp.to_numpy()

def to_numeric_labels(y_arr):
    """AUC için gerekirse string etiketleri 0/1/2'ye çevirir."""
    if pd.Series(y_arr).dtype == "object":
        return np.array([LBL2IDX.get(v, v) for v in y_arr])
    return y_arr

# =========================
# Ana analiz fonksiyonu
# =========================
def diagnose_mission_bias(model, X_test: pd.DataFrame, y_test, mission_col="mission", label_names=None):
    """
    1) Mission-bazlı performans (Kepler vs K2)
    2) Confounding / leakage: Mission-only baseline
    3) Mission shuffle: Mission karışınca metrik düşümü
    + Basit önemler / SHAP (varsa)
    """
    assert mission_col in X_test.columns, f"{mission_col} kolonu X_test içinde olmalı!"

    print("\n====== 1) Mission-bazlı performans ======")
    missions = sorted(X_test[mission_col].astype(str).unique())
    f1_scores = {}

    for m in missions:
        idx = (X_test[mission_col].astype(str) == m)
        if idx.sum() == 0:
            continue
        X_sub = X_test.loc[idx].drop(columns=[mission_col], errors="ignore")
        y_true_sub = y_test[idx]

        y_pred_sub, _ = predict_aligned(model, X_sub)
        y_true_norm, y_pred_norm = normalize_label_types(y_true_sub, y_pred_sub, label_names)

        print(f"\n=== {m.upper()} ===")
        print(classification_report(y_true_norm, y_pred_norm, target_names=label_names, digits=3))
        f1_scores[m] = f1_score(y_true_norm, y_pred_norm, average="macro")

    if len(f1_scores) > 1:
        ms = list(f1_scores.keys())
        print("\nF1 farkları:")
        for i in range(1, len(ms)):
            print(f"  {ms[0]} - {ms[i]}: {f1_scores[ms[0]] - f1_scores[ms[i]]:.3f}")

    # 2) Confounding / leakage – Mission-only baseline
    print("\n====== 2) Confounding / Leakage testleri ======")
    print("\n-- 2A) Yalnızca mission ile baseline --")
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    M = enc.fit_transform(X_test[[mission_col]])
    base = LogisticRegression(max_iter=200)
    base.fit(M, y_test)  # not: demonstratif; istersen train set ile fit edip testte ölç
    y_pred_base = base.predict(M)
    y_true_norm_b, y_pred_norm_b = normalize_label_types(y_test, y_pred_base, label_names)
    base_f1 = f1_score(y_true_norm_b, y_pred_norm_b, average="macro")
    print(f"Mission-only baseline F1: {base_f1:.3f}")
    print("Yorum: Bu skor şaşırtıcı derecede yüksekse (≈0.5+), confounding riski var.")

    # 3) Mission shuffle – ΔF1 / ΔAUC
    print("\n-- 2B) Mission shuffle testi --")
    X_no_m = X_test.drop(columns=[mission_col], errors="ignore")
    X_shuf = X_test.copy()
    X_shuf[mission_col] = np.random.permutation(X_shuf[mission_col].values)
    X_shuf_no_m = X_shuf.drop(columns=[mission_col], errors="ignore")

    y_pred_o, proba_o = predict_aligned(model, X_no_m)
    y_pred_s, proba_s = predict_aligned(model, X_shuf_no_m)

    y_true_norm_o, y_pred_norm_o = normalize_label_types(y_test, y_pred_o, label_names)
    y_true_norm_s, y_pred_norm_s = normalize_label_types(y_test, y_pred_s, label_names)

    f1_o = f1_score(y_true_norm_o, y_pred_norm_o, average="macro")
    f1_s = f1_score(y_true_norm_s, y_pred_norm_s, average="macro")
    print(f"F1 orijinal: {f1_o:.3f} | F1 shuffle: {f1_s:.3f} | ΔF1: {f1_o - f1_s:.3f}")

    if (proba_o is not None) and (proba_s is not None):
        try:
            y_idx_o = to_numeric_labels(y_true_norm_o)
            y_idx_s = to_numeric_labels(y_true_norm_s)
            auc_o = roc_auc_score(y_idx_o, proba_o, multi_class="ovr")
            auc_s = roc_auc_score(y_idx_s, proba_s, multi_class="ovr")
            print(f"AUC orijinal: {auc_o:.3f} | AUC shuffle: {auc_s:.3f} | ΔAUC: {auc_o - auc_s:.3f}")
        except Exception as e:
            print("(AUC hesaplanamadı):", e)

    # 4) Basit önemler / SHAP
    print("\n====== 3) Özellik önemleri ======")
    try:
        import shap  # opsiyonel
        X_no_m_small = X_no_m.copy()
        # SHAP için boyutu küçültmek iyi olur
        if len(X_no_m_small) > 500:
            X_no_m_small = X_no_m_small.sample(500, random_state=42)

        # Ağaç tabanlı ise TreeExplainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_no_m_small)
        except Exception:
            # Genel amaçlı
            explainer = shap.KernelExplainer(lambda z: model.predict_proba(pd.DataFrame(z, columns=X_no_m.columns)),
                                             X_no_m_small.iloc[:100, :])
            shap_values = explainer.shap_values(X_no_m_small.iloc[:200, :])

        print("SHAP değerleri hesaplandı. Notebook'ta `shap.summary_plot(shap_values, X_no_m_small)` ile görselleştirebilirsin.")
    except Exception as e:
        # feature_importances_ veya coef_
        if hasattr(model, "feature_importances_"):
            imps = pd.Series(model.feature_importances_, index=X_no_m.columns).sort_values(ascending=False)
            print("\nTop feature_importances_:")
            print(imps.head(15))
        elif hasattr(model, "coef_"):
            coefs = pd.Series(np.ravel(model.coef_), index=X_no_m.columns).sort_values(key=np.abs, ascending=False)
            print("\nTop |coef|:")
            print(coefs.head(15))
        else:
            print("Önem metrikleri mevcut değil. (SHAP yok) Detay:", e)

    print("\n[✓] Analiz tamamlandı.")


# =========================
# MAIN: Dosyaları oku ve çalıştır
# =========================
if __name__ == "__main__":
    # CSV'yi oku
    df = pd.read_csv("data/processed/exo_test.csv")

    # Hedef ve özellikleri ayır
    y_test = df["disposition"]
    X_test = df.drop(columns=["disposition"])

    # mission_onehot'tan mission üret (gerekirse)
    if "mission" not in X_test.columns and "mission_onehot" in X_test.columns:
        X_test["mission"] = X_test["mission_onehot"].map({0: "kepler", 1: "k2"})

    # Modeli yükle
    model = joblib.load("models/sonilk/model.pkl")

    # Analizi çalıştır
    diagnose_mission_bias(
        model,
        X_test,
        y_test,
        mission_col="mission",
        label_names=LABEL_ORDER
    )
