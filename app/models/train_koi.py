#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, hashlib, datetime

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from xgboost import XGBClassifier

# ================================
# Sabitler
# ================================
LABEL_MAP = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
IDX2LBL = {v: k for k, v in LABEL_MAP.items()}
RANDOM_STATE = 42

# ================================
# 1) Yardımcılar
# ================================
def load_base_model(path: str | Path) -> XGBClassifier:
    """Joblib'ten XGBClassifier (veya {'model': XGBClassifier, ...}) yükler."""
    p = Path(path)
    obj = joblib.load(p)
    if isinstance(obj, dict) and "model" in obj:
        base = obj["model"]
    elif isinstance(obj, XGBClassifier):
        base = obj
    else:
        raise ValueError("Base model joblib dosyası bekleneni içermiyor ('model' veya XGBClassifier).")
    if not isinstance(base, XGBClassifier):
        raise ValueError("Base model XGBoost değil.")
    return base

def load_koi_df(csv_path: str, dedup=True) -> pd.DataFrame:
    """KOI/cumulative CSV'yi okur. '#' ile başlayan yorum satırlarını atar. İsteğe bağlı tekilleştirme yapar."""
    df = pd.read_csv(csv_path, comment="#")
    if "koi_disposition" not in df.columns:
        raise ValueError("CSV içinde 'koi_disposition' kolonu bulunamadı!")

    if dedup:
        key = "kepoi_name" if "kepoi_name" in df.columns else ("kepid" if "kepid" in df.columns else None)
        if key:
            df = df.sort_values(key).drop_duplicates(subset=[key], keep="first")

    df = df[df["koi_disposition"].isin(LABEL_MAP.keys())].copy()
    return df

BASE_FEATURES = [
    "koi_period","koi_duration","koi_depth","koi_prad",
    "koi_steff","koi_slogg","koi_srad","koi_smass",
    "koi_impact","koi_kepmag",
    "koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # güvenli tip dönüşümü
    for c in BASE_FEATURES:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # oran ve log dönüşümleri
    X["ratio_duration_period"] = (X["koi_duration"] / 24.0) / X["koi_period"]
    X["log_period"]   = np.log1p(X["koi_period"])
    X["log_duration"] = np.log1p(X["koi_duration"])
    X["log_depth"]    = np.log1p(X["koi_depth"])
    X["log_prad"]     = np.log1p(X["koi_prad"])

    # rp_over_rstar ~ sqrt(depth_fraction) (depth ppm varsayımıyla)
    frac_depth = pd.to_numeric(X.get("koi_depth", np.nan), errors="coerce") / 1e6
    X["rp_over_rstar"] = np.sqrt(np.clip(frac_depth, a_min=0, a_max=None))

    # yıldız yoğunluk proxy ~ M / R^3
    if "koi_smass" in X.columns and "koi_srad" in X.columns:
        X["stellar_density_proxy"] = X["koi_smass"] / (X["koi_srad"] ** 3)
    else:
        X["stellar_density_proxy"] = np.nan

    # beklenen süre proxy & anomaly (kabaca)
    P = pd.to_numeric(X.get("koi_period", np.nan), errors="coerce")
    Rstar = pd.to_numeric(X.get("koi_srad", np.nan), errors="coerce")
    expected = (P ** (1/3)) / (Rstar.replace(0, np.nan))
    X["duration_expected_proxy"] = expected
    X["duration_anomaly"] = (X["koi_duration"] / 24.0) / expected

    use_cols = [c for c in BASE_FEATURES if c in X.columns] + [
        "ratio_duration_period","log_period","log_duration","log_depth","log_prad",
        "rp_over_rstar","stellar_density_proxy","duration_expected_proxy","duration_anomaly",
    ]
    return X[use_cols]

# ================================
# 3A) Satır bazlı stratified split
# ================================
def stratified_tvt_split(df, test_size=0.2, val_size=0.1, save_dir=None):
    y_all = df["koi_disposition"].map(LABEL_MAP).values

    # test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    idx_trainval, idx_test = next(sss1.split(df, y_all))
    df_trainval, df_test = df.iloc[idx_trainval], df.iloc[idx_test]

    # valid
    y_trainval = df_trainval["koi_disposition"].map(LABEL_MAP).values
    val_ratio_within_trainval = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_within_trainval, random_state=RANDOM_STATE)
    idx_train, idx_val = next(sss2.split(df_trainval, y_trainval))
    df_train, df_val = df_trainval.iloc[idx_train], df_trainval.iloc[idx_val]

    if save_dir:
        p = Path(save_dir); p.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(p / "koi_train.csv", index=False)
        df_val.to_csv(p / "koi_valid.csv", index=False)
        df_test.to_csv(p / "koi_test.csv", index=False)
        print(f"[SAVED] Train/valid/test CSVs have been saved under '{p}/'.")

    Xtr, ytr = build_features(df_train), df_train["koi_disposition"].map(LABEL_MAP).values
    Xva, yva = build_features(df_val), df_val["koi_disposition"].map(LABEL_MAP).values
    Xte, yte = build_features(df_test), df_test["koi_disposition"].map(LABEL_MAP).values
    return (Xtr, ytr), (Xva, yva), (Xte, yte), (df_train, df_val, df_test)

# ================================
# 3B) Yıldız (kepid) bazlı group split — leakage yok
# ================================
def _mode_label(series: pd.Series) -> int:
    m = series.mode()
    label = m.iloc[0] if not m.empty else series.iloc[0]
    return LABEL_MAP[label]

def split_by_star_groups(df: pd.DataFrame, test_size=0.20, val_size=0.10, save_dir: str | None = None):
    if "kepid" not in df.columns:
        raise ValueError("Grup bazlı split için 'kepid' sütunu gereklidir.")

    # FutureWarning'siz: groupby.apply yerine agg
    mode_per_star = df.groupby("kepid")["koi_disposition"].agg(_mode_label)
    stars = mode_per_star.index.to_numpy()
    y_star = mode_per_star.values.astype(int)

    # test ayır
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    trv_idx, te_idx = next(gss1.split(stars, y_star, groups=stars))
    trv_stars, te_stars = stars[trv_idx], stars[te_idx]

    trv_df = df[df["kepid"].isin(trv_stars)].reset_index(drop=True)
    te_df  = df[df["kepid"].isin(te_stars)].reset_index(drop=True)

    # valid ayır
    val_ratio_within_trainval = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
    trv_mode = mode_per_star.loc[trv_stars].values.astype(int)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_within_trainval, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(gss2.split(trv_stars, trv_mode, groups=trv_stars))
    tr_stars, va_stars = trv_stars[tr_idx], trv_stars[va_idx]

    tr_df = df[df["kepid"].isin(tr_stars)].reset_index(drop=True)
    va_df = df[df["kepid"].isin(va_stars)].reset_index(drop=True)

    # güvenlik: overlap 0 mı?
    print(f"[CHECK] Overlap(train,test)={len(set(tr_df['kepid']) & set(te_df['kepid']))}, "
          f"Overlap(val,test)={len(set(va_df['kepid']) & set(te_df['kepid']))}, "
          f"Overlap(train,val)={len(set(tr_df['kepid']) & set(va_df['kepid']))}")

    if save_dir:
        p = Path(save_dir); p.mkdir(parents=True, exist_ok=True)
        tr_df.to_csv(p / "koi_train.csv", index=False)
        va_df.to_csv(p / "koi_valid.csv", index=False)
        te_df.to_csv(p / "koi_test.csv",  index=False)
        print(f"[SAVED] Star-group splits → {p}/koi_train.csv, koi_valid.csv, koi_test.csv")

    Xtr, ytr = build_features(tr_df), tr_df["koi_disposition"].map(LABEL_MAP).values
    Xva, yva = build_features(va_df), va_df["koi_disposition"].map(LABEL_MAP).values
    Xte, yte = build_features(te_df), te_df["koi_disposition"].map(LABEL_MAP).values
    return (Xtr, ytr), (Xva, yva), (Xte, yte), (tr_df, va_df, te_df)

# ================================
# 4) Class weights → sample_weight
# ================================
def make_sample_weight(y: np.ndarray) -> np.ndarray:
    cnt = Counter(y)
    total = len(y)
    ncls = len(cnt)
    weights = {c: total / (ncls * cnt[c]) for c in cnt}  # N/(k*nc)
    return np.array([weights[int(c)] for c in y], dtype=float)

# ================================
# 5) Ana akış
# ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="KOI cumulative CSV yolu")
    ap.add_argument("--splits_dir", default="data/splits", help="Split dosyalarını kaydetmek için klasör")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--model_out", default="models/xgb_koi.joblib")
    ap.add_argument("--group_split_by_kepid", action="store_true",
                    help="Train/Val/Test'i kepid gruplarına göre ayır (data leakage engellenir).")
    ap.add_argument("--base_model", type=str, default=None,
                    help="Warm-start: mevcut XGBoost modelinden devam et.")
    ap.add_argument("--extra_estimators", type=int, default=0,
                    help="Baz modelin üzerine eklenecek ağaç sayısı (yalnızca warm-start).")
    ap.add_argument("--learning_rate", type=float, default=0.05,
                    help="Yeni ağaçların katkı gücü (scratch ve warm-start)")
    ap.add_argument("--no_split", action="store_true",
                    help="Veri setini bölme; tamamını train olarak kullan.")
    args = ap.parse_args()

    print("[INFO] Loading data...")
    df_all = load_koi_df(args.train_csv)
    print(f"[INFO] Total {len(df_all)} rows read.")
    print(df_all["koi_disposition"].value_counts())

    # ===== Split kararı =====
    if args.no_split:
        print("[INFO] no_split active: all data will be used as train; val/test will be empty.")
        df_train = df_all.copy()
        df_val   = df_all.iloc[0:0].copy()
        df_test  = df_all.iloc[0:0].copy()

        Xtr = build_features(df_train)
        # Aynı kolon yapısıyla boş DF oluştur
        Xva = Xtr.iloc[0:0].copy()
        Xte = Xtr.iloc[0:0].copy()

        ytr = df_train["koi_disposition"].map(LABEL_MAP).values
        yva = np.array([])
        yte = np.array([])
    else:
        if args.group_split_by_kepid:
            print("[INFO] Star-group (kepid) split is being applied...")
            (Xtr, ytr), (Xva, yva), (Xte, yte), (df_train, df_val, df_test) = split_by_star_groups(
                df_all, test_size=args.test_size, val_size=args.val_size, save_dir=args.splits_dir
            )
        else:
            print("[INFO] Row-based stratified split is being applied (leakage risk possible)...")
            (Xtr, ytr), (Xva, yva), (Xte, yte), (df_train, df_val, df_test) = stratified_tvt_split(
                df_all, test_size=args.test_size, val_size=args.val_size, save_dir=args.splits_dir
            )
            # Ek bilgi: kepid overlap kontrolü
            try:
                if "kepid" in df_train.columns and "kepid" in df_test.columns:
                    overlap = set(df_train["kepid"]).intersection(df_test["kepid"])
                    print(f"[DEBUG] Number of common kepid in Train/Test: {len(overlap)}")
                    if len(overlap) > 0:
                        print("⚠️  WARNING: The same star (kepid) exists in both train and test! "
                              "For real generalization, use --group_split_by_kepid.")
            except Exception as e:
                print(f"[WARN] Error during overlap check: {e}")

    # ===== Imputer =====
    print("[INFO] Filling missing values...")
    imp = SimpleImputer(strategy="median")
    Xtr_i = imp.fit_transform(Xtr)
    Xva_i = imp.transform(Xva) if isinstance(Xva, pd.DataFrame) and len(Xva) > 0 else None
    Xte_i = imp.transform(Xte) if isinstance(Xte, pd.DataFrame) and len(Xte) > 0 else None

    # ===== Weights =====
    sw_tr = make_sample_weight(ytr)

    # ===== Model =====
    print("[INFO] Training XGBoost model...")
    eval_set = []
    if Xva_i is not None and len(yva) > 0:
        eval_set.append((Xva_i, yva))

    if args.base_model:
        print(f"[INFO] Warm-start: loading base model → {args.base_model}")
        base_model = load_base_model(args.base_model)
        base_params = base_model.get_params(deep=True)

        base_n = int(base_params.get("n_estimators", 600))
        extra  = max(0, int(args.extra_estimators))
        total_n = base_n + extra
        print(f"[INFO] Base n_estimators={base_n}, extra={extra} ⇒ total={total_n}")

        model = XGBClassifier(
            objective="multi:softprob", num_class=3,
            n_estimators=total_n,
            learning_rate=args.learning_rate,
            max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.5, reg_alpha=0.5,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
            tree_method="hist", n_jobs=-1,
        )
        fit_kwargs = {"verbose": False}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set

        model.fit(
            Xtr_i, ytr,
            sample_weight=sw_tr,
            xgb_model=base_model.get_booster(),  # booster’dan devam
            **fit_kwargs
        )
    else:
        model = XGBClassifier(
            objective="multi:softprob", num_class=3,
            n_estimators=600,
            learning_rate=args.learning_rate,
            max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.5, reg_alpha=0.5,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
            tree_method="hist", n_jobs=-1,
        )
        fit_kwargs = {"verbose": False}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
        model.fit(Xtr_i, ytr, sample_weight=sw_tr, **fit_kwargs)

    # ===== VALIDATION =====
    print("\n=== VALIDATION ===")
    if Xva_i is not None and len(yva) > 0:
        yva_pred = model.predict(Xva_i)
        print(classification_report(yva, yva_pred, target_names=[IDX2LBL[i] for i in range(3)]))
        print("Confusion (val):\n", confusion_matrix(yva, yva_pred))
    else:
        print("(empty)")

    # ===== TEST =====
    print("\n=== TEST ===")
    if Xte_i is not None and len(yte) > 0:
        yte_pred = model.predict(Xte_i)
        print(classification_report(yte, yte_pred, target_names=[IDX2LBL[i] for i in range(3)]))
        print("Confusion (test):\n", confusion_matrix(yte, yte_pred))
        print(f"Macro F1 (test): {f1_score(yte, yte_pred, average='macro'):.3f}")
    else:
        print("(empty)")

    # ===== Kaydet =====
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    # Not: feature isimlerini saklayalım
    feature_names = list(pd.DataFrame(Xtr).columns)
    joblib.dump({"model": model, "imputer": imp, "features": feature_names}, args.model_out)
    val_report = None
    test_report = None
    val_conf = None
    test_conf = None

    if Xva_i is not None and len(yva) > 0:
        yva_pred = model.predict(Xva_i)
        val_report = classification_report(yva, yva_pred,
                                        target_names=[IDX2LBL[i] for i in range(3)],
                                        output_dict=True)
        val_conf = confusion_matrix(yva, yva_pred).tolist()

    if Xte_i is not None and len(yte) > 0:
        yte_pred = model.predict(Xte_i)
        test_report = classification_report(yte, yte_pred,
                                            target_names=[IDX2LBL[i] for i in range(3)],
                                            output_dict=True)
        test_conf = confusion_matrix(yte, yte_pred).tolist()

    metrics = {
        "val_report": val_report,
        "val_confusion": val_conf,
        "test_report": test_report,
        "test_confusion": test_conf,
    }

    # ---- (2) MANIFEST OLUŞTUR ----
    model_path = Path(args.model_out)
    manifest_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
    metrics_path  = model_path.with_suffix(model_path.suffix + ".metrics.json")

    # model dosyasının hash’i (depolama bütünlüğü için faydalı)
    sha256 = hashlib.sha256(Path(args.model_out).read_bytes()).hexdigest()

    manifest = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "artifact_path": str(model_path),
        "artifact_sha256": sha256,

        # Eğitimde kullanılan argümanlar / ayarlar
        "random_state": RANDOM_STATE,
        "imputer": {"strategy": "median"},
        "base_model": args.base_model,
        "extra_estimators": int(args.extra_estimators),
        "learning_rate": float(args.learning_rate),
        "no_split": bool(args.no_split),
        "group_split_by_kepid": bool(args.group_split_by_kepid),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),

        # Veri şeması / özellikler
        "label_map": LABEL_MAP,               # {"FALSE POSITIVE":0, ...}
        "classes": [IDX2LBL[i] for i in range(3)],
        "feature_names": feature_names,       # inference tarafı bunları bekler
        "base_features": [c for c in BASE_FEATURES if c in df_all.columns],
        "engineered_features": [
            "ratio_duration_period","log_period","log_duration","log_depth","log_prad",
            "rp_over_rstar","stellar_density_proxy","duration_expected_proxy","duration_anomaly",
        ],

        # Split özeti (satır sayıları)
        "split_summary": {
            "train_rows": int(len(Xtr)),
            "val_rows":   int(len(Xva)) if isinstance(Xva, pd.DataFrame) else 0,
            "test_rows":  int(len(Xte)) if isinstance(Xte, pd.DataFrame) else 0,
        },
    }

    # ---- (3) DİSK’E YAZ ----
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] Model saved → {args.model_out}")
    print(f"[SAVED] Manifest → {manifest_path}")
    print(f"[SAVED] Metrics  → {metrics_path}")

    print(f"[SAVED] Model saved → {args.model_out}")

if __name__ == "__main__":
    main()
