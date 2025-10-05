#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

LABEL_MAP = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
IDX2LBL = {v: k for k, v in LABEL_MAP.items()}
RANDOM_STATE = 42

def load_koi_df(csv_path: str, dedup=True) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#")

    if "koi_disposition" not in df.columns:
        raise ValueError("Column 'koi_disposition' not found in CSV!")

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

    for c in BASE_FEATURES:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X["ratio_duration_period"] = (X["koi_duration"] / 24.0) / X["koi_period"]
    X["log_period"]   = np.log1p(X["koi_period"])
    X["log_duration"] = np.log1p(X["koi_duration"])
    X["log_depth"]    = np.log1p(X["koi_depth"])
    X["log_prad"]     = np.log1p(X["koi_prad"])

    frac_depth = pd.to_numeric(X.get("koi_depth", np.nan), errors="coerce") / 1e6
    X["rp_over_rstar"] = np.sqrt(np.clip(frac_depth, a_min=0, a_max=None))

    if "koi_smass" in X.columns and "koi_srad" in X.columns:
        X["stellar_density_proxy"] = X["koi_smass"] / (X["koi_srad"] ** 3)
    else:
        X["stellar_density_proxy"] = np.nan

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

def stratified_tvt_split(df, test_size=0.2, val_size=0.1, save_dir=None):
    y_all = df["koi_disposition"].map(LABEL_MAP).values

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    idx_trainval, idx_test = next(sss1.split(df, y_all))
    df_trainval, df_test = df.iloc[idx_trainval], df.iloc[idx_test]

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
        print(f"✓ Splits saved to: {p}/")

    Xtr, ytr = build_features(df_train), df_train["koi_disposition"].map(LABEL_MAP).values
    Xva, yva = build_features(df_val), df_val["koi_disposition"].map(LABEL_MAP).values
    Xte, yte = build_features(df_test), df_test["koi_disposition"].map(LABEL_MAP).values
    return (Xtr, ytr), (Xva, yva), (Xte, yte), (df_train, df_val, df_test)

def _star_mode_label(group_df: pd.DataFrame) -> int:
    mode_disp = group_df["koi_disposition"].mode().iloc[0]
    return LABEL_MAP[mode_disp]

def split_by_star_groups(df: pd.DataFrame, test_size=0.20, val_size=0.10, save_dir: str | None = None):
    if "kepid" not in df.columns:
        raise ValueError("Column 'kepid' required for group-based split.")

    mode_per_star = df.groupby("kepid").apply(_star_mode_label)
    stars = mode_per_star.index.to_numpy()
    y_star = mode_per_star.values.astype(int)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    trv_idx, te_idx = next(gss1.split(stars, y_star, groups=stars))
    trv_stars, te_stars = stars[trv_idx], stars[te_idx]

    trv_df = df[df["kepid"].isin(trv_stars)].reset_index(drop=True)
    te_df  = df[df["kepid"].isin(te_stars)].reset_index(drop=True)

    val_ratio_within_trainval = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
    trv_mode = mode_per_star.loc[trv_stars].values.astype(int)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_within_trainval, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(gss2.split(trv_stars, trv_mode, groups=trv_stars))
    tr_stars, va_stars = trv_stars[tr_idx], trv_stars[va_idx]

    tr_df = df[df["kepid"].isin(tr_stars)].reset_index(drop=True)
    va_df = df[df["kepid"].isin(va_stars)].reset_index(drop=True)

    overlap_train_test = len(set(tr_df['kepid']) & set(te_df['kepid']))
    overlap_val_test = len(set(va_df['kepid']) & set(te_df['kepid']))
    overlap_train_val = len(set(tr_df['kepid']) & set(va_df['kepid']))
    
    print(f"✓ Group overlap check: train↔test={overlap_train_test}, val↔test={overlap_val_test}, train↔val={overlap_train_val}")

    if save_dir:
        p = Path(save_dir); p.mkdir(parents=True, exist_ok=True)
        tr_df.to_csv(p / "koi_train.csv", index=False)
        va_df.to_csv(p / "koi_valid.csv", index=False)
        te_df.to_csv(p / "koi_test.csv",  index=False)
        print(f"✓ Star-grouped splits saved to: {p}/")

    Xtr, ytr = build_features(tr_df), tr_df["koi_disposition"].map(LABEL_MAP).values
    Xva, yva = build_features(va_df), va_df["koi_disposition"].map(LABEL_MAP).values
    Xte, yte = build_features(te_df), te_df["koi_disposition"].map(LABEL_MAP).values
    return (Xtr, ytr), (Xva, yva), (Xte, yte), (tr_df, va_df, te_df)

def make_sample_weight(y: np.ndarray) -> np.ndarray:
    cnt = Counter(y)
    total = len(y)
    ncls = len(cnt)
    weights = {c: total / (ncls * cnt[c]) for c in cnt}
    return np.array([weights[int(c)] for c in y], dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Path to KOI cumulative CSV")
    ap.add_argument("--splits_dir", default="data/splits", help="Directory to save train/val/test splits")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--model_out", default="models/xgb_koi.joblib")
    ap.add_argument("--group_split_by_kepid", action="store_true",
                    help="Split by kepid groups to prevent data leakage")
    args = ap.parse_args()

    print("=" * 80)
    print("KOI EXOPLANET CLASSIFICATION TRAINING")
    print("=" * 80)
    
    print(f"\n→ Loading dataset: {args.train_csv}")
    df_all = load_koi_df(args.train_csv)
    print(f"✓ Loaded {len(df_all)} rows")
    print("\nClass distribution:")
    print(df_all["koi_disposition"].value_counts().to_string())

    if args.group_split_by_kepid:
        print(f"\n→ Applying star-grouped split (kepid-based, leakage-free)")
        (Xtr, ytr), (Xva, yva), (Xte, yte), (df_train, df_val, df_test) = split_by_star_groups(
            df_all, test_size=args.test_size, val_size=args.val_size, save_dir=args.splits_dir
        )
    else:
        print(f"\n→ Applying stratified row-based split (potential leakage risk)")
        (Xtr, ytr), (Xva, yva), (Xte, yte), (df_train, df_val, df_test) = stratified_tvt_split(
            df_all, test_size=args.test_size, val_size=args.val_size, save_dir=args.splits_dir
        )
        try:
            if "kepid" in df_train.columns and "kepid" in df_test.columns:
                overlap = set(df_train["kepid"]).intersection(df_test["kepid"])
                if len(overlap) > 0:
                    print(f"⚠ WARNING: {len(overlap)} kepid(s) appear in both train and test!")
                    print("  Consider using --group_split_by_kepid for true generalization.")
        except Exception as e:
            print(f"⚠ Overlap check failed: {e}")

    print(f"\n→ Imputing missing values (median strategy)")
    imp = SimpleImputer(strategy="median")
    Xtr_i, Xva_i, Xte_i = imp.fit_transform(Xtr), imp.transform(Xva), imp.transform(Xte)

    sw_tr = make_sample_weight(ytr)

    print(f"\n→ Training XGBoost model")
    print(f"  • n_estimators: 600")
    print(f"  • learning_rate: 0.05")
    print(f"  • max_depth: 6")
    print(f"  • subsample: 0.9")
    print(f"  • colsample_bytree: 0.9")
    print(f"  • reg_lambda: 1.5, reg_alpha: 0.5")
    
    model = XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.5, reg_alpha=0.5,
        random_state=RANDOM_STATE, eval_metric="mlogloss",
        tree_method="hist", n_jobs=-1,
    )

    model.fit(Xtr_i, ytr, sample_weight=sw_tr, eval_set=[(Xva_i, yva)], verbose=False)
    print("✓ Training complete")

    print("\n" + "=" * 80)
    print("VALIDATION SET PERFORMANCE")
    print("=" * 80)
    yva_pred = model.predict(Xva_i)
    print(classification_report(yva, yva_pred, target_names=[IDX2LBL[i] for i in range(3)]))
    print("\nConfusion Matrix (Validation):")
    print(confusion_matrix(yva, yva_pred))

    print("\n" + "=" * 80)
    print("TEST SET PERFORMANCE")
    print("=" * 80)
    yte_pred = model.predict(Xte_i)
    print(classification_report(yte, yte_pred, target_names=[IDX2LBL[i] for i in range(3)]))
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(yte, yte_pred))
    macro_f1 = f1_score(yte, yte_pred, average='macro')
    print(f"\n→ Macro F1 Score (Test): {macro_f1:.4f}")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "imputer": imp, "features": list(pd.DataFrame(Xtr).columns)}, args.model_out)
    print(f"\n✓ Model saved to: {args.model_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()