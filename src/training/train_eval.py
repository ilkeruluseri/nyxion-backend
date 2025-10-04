#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

try:
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

LABEL_MAP = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
IDX2LBL   = {v: k for k, v in LABEL_MAP.items()}
RANDOM_STATE = 42

THR1_GRID = [0.20,0.25,0.30,0.35,0.40,0.45]   # Aşama-1 (CAND eşiği)
def summarize_metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "cand_f1": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "cand_prec": float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "cand_rec": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
    }

def build_lgbm_bin(class_weight_pos=1.0, **ovr):
    params = dict(
        objective="binary", boosting_type="gbdt",
        learning_rate=0.05, n_estimators=1200, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8,
        class_weight={0:1.0, 1:float(class_weight_pos)},
        random_state=RANDOM_STATE, n_jobs=-1, early_stopping_rounds=100,
    )
    params.update(ovr or {})
    return LGBMClassifier(**params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="models_cascade")
    ap.add_argument("--class-weight-cand1", type=float, default=4.0,
                    help="Aşama-1: CAND (pozitif) sınıf ağırlığı")
    ap.add_argument("--target-recall-cand1", type=float, default=0.65,
                    help="Val üzerinde CAND recall hedefi (Aşama-1 eşiği için)")
    ap.add_argument("--use-smotetomek", action="store_true",
                    help="Sadece TRAIN'da SMOTETomek uygula (Aşama-1)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # --- Veriyi oku
    df = pd.read_csv(args.csv)
    assert "disposition" in df.columns
    df = df.dropna(subset=["disposition"]).reset_index(drop=True)
    df["y"] = df["disposition"].map(LABEL_MAP).astype(int)
    features = [c for c in df.columns if c not in ("disposition","y")]
    X = df[features]; y = df["y"]

    # --- %15 test ayır
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    trval_idx, te_idx = next(sss1.split(X, y))
    X_trval, X_te = X.iloc[trval_idx], X.iloc[te_idx]
    y_trval, y_te = y.iloc[trval_idx], y.iloc[te_idx]

    # --- %15 validation (0.15/0.85)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1764706, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(sss2.split(X_trval, y_trval))
    X_tr, X_va = X_trval.iloc[tr_idx], X_trval.iloc[va_idx]
    y_tr, y_va = y_trval.iloc[tr_idx], y_trval.iloc[va_idx]

    # ===== AŞAMA-1: CAND vs OTHERS =====
    y_tr_bin = (y_tr == 1).astype(int)
    y_va_bin = (y_va == 1).astype(int)
    imp1 = SimpleImputer(strategy="median")
    Xtr1 = imp1.fit_transform(X_tr); Xva1 = imp1.transform(X_va)

    # Opsiyonel SMOTETomek (sadece Aşama-1 train)
    if args.use_smotetomek:
        if HAS_IMBLEARN:
            sm = SMOTETomek(random_state=RANDOM_STATE)
            Xtr1, y_tr_bin = sm.fit_resample(Xtr1, y_tr_bin)
            print(f"[SMOTETomek A1] -> {len(y_tr_bin)} örnek")
        else:
            print("⚠️ imblearn yok; SMOTETomek atlandı.")

    clf1 = build_lgbm_bin(class_weight_pos=args.class_weight_cand1)
    clf1.fit(Xtr1, y_tr_bin, eval_set=[(Xva1, y_va_bin)], eval_metric="logloss")

    # Kalibrasyon (isotonic, prefit)
    cal1 = CalibratedClassifierCV(clf1, method="isotonic", cv="prefit")
    cal1.fit(Xva1, y_va_bin)

    # Val üstünde thr1 seçimi: CAND recall >= hedef
    P1_va = cal1.predict_proba(Xva1)[:,1]
    best = {"thr1": None, "cand_rec": -1, "cand_f1": -1, "cand_prec": -1, "macro_f1": -1}
    for thr1 in THR1_GRID:
        y_hat1 = (P1_va >= thr1).astype(int)  # 1=CAND, 0=OTHERS
        # nihai 3 sınıfa çevirmek için Aşama-2 yokken kabaca ölçüm: CAND doğru/yanlış sinyali
        rec = recall_score(y_va_bin, y_hat1, zero_division=0)
        prec= precision_score(y_va_bin, y_hat1, zero_division=0)
        f1  = f1_score(y_va_bin, y_hat1, zero_division=0)
        # hedef recall koşulu
        if rec >= args.target_recall_cand1:
            # macro_f1 için ikili yerine üçlü sonuca geçmek daha anlamlı ama hızlı seçim için f1 cand kullanıyoruz
            better = (f1 > best["cand_f1"]) or (np.isclose(f1, best["cand_f1"]) and rec > best["cand_rec"])
            if better:
                best.update({"thr1": float(thr1), "cand_rec": float(rec), "cand_prec": float(prec), "cand_f1": float(f1)})
    # hiçbiri hedefi tutturmazsa en iyi f1'i seç
    if best["thr1"] is None:
        for thr1 in THR1_GRID:
            y_hat1 = (P1_va >= thr1).astype(int)
            rec = recall_score(y_va_bin, y_hat1, zero_division=0)
            prec= precision_score(y_va_bin, y_hat1, zero_division=0)
            f1  = f1_score(y_va_bin, y_hat1, zero_division=0)
            if f1 > best["cand_f1"]:
                best.update({"thr1": float(thr1), "cand_rec": float(rec), "cand_prec": float(prec), "cand_f1": float(f1)})

    thr1 = best["thr1"] if best["thr1"] is not None else 0.35

    # ===== AŞAMA-2: CONF vs FP (CAND olmayanlarda) =====
    # Train+Val (85%) ile final eğitimi yapacağız; ama A2 veri hazırlığı için trval üzerinde mantığı kuruyoruz
    # Final eğitime geçmeden önce: TRAIN+VAL birleştirme ve yeniden fit (A1 + A2)

    # ---- Final: TRAIN+VAL = 85%
    imp_final = SimpleImputer(strategy="median")
    Xtrval = imp_final.fit_transform(X_trval)
    # A1 final (cv kalibrasyon)
    y_trval_bin = (y_trval == 1).astype(int)
    clf1_final = build_lgbm_bin(class_weight_pos=args.class_weight_cand1)
    clf1_final.set_params(early_stopping_rounds=None)
    clf1_final.fit(Xtrval, y_trval_bin)
    cal1_final = CalibratedClassifierCV(clf1_final, method="isotonic", cv=5)
    cal1_final.fit(Xtrval, y_trval_bin)

    # A2 final verisi: CAND olmayanlar
    P1_trval = cal1_final.predict_proba(Xtrval)[:,1]
    non_cand_mask = (P1_trval < thr1)
    Xtrval_a2 = Xtrval[non_cand_mask]
    ytrval_a2 = y_trval.to_numpy()[non_cand_mask]
    y_a2 = (ytrval_a2 == 2).astype(int)  # 1=CONF, 0=FP

    clf2_final = build_lgbm_bin(class_weight_pos=1.0)
    clf2_final.set_params(early_stopping_rounds=None)
    clf2_final.fit(Xtrval_a2, y_a2)
    cal2_final = CalibratedClassifierCV(clf2_final, method="isotonic", cv=5)
    cal2_final.fit(Xtrval_a2, y_a2)

    # --- Test: tamamen görülmemiş
    Xte_ = imp_final.transform(X_te)
    P1_te = cal1_final.predict_proba(Xte_)[:,1]
    is_cand = (P1_te >= thr1)

    y_hat = np.full(len(X_te), -1, dtype=int)
    y_hat[is_cand] = 1  # CAND

    # A2 sadece CAND olmayanlarda
    idx_rest = np.where(~is_cand)[0]
    if len(idx_rest) > 0:
        P2_te = cal2_final.predict_proba(Xte_[idx_rest])[:,1]  # p(CONF)
        y_hat[idx_rest] = np.where(P2_te >= 0.5, 2, 0)

    # metrikler
    test_metrics = summarize_metrics(y_te.to_numpy(), y_hat)
    cls_rep = classification_report(y_te, y_hat, target_names=LABEL_MAP.keys(), zero_division=0)
    cm = confusion_matrix(y_te, y_hat).tolist()

    # Kaydet
    pkg = {
        "imputer": imp_final,
        "stage1": cal1_final, "thr1": float(thr1),
        "stage2": cal2_final,
        "features": features,
        "label_map": LABEL_MAP, "idx2lbl": IDX2LBL,
        "split_info": {"train":0.70, "val":0.15, "test":0.15},
        "class_weight_cand1": float(args.class_weight_cand1),
        "target_recall_cand1": float(args.target_recall_cand1),
        "used_smotetomek": bool(args.use_smotetomek and HAS_IMBLEARN),
    }
    out_model = outdir / "model_cascade.ptk"
    joblib.dump(pkg, out_model)

    test_out = X_te.copy()
    test_out["true_label"] = y_te.to_numpy()
    test_out["pred_label"] = y_hat
    test_out["pred_disposition"] = [IDX2LBL[int(i)] for i in y_hat]
    test_out["p1_cand"] = P1_te
    test_csv = outdir / f"test_predictions_{ts}.csv"
    test_out.to_csv(test_csv, index=False)

    metrics = {
        "timestamp": ts, "source_csv": str(Path(args.csv).resolve()),
        "thr1": float(thr1), "val_stage1_choice": best,
        "test_metrics": test_metrics, "confusion_matrix": cm,
        "classification_report": cls_rep,
        "sizes": {"train_plus_val": int(len(X_trval)), "test": int(len(X_te)), "total": int(len(df))},
    }
    metrics_path = outdir / f"test_metrics_{ts}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Cascade model saved -> {out_model}")
    print(f"✅ Test predictions   -> {test_csv}")
    print(f"✅ Test metrics       -> {metrics_path}")
    print("\n=== Test Summary (Cascade) ===")
    print(cls_rep)

if __name__ == "__main__":
    main()


