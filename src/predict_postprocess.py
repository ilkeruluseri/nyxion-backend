# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

LABELS = ["FALSE POSITIVE","CANDIDATE","CONFIRMED"]  # 0,1,2
LBL2IDX = {n:i for i,n in enumerate(LABELS)}
IDX2LBL = {i:n for n,i in LBL2IDX.items()}
LABEL_CANDIDATES = {"disposition","koi_disposition","label","target","y","class","CLASS","Disposition"}

def load_model(path):
    return joblib.load(path)

def infer_expected_features(pipe):
    if hasattr(pipe, "feature_names_in_"):
        return list(pipe.feature_names_in_)
    if hasattr(pipe, "named_steps"):
        for _, step in pipe.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def read_align_df(csv_path, expected_cols=None):
    df = pd.read_csv(csv_path)
    # label sütunu varsa ayır
    label_col = None
    for c in df.columns:
        if c in LABEL_CANDIDATES:
            label_col = c
            break
    y_true = None
    if label_col is not None:
        y_true = df[label_col].astype(str).str.upper().map({
            "FALSE POSITIVE":"FALSE POSITIVE", "FALSE_POSITIVE":"FALSE POSITIVE", "FALSE-POSITIVE":"FALSE POSITIVE", "FP":"FALSE POSITIVE",
            "CANDIDATE":"CANDIDATE", "CAND":"CANDIDATE",
            "CONFIRMED":"CONFIRMED", "CONF":"CONFIRMED",
        })
        df = df.drop(columns=[label_col])

    # object -> numeric
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf,-np.inf], np.nan)

    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in df.columns]
        for m in missing:
            df[m] = np.nan
        df = df[expected_cols]  # reorder
    return df, y_true

# ------- CANDIDATE-focused post-processing rules -------
def postprocess_candidate(proba, thr_cand=0.40, margin=0.06, ratio_k=1.10):
    """
    proba: [N,3] -> columns = [FP, CANDIDATE, CONFIRMED]
    Kurallar:
      A) Eğer p_cand >= thr_cand ve p_cand / max(p_fp,p_conf) >= ratio_k => CANDIDATE
      B) Margin küçükse (top1 - top2 < margin) ve p_cand >= max(0.25, thr_cand-0.05) => CANDIDATE
      (Aksi halde argmax)
    """
    p = proba
    argmax = p.argmax(axis=1)
    y = argmax.copy()

    best_other = np.maximum(p[:,0], p[:,2])
    ruleA = (p[:,1] >= thr_cand) & (p[:,1] / (best_other + 1e-12) >= ratio_k)

    # top1-top2 margin
    # (en büyük iki olasılığı bul)
    top2 = np.partition(-p, 1, axis=1)   # negatif, o yüzden - ile çalışıyoruz
    top1v = -top2[:,0]; top2v = -top2[:,1]
    ruleB = (top1v - top2v < margin) & (p[:,1] >= max(0.25, thr_cand - 0.05))

    y[ruleA | ruleB] = 1  # 1=CANDIDATE
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pkl")
    ap.add_argument("--input", required=True, help="CSV path (test/inference)")
    ap.add_argument("--output", default="predictions_pp.csv")
    ap.add_argument("--thr-cand", type=float, default=0.40)
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--ratio-k", type=float, default=1.10)
    ap.add_argument("--eval", action="store_true", help="If ground-truth exists in CSV, print metrics")
    args = ap.parse_args()

    pipe = load_model(args.model)
    expected = infer_expected_features(pipe)

    X, y_true = read_align_df(args.input, expected_cols=expected)

    # proba + raw argmax
    proba = pipe.predict_proba(X)
    y_raw = proba.argmax(axis=1)

    # post-process
    y_pp = postprocess_candidate(
        proba,
        thr_cand=args.thr_cand,
        margin=args.margin,
        ratio_k=args.ratio_k
    )

    # build output
    out = X.copy()
    out["pred_raw_idx"] = y_raw
    out["pred_pp_idx"]  = y_pp
    out["pred_raw"] = [IDX2LBL[i] for i in y_raw]
    out["pred_pp"]  = [IDX2LBL[i] for i in y_pp]
    out["proba_fp"]   = proba[:,0]
    out["proba_cand"] = proba[:,1]
    out["proba_conf"] = proba[:,2]

    # if ground truth available, evaluate both
    if args.eval and y_true is not None:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
        y_true_idx = [LBL2IDX.get(s, -1) for s in y_true]
        mask = np.array([i in (0,1,2) for i in y_true_idx])
        y_true_idx = np.array(y_true_idx)[mask]
        y_raw_m = y_raw[mask]
        y_pp_m  = y_pp[mask]

        def prn(title, y_pred):
            print(f"\n== {title} ==")
            print("Accuracy:", accuracy_score(y_true_idx, y_pred).round(4))
            print("Macro-F1:", f1_score(y_true_idx, y_pred, average="macro").round(4))
            print("\nClassification Report:")
            print(classification_report(y_true_idx, y_pred, target_names=LABELS))
            print("\nConfusion Matrix (rows=true, cols=pred):")
            print(pd.DataFrame(confusion_matrix(y_true_idx, y_pred, labels=[0,1,2]),
                               index=[f"T_{l}" for l in LABELS],
                               columns=[f"P_{l}" for l in LABELS]))

        prn("RAW (argmax)", y_raw_m)
        prn("POST-PROCESSED (CANDIDATE-biased)", y_pp_m)

    # save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[ok] saved: {args.output}")
    print(f"Params -> thr_cand={args.thr_cand}, margin={args.margin}, ratio_k={args.ratio_k}")

if __name__ == "__main__":
    main()
