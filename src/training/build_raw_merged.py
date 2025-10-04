# src/training/build_raw_merged.py
import pandas as pd, numpy as np
from pathlib import Path
import json, sys, re

RAW = Path("data/raw")
OUT = Path("data/interim"); OUT.mkdir(parents=True, exist_ok=True)

def read_csv(p: Path) -> pd.DataFrame:
    # C engine yeterli; '#' yorum satırlarını at
    return pd.read_csv(p, comment="#")

def first_col(df, names):
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series(np.nan, index=df.index)

def unify_kepler(df):
    o = pd.DataFrame(index=df.index)
    o["mission"]       = "Kepler"
    o["object_name"]   = first_col(df, ["kepoi_name","kepler_name","kepid"])
    o["period_days"]   = first_col(df, ["koi_period"])
    o["duration_days"] = first_col(df, ["koi_duration"])
    o["depth"]         = first_col(df, ["koi_depth"])
    o["prad_re"]       = first_col(df, ["koi_prad"])
    o["steff_K"]       = first_col(df, ["koi_steff"])
    o["srad_Rsun"]     = first_col(df, ["koi_srad"])
    o["smass_MSun"]    = first_col(df, ["koi_smass"])
    o["disposition"]   = first_col(df, ["koi_disposition"])
    return o

import pandas as pd
import numpy as np

def unify_k2(df: pd.DataFrame) -> pd.DataFrame:
    """
    K2 dataset'ini modelin beklediği ortak feature isimlerine dönüştürür.
    Beklenen giriş kolonları (örnek K2 yapısı):
      - pl_orbper      : orbital period [days]
      - pl_trandur     : transit duration [hours]
      - pl_trandep     : transit depth [ppm or fraction]
      - pl_rade        : planet radius [Earth radii]
      - st_teff        : stellar effective temperature [Kelvin]
      - st_rad         : stellar radius [Solar radii]
      - st_mass        : stellar mass [Solar masses]
      - disposition    : label (CONFIRMED / CANDIDATE / FALSE POSITIVE)
    """

    o = pd.DataFrame(index=df.index)

    # --- Mission bilgisi sabit ---
    o["mission"] = "K2"

    # --- Ana fiziksel kolonlar ---
    o["period_days"]   = df.get("pl_orbper", np.nan)
    o["duration_days"] = df.get("pl_trandur", np.nan) / 24.0   # saat → gün
    o["depth"]         = df.get("pl_trandep", np.nan)
    o["prad_re"]       = df.get("pl_rade", np.nan)
    o["steff_K"]       = df.get("st_teff", np.nan)
    o["srad_Rsun"]     = df.get("st_rad", np.nan)
    o["smass_MSun"]    = df.get("st_mass", np.nan)

    # --- Label ---
    if "disposition" in df.columns:
        o["disposition"] = df["disposition"]
    elif "koi_disposition" in df.columns:
        o["disposition"] = df["koi_disposition"]
    else:
        o["disposition"] = np.nan

    return o


def main():
    # 1) Esnek dosya bulma (ad ufak farklıysa da yakalasın)
    try:
        kep = next(iter(sorted(RAW.glob("*cum*csv"))))
    except StopIteration:
        print("Kepler CSV bulunamadı (beklenen: cumulative_*.csv).", file=sys.stderr); sys.exit(1)
    try:
        k2  = next(iter(sorted(RAW.glob("*k2*csv"))))
    except StopIteration:
        print("Uyarı: K2 CSV bulunamadı (beklenen: k2pandc_*.csv). Sadece Kepler birleştirilecek.", file=sys.stderr)
        k2  = None

    dfk = read_csv(kep)
    parts = [unify_kepler(dfk)]
    if k2 is not None:
        df2 = read_csv(k2)
        parts.append(unify_k2(df2))

    merged = pd.concat(parts, ignore_index=True)

    # 2) mission’ı garanti doldur (NaN kalırsa isim ipucu kullan)
    # EPIC içeriyorsa K2 varsay
    guess_k2 = merged["object_name"].astype(str).str.contains(r"^EPIC|\bK2\b", flags=re.I, na=False)
    merged["mission"] = merged["mission"].astype("string")
    merged.loc[merged["mission"].isna() & guess_k2, "mission"] = "K2"
    merged["mission"] = merged["mission"].fillna("Kepler")

    # 3) sadece 3 sınıfı tut
    valid = {"FALSE POSITIVE","CANDIDATE","CONFIRMED"}
    merged = merged[merged["disposition"].isin(valid)].reset_index(drop=True)

    # 4) sayaçlar
    k_cnt = int((merged["mission"] == "Kepler").sum())
    k2_cnt = int((merged["mission"] == "K2").sum())
    print(f"Kepler rows: {k_cnt} | K2 rows: {k2_cnt}")

    out_csv = OUT / "exoplanets_merged_train.csv"
    merged.to_csv(out_csv, index=False)

    schema = {
        "columns": merged.columns.tolist(),
        "notes": [
            "K2 duration hours→days dönüştürüldü",
            "REFUTED ve boş etiketler çıkarıldı",
            "Kepler/K2 ortak şema",
            "mission kolonu concat sonrası fillna ile garanti dolduruldu"
        ]
    }
    (OUT/"exoplanets_merged_train.schema.json").write_text(json.dumps(schema, indent=2))
    print(f"[OK] saved {out_csv} shape={merged.shape}")

if __name__ == "__main__":
    main()
