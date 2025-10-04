import pandas as pd
from pathlib import Path
from src.features.feature_builder import FeatureBuilder

INP = Path("data/interim/exoplanets_merged_train.csv")
OUT = Path("data/processed/exoplanets_final_features.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(INP)
    fb = FeatureBuilder(mode="extended", use_depth_unit=None, add_teff_bins=True).fit(df)
    feats = fb.transform(df)
    feats["disposition"] = df["disposition"]
    feats.to_csv(OUT, index=False)
    print(f"[OK] saved {OUT} shape={feats.shape}")

if __name__ == "__main__":
    main()
