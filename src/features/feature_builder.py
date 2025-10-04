import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    mode: 'minimal' | 'extended'
    use_depth_unit: None | 'ppm' | 'percent'  (None -> depth'ten rp/R* türetme)
    add_teff_bins: True/False
    """
    CORE_COLS = [
        "period_days","duration_days","depth","prad_re",
        "steff_K","srad_Rsun","smass_MSun","mission"
    ]

    def __init__(self, mode="minimal", use_depth_unit=None, add_teff_bins=False):
        assert mode in ("minimal","extended")
        assert use_depth_unit in (None,"ppm","percent")
        self.mode = mode
        self.use_depth_unit = use_depth_unit
        self.add_teff_bins = add_teff_bins
        self.cols_ = None

    def fit(self, X, y=None):
        feats = self._build(X.copy())
        self.cols_ = feats.columns.tolist()
        return self

    def transform(self, X):
        feats = self._build(X.copy())
        # sütun sırasını sabitle
        for c in self.cols_:
            if c not in feats: feats[c] = np.nan
        return feats[self.cols_]

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        # eksik temel kolonları hazırla
        for c in self.CORE_COLS:
            if c not in df: df[c] = np.nan

        # mission → onehot (K2=1, Kepler=0)
        mission = df["mission"].astype(str).str.lower()
        df["mission_onehot"] = mission.map({"k2":1,"kepler":0}).fillna(0).astype(int)

        out = df[[
            "period_days","duration_days","depth","prad_re",
            "steff_K","srad_Rsun","smass_MSun","mission_onehot"
        ]].copy()

        # log1p
        out["log_period"]   = np.log1p(out["period_days"])
        out["log_duration"] = np.log1p(out["duration_days"])
        out["log_depth"]    = np.log1p(np.abs(out["depth"]))
        out["log_prad"]     = np.log1p(out["prad_re"])

        # oranlar
        out["duration_over_period"]    = out["duration_days"] / (out["period_days"] + 1e-6)
        out["stellar_density_catalog"] = out["smass_MSun"] / (out["srad_Rsun"]**3 + 1e-6)

        # rp/R* (güvenli yol: prad/srad; 109.1 ~ R_sun/R_earth)
        out["rp_over_rstar"] = out["prad_re"] / (out["srad_Rsun"] * 109.1 + 1e-6)

        # depth'ten rp/R* (birim netse)
        if self.mode == "extended" and self.use_depth_unit:
            scale = 1e6 if self.use_depth_unit == "ppm" else 1e2
            out["rp_over_rstar_from_depth"] = np.sqrt(np.clip(out["depth"], 0, None) / (scale + 1e-12))

        # fizik-temelli
        out["a_over_rstar"] = (out["smass_MSun"]**(1/3)) * (out["period_days"]**(2/3)) / (out["srad_Rsun"] + 1e-6)
        out["duration_expected_proxy"] = (out["period_days"]/np.pi) * (1/(out["a_over_rstar"] + 1e-6))
        out["duration_anomaly"] = out["duration_days"] / (out["duration_expected_proxy"] + 1e-6)

        # bayraklar
        out["flag_giant_star"]        = (out["srad_Rsun"] > 3.0).astype(int)
        out["flag_implausible_prad"]  = (out["prad_re"] > 15.0).astype(int)
        out["flag_ultrashort_period"] = (out["period_days"] < 1.0).astype(int)
        out["flag_long_period"]       = (out["period_days"] > 300).astype(int)
        out["flag_long_duty"]         = (out["duration_over_period"] > 0.05).astype(int)

        # teff binleri (opsiyonel)
        teff_cols = []
        if self.add_teff_bins:
            bins = [-np.inf, 4000, 5300, 6000, np.inf]
            labs = ["M","K","G","F/A"]
            teff_bin = pd.cut(out["steff_K"], bins=bins, labels=labs)
            for lab in labs:
                col = f"teff_bin_{lab}"
                out[col] = (teff_bin == lab).astype(int)
                teff_cols.append(col)

        # minimal/extended seçimi
        minimal = [
            "period_days","duration_days","depth","prad_re","steff_K","srad_Rsun","smass_MSun","mission_onehot",
            "log_period","log_duration","log_depth",
            "duration_over_period","stellar_density_catalog",
            "flag_giant_star","flag_implausible_prad"
        ]
        extended_plus = [
            "log_prad","rp_over_rstar","a_over_rstar",
            "duration_expected_proxy","duration_anomaly",
            "flag_ultrashort_period","flag_long_period","flag_long_duty",
        ]
        if "rp_over_rstar_from_depth" in out: extended_plus.append("rp_over_rstar_from_depth")
        keep = minimal if self.mode == "minimal" else minimal + extended_plus + teff_cols
        return out[keep]
