"""
Scoring readings and rolling them up to one row per meter.

**Confidence is not the probability.** A meter that scores 1.0 on one reading out
of two hundred is not the same as one whose pattern holds across a hundred and
ninety. The first is a glitch, the second is a standing condition. So the tier
combines three things: how high the probability got, how *consistently* the
pattern repeated, and how many readings were scorable at all.

**Nothing is stored.** The uploaded file lives in memory for the length of one
request and is never written to disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from detector.features import I_COLS, V_COLS, idle_mask, make_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"

# (label, min probability, min share of flagged readings, min scored readings)
TIERS: list[tuple[str, float, float, int]] = [
    ("Very likely", 0.90, 0.25, 8),
    ("Likely", 0.70, 0.15, 4),
    ("Possible", 0.40, 0.05, 1),
]
WEAK = "Weak"
UNSCORED = "Not scorable"
IDLE = "Idle"

TIER_ORDER = [t[0] for t in TIERS] + [WEAK, IDLE, UNSCORED]


def load_model(path: Path | None = None) -> dict[str, Any]:
    """
    Load the bundle once per process.

    Cached by the caller with `st.cache_resource` — a model is a shared handle,
    not per-user data, and reloading eight megabytes on every rerun makes the
    app feel broken.
    """
    import joblib
    return joblib.load(path or MODEL_PATH)


def score_readings(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """
    Probability per reading. NaN where the reading cannot be scored.

    Unscorable and idle are separated on purpose: the first is a data problem,
    the second is a normal operating state. Collapsing them hides the difference
    between "we could not look" and "there was nothing to see".
    """
    v = df[V_COLS].to_numpy(dtype="float64")
    i = df[I_COLS].to_numpy(dtype="float64")
    ok = np.isfinite(v).all(axis=1) & np.isfinite(i).all(axis=1)

    p = np.full(len(df), np.nan)
    if not ok.any():
        return p

    idle = np.zeros(len(df), dtype=bool)
    idle[ok] = idle_mask(v[ok], i[ok])
    live = ok & ~idle
    if live.any():
        X = make_features(v[live], i[live])[bundle["features"]]
        model = bundle["model"]
        if bundle.get("space") == "logit":
            z = model.predict(X)
            p[live] = 1.0 / (1.0 + np.exp(-z))
        else:
            p[live] = model.predict_proba(X)[:, 1]
    return p


def _tier(prob: float, share: float, scored: int, idle: int) -> str:
    if scored == 0:
        return IDLE if idle > 0 else UNSCORED
    for label, p_min, s_min, n_min in TIERS:
        if prob >= p_min and share >= s_min and scored >= n_min:
            return label
    return WEAK


def analyse(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """One row per meter, ordered by how strongly it stands out."""
    work = df.copy()
    work["p"] = score_readings(work, bundle)
    work["flag"] = work["p"] >= float(bundle["threshold"])

    v = work[V_COLS].to_numpy(dtype="float64")
    i = work[I_COLS].to_numpy(dtype="float64")
    ok = np.isfinite(v).all(axis=1) & np.isfinite(i).all(axis=1)
    work["idle"] = False
    if ok.any():
        work.loc[ok, "idle"] = idle_mask(v[ok], i[ok])

    g = work.groupby("meter_id", observed=True)
    res = pd.DataFrame({
        "probability": g["p"].max(),
        "readings": g["p"].size(),
        "scored": g["p"].count(),
        "flagged_readings": g["flag"].sum().astype(int),
        "idle_readings": g["idle"].sum().astype(int),
    })
    res["consistency"] = (res["flagged_readings"]
                          / res["scored"].replace(0, np.nan)).fillna(0.0)

    res["tier"] = [
        _tier(0.0 if pd.isna(p) else float(p), float(s), int(n), int(idl))
        for p, s, n, idl in zip(res["probability"], res["consistency"],
                                res["scored"], res["idle_readings"])
    ]

    # The reading that raised the meter — not its average.
    # An average hides the signature: a meter that loses a phase in half its
    # readings averages to roughly half nominal, which reads as a mild sag.
    # The peak reading shows the outright zero.
    peak = work.loc[work.groupby("meter_id", observed=True)["p"]
                    .idxmax().dropna()]
    cols = {c: f"peak_{c}" for c in V_COLS + I_COLS}
    if "ts" in peak.columns:
        cols["ts"] = "peak_time"
    res = res.join(peak.set_index("meter_id")[list(cols)].rename(columns=cols))
    if "office" in work.columns:
        res = res.join(g["office"].agg(
            lambda s: s.dropna().iloc[0] if s.notna().any() else None))

    res = res.reset_index()
    order = {t: n for n, t in enumerate(TIER_ORDER)}
    res["_o"] = res["tier"].map(order)
    res = (res.sort_values(["_o", "probability", "consistency"],
                           ascending=[True, False, False])
           .drop(columns="_o").reset_index(drop=True))
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    return res


def summary(res: pd.DataFrame) -> dict[str, int]:
    counts = res["tier"].value_counts()
    out = {t: int(counts.get(t, 0)) for t in TIER_ORDER}
    out["meters"] = int(len(res))
    out["raised"] = sum(out[t[0]] for t in TIERS)
    return out
