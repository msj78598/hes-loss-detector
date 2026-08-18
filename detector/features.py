"""
Per-reading features from three phase voltages and three phase currents.

Everything here is arithmetic on the six measured numbers — no lookups, no
reference data, no site information. The same reading always yields the same
features on any machine.

**Why the nominal voltage is inferred per reading, not averaged.** A meter that
has lost a phase reads e.g. (127, 127, 0). The mean of that is 84.7 V, which
sits closest to a 127 V nominal but distorts the per-unit scale of every phase.
Taking the highest *live* phase instead keeps the surviving phases at 1.0 pu and
shows the dead one as the anomaly it is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Possible phase-to-neutral nominal voltages on this network
NOMINALS = np.array([127.0, 230.0])

V_ZERO_EPS = 1e-6          # at or below this a phase counts as dead
I_ZERO_EPS = 1e-6
V_LOW_PU = 0.85            # below this a live phase counts as sagging
I_PRESENT = 0.05           # amperes — below this is measurement noise

V_COLS = ["v1", "v2", "v3"]
I_COLS = ["i1", "i2", "i3"]


def _unbalance(a: np.ndarray) -> np.ndarray:
    """NEMA definition: greatest deviation from the mean, over the mean."""
    m = a.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.abs(a - m[:, None]).max(axis=1) / m
    return np.nan_to_num(u, nan=0.0, posinf=0.0) * 100


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(a / b, nan=0.0, posinf=0.0)


def make_features(v: np.ndarray, i: np.ndarray) -> pd.DataFrame:
    """
    Build the feature table the model expects.

    `v` and `i` are (n, 3) arrays of volts and amperes in phase order R, Y, B.
    """
    v = np.asarray(v, dtype="float64")
    i = np.asarray(i, dtype="float64")

    with np.errstate(invalid="ignore"):
        v_live = np.where(v > V_ZERO_EPS, v, np.nan)
        v_ref = np.nanmax(v_live, axis=1)
    v_ref = np.where(np.isfinite(v_ref), v_ref, NOMINALS[0])
    nominal = NOMINALS[np.argmin(np.abs(v_ref[:, None] - NOMINALS[None, :]),
                                 axis=1)]
    pu = v / nominal[:, None]

    v_zero = v <= V_ZERO_EPS
    i_zero = i <= I_ZERO_EPS
    pu_min, pu_max = pu.min(axis=1), pu.max(axis=1)
    i_min, i_max = i.min(axis=1), i.max(axis=1)
    p_phase = v * i

    # ── Current on dead phases ──────────────────────────────────────────
    # Apparent power alone is blind to a cut voltage lead: the phase carrying
    # current reads zero volts so their product is zero, and the phase with
    # healthy voltage carries no current. The sum then looks exactly like an
    # idle meter. These features measure how much current flows on phases whose
    # voltage is gone, and value it at the nominal voltage — what would have
    # been metered had the lead not been cut.
    #
    # `metered_share` is the summary: the fraction of load the meter actually
    # measures. It is 1.0 on a healthy meter, 0.0 when every ampere flows on a
    # dead phase, and in between when one lead of three is lost.
    i_dead = np.where(v_zero, i, 0.0).sum(axis=1)
    i_total = i.sum(axis=1)
    p_missing = i_dead * nominal
    p_measured = p_phase.sum(axis=1)

    return pd.DataFrame({
        "v1": v[:, 0], "v2": v[:, 1], "v3": v[:, 2],
        "i1": i[:, 0], "i2": i[:, 1], "i3": i[:, 2],

        "v_pu_mean": pu.mean(axis=1), "v_pu_min": pu_min, "v_pu_max": pu_max,
        "v_pu_range": pu_max - pu_min, "v_pu_std": pu.std(axis=1),
        "v_unbalance": _unbalance(v),
        "n_v_zero": v_zero.sum(axis=1),
        "n_v_low": ((pu < V_LOW_PU) & ~v_zero).sum(axis=1),
        "v_min_over_max": _ratio(pu_min, pu_max),

        "i_mean": i.mean(axis=1), "i_min": i_min, "i_max": i_max,
        "i_range": i_max - i_min, "i_std": i.std(axis=1),
        "i_unbalance": _unbalance(i),
        "n_i_zero": i_zero.sum(axis=1),
        "load_factor": _ratio(i.mean(axis=1), i_max),
        "i_min_over_max": _ratio(i_min, i_max),

        # healthy voltage on two or more phases with no current at all
        "live_v_no_i": (((pu > V_LOW_PU).sum(axis=1) >= 2)
                        & (i_zero.sum(axis=1) == 3)).astype(float),
        # voltage missing while current flows — the cut-lead signature
        "zero_v_with_i": ((v_zero.sum(axis=1) > 0)
                          & (i_max > I_PRESENT)).astype(float),
        # do dead voltages line up with dead currents? separates a lost supply
        # phase from a phase that simply carries no load
        "phase_match": (v_zero == i_zero).sum(axis=1).astype(float),
        # phases with zero volts and current flowing — the sharpest single
        # contradiction a reading can hold
        "n_phase_v0_i_ok": (v_zero & (i > I_PRESENT)).sum(axis=1).astype(float),

        "power_proxy": p_measured,
        "power_unbalance": _unbalance(p_phase),

        "i_dead": i_dead,
        "i_dead_share": _ratio(i_dead, i_total),
        "power_missing": p_missing,
        "metered_share": np.where(p_measured + p_missing > 1e-9,
                                  p_measured / (p_measured + p_missing), 1.0),
    })


def idle_mask(v: np.ndarray, i: np.ndarray, v_min_pu: float = 0.9,
              i_max_a: float = I_PRESENT) -> np.ndarray:
    """
    Readings where the supply is healthy and nothing is drawing.

    Kept apart from anomalies on purpose: an idle meter is a normal operating
    state — a closed shop, an off-season farm — not evidence of anything. Scoring
    it would fill the results with premises that simply were not consuming.
    """
    v = np.asarray(v, dtype="float64")
    i = np.asarray(i, dtype="float64")
    with np.errstate(invalid="ignore"):
        v_live = np.where(v > V_ZERO_EPS, v, np.nan)
        v_ref = np.nanmax(v_live, axis=1)
    v_ref = np.where(np.isfinite(v_ref), v_ref, NOMINALS[0])
    nominal = NOMINALS[np.argmin(np.abs(v_ref[:, None] - NOMINALS[None, :]),
                                 axis=1)]
    return ((v / nominal[:, None] >= v_min_pu).all(axis=1)
            & (i <= i_max_a).all(axis=1))
