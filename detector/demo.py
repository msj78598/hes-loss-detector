"""
A synthetic sample file, so the app can be tried without real data.

Every number here comes from a random generator seeded in this module. No
recorded reading, no real meter identifier, and no site is involved — the
patterns are written from the physics they are meant to show.

Which is also why the sample is honest about its limits: it demonstrates the
shape of the output, not the accuracy of the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NOMINAL = 230.0
PATTERNS = ("healthy", "idle", "one_phase_lost", "two_phases_lost",
            "current_on_dead_phase", "unbalanced")


def _reading(kind: str, rng: np.random.Generator) -> tuple[list, list]:
    v = rng.normal(NOMINAL, 2.5, 3).clip(0)
    base = float(rng.uniform(0.4, 9.0))
    i = (base * rng.normal(1.0, 0.05, 3)).clip(0)

    if kind == "healthy":
        pass
    elif kind == "idle":
        i = np.zeros(3)
    elif kind == "one_phase_lost":
        v[rng.integers(3)] = 0.0
        i[v <= 0] = 0.0
    elif kind == "two_phases_lost":
        dead = rng.choice(3, 2, replace=False)
        v[dead] = 0.0
        i[dead] = 0.0
    elif kind == "current_on_dead_phase":
        # the cut-lead signature: no voltage, current still flowing
        k = int(rng.integers(3))
        v[k] = 0.0
        i[k] = base * float(rng.uniform(0.8, 1.3))
    elif kind == "unbalanced":
        k = int(rng.integers(3))
        i[k] = base * float(rng.uniform(3.0, 6.0))

    return [round(float(x), 2) for x in v], [round(float(x), 3) for x in i]


def sample_readings(meters: int = 60, per_meter: int = 24,
                    seed: int = 7) -> pd.DataFrame:
    """Build a demo readings table in the same shape a real export has."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-06-01 00:00")
    rows = []
    for n in range(meters):
        kind = PATTERNS[n % len(PATTERNS)]
        mid = f"DEMO{n:04d}"
        for t in range(per_meter):
            # a meter drifts in and out of its condition rather than holding it
            # every hour — a constant fault would make consistency meaningless
            k = kind if rng.random() < 0.75 else "healthy"
            v, i = _reading(k, rng)
            rows.append({
                "HES Meter Id": mid,
                "Meter Datetime": start + pd.Timedelta(hours=t),
                "R-N Phase average voltage [V]": v[0],
                "Y-N Phase average voltage [V]": v[1],
                "B-N Phase average voltage [V]": v[2],
                "R Phase average current[A]": i[0],
                "Y Phase average current[A]": i[1],
                "B Phase average current[A]": i[2],
            })
    return pd.DataFrame(rows)


def sample_bytes() -> bytes:
    """The demo file as .xlsx bytes, ready to hand to a download button."""
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        sample_readings().to_excel(xw, index=False, sheet_name="Readings")
    return buf.getvalue()
