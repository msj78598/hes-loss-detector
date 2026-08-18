"""
Reading the uploaded file — and refusing it clearly when it is the wrong one.

**Column names are matched, not assumed.** The same export lands as
`R-N Phase average voltage [V]` from one screen and plain `V1` from another, and
a report pulled with different settings reorders the columns entirely. Matching
on a normalised name keeps both working; matching on position breaks silently
the first time a column moves — and a silent break here means a plausible-looking
number computed from the wrong column.

**And a wrong file is rejected with what it actually holds.** "Missing columns"
alone leaves the question open: is the file damaged, exported from the wrong
screen, or a different report altogether? Listing what was found answers it in
one line.
"""
from __future__ import annotations

import io
from typing import BinaryIO

import pandas as pd

from detector.features import I_COLS, V_COLS

REQUIRED = V_COLS + I_COLS

# normalised source name → canonical name
ALIASES: dict[str, str] = {
    "meternumber": "meter_id", "hesmeterid": "meter_id", "meterid": "meter_id",
    "meterno": "meter_id", "meternumberid": "meter_id",

    "meterdatetime": "ts", "datetime": "ts", "timestamp": "ts",
    "date": "ts", "readingdatetime": "ts", "readtime": "ts",

    "v1": "v1", "v2": "v2", "v3": "v3",
    "rnphaseaveragevoltagev": "v1", "ynphaseaveragevoltagev": "v2",
    "bnphaseaveragevoltagev": "v3",
    "rnphaseaveragevoltage": "v1", "ynphaseaveragevoltage": "v2",
    "bnphaseaveragevoltage": "v3",
    "voltager": "v1", "voltagey": "v2", "voltageb": "v3",
    "vr": "v1", "vy": "v2", "vb": "v3",

    "i1": "i1", "i2": "i2", "i3": "i3", "a1": "i1", "a2": "i2", "a3": "i3",
    "rphaseaveragecurrenta": "i1", "yphaseaveragecurrenta": "i2",
    "bphaseaveragecurrenta": "i3",
    "rphaseaveragecurrent": "i1", "yphaseaveragecurrent": "i2",
    "bphaseaveragecurrent": "i3",
    "currentr": "i1", "currenty": "i2", "currentb": "i3",
    "ir": "i1", "iy": "i2", "ib": "i3",

    "office": "office", "city": "city",
}


class WrongFile(ValueError):
    """The upload is readable but is not a readings export."""


def normalise(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: ALIASES[normalise(c)] for c in df.columns
               if normalise(c) in ALIASES}
    out = df.rename(columns=mapping)
    return out.loc[:, ~out.columns.duplicated()]


def read(buffer: BinaryIO | bytes, filename: str) -> pd.DataFrame:
    """
    Read an uploaded readings file into canonical columns.

    Raises `WrongFile` with a readable explanation when the six electrical
    columns are not present.
    """
    if isinstance(buffer, bytes):
        buffer = io.BytesIO(buffer)
    name = str(filename).lower()

    if name.endswith(".csv"):
        raw = pd.read_csv(buffer, encoding="utf-8-sig", low_memory=False)
    else:
        raw = pd.read_excel(buffer)

    df = _rename(raw)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        found = ", ".join(str(c) for c in list(raw.columns)[:10])
        raise WrongFile(
            f"This file has no phase voltage and current columns "
            f"(missing: {', '.join(missing)}).\n\n"
            f"Expected three phase-to-neutral voltages and their currents — "
            f"full export names or short V1/V2/V3, I1/I2/I3 both work.\n\n"
            f"What the file actually contains: {found}"
            + (" …" if len(raw.columns) > 10 else ""))

    if "meter_id" not in df.columns:
        # One meter per file is legitimate — an id is only needed to group
        df["meter_id"] = "—"
    df["meter_id"] = df["meter_id"].astype(str).str.strip()

    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", format="mixed")

    keep = ["meter_id", *REQUIRED] + [c for c in ("ts", "office")
                                      if c in df.columns]
    df = df[keep].dropna(subset=REQUIRED, how="all")
    if df.empty:
        raise WrongFile("The columns are there but every reading is empty.")
    return df.reset_index(drop=True)
