"""
Meter loss screening — upload a readings export, get a ranked list.

One screen, one job: read the file, score every reading, roll it up per meter,
and let the result leave as a spreadsheet. Nothing is written to disk and nothing
is remembered between sessions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from detector import demo, export
from detector.reading_file import WrongFile, read
from detector.scoring import TIERS, analyse, load_model, summary

st.set_page_config(page_title="Meter loss screening",
                   page_icon=":material/bolt:", layout="wide")

TIER_COLOUR = {"Very likely": "red", "Likely": "orange", "Possible": "blue"}


@st.cache_resource(show_spinner=False)
def _model():
    return load_model()


# The file is the cache key, so re-running a filter never re-scores it.
@st.cache_data(show_spinner=False, max_entries=3)
def _analyse(payload: bytes, filename: str):
    readings = read(payload, filename)
    result = analyse(readings, _model())
    return readings, result


st.title("Meter loss screening")
st.caption("Upload a three-phase readings export. Every reading is scored, "
           "then rolled up to one row per meter. Files are held in memory for "
           "the length of the request and never stored.")

left, right = st.columns([2, 1], gap="large")
with left:
    up = st.file_uploader("Readings file", type=["xlsx", "xls", "csv"],
                          label_visibility="collapsed")
with right:
    st.download_button("Sample file", demo.sample_bytes(),
                       file_name="sample_readings.xlsx",
                       icon=":material/science:", width="stretch",
                       help="Synthetic data — generated, not recorded.")

if up is None:
    st.info("Expected columns: three phase-to-neutral voltages and their "
            "currents. Full export names or short `V1/V2/V3`, `I1/I2/I3` both "
            "work. A meter id column groups the readings; without one the whole "
            "file is treated as a single meter.",
            icon=":material/upload_file:")

    with st.expander("How a meter is raised", icon=":material/help:"):
        st.markdown(
            "**Every reading is scored** on the relationship between the three "
            "phase voltages and their currents — per-unit levels, imbalance, "
            "and above all *contradictions*: current flowing on a phase whose "
            "voltage is gone, or healthy voltage on every phase with no load "
            "at all.\n\n"
            "**Readings are then grouped by meter.** A meter's confidence "
            "combines how high it scored, how consistently the pattern "
            "repeated, and how many readings were scorable. A single high "
            "reading out of two hundred is a glitch; the same pattern in a "
            "hundred and ninety is a standing condition.\n\n"
            "**Idle readings are set aside, not scored.** Healthy voltage with "
            "no current is a closed shop or an off-season pump — a normal "
            "state, not a finding.")
    st.stop()

try:
    with st.spinner("Scoring readings…"):
        readings, res = _analyse(up.getvalue(), up.name)
except WrongFile as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()
except Exception as exc:                                   # noqa: BLE001
    st.error(f"Could not read this file — {exc}", icon=":material/error:")
    st.stop()

s = summary(res)

# ───────────────────────────────────────────────────────────── headline
cols = st.columns(5)
cols[0].metric("Meters", f"{s['meters']:,}",
               help=f"from {len(readings):,} readings")
cols[1].metric("Raised", f"{s['raised']:,}",
               delta=f"{s['raised'] / max(s['meters'], 1):.1%} of file",
               delta_color="off")
for c, (label, *_rest) in zip(cols[2:], TIERS):
    c.metric(label, f"{s[label]:,}")

st.badge(f"{s['Idle']:,} idle", icon=":material/pause_circle:", color="gray")

# ───────────────────────────────────────────────────────────── filters
raised_only = st.toggle("Raised meters only", value=True)
view = res[res["tier"].isin([t[0] for t in TIERS])] if raised_only else res

with st.container(horizontal=True):
    picked = st.pills("Confidence", [t[0] for t in TIERS],
                      selection_mode="multi", label_visibility="collapsed")
if picked:
    view = view[view["tier"].isin(picked)]

if view.empty:
    st.success("No meter in this file meets the screening thresholds.",
               icon=":material/check_circle:")
    st.stop()

# ───────────────────────────────────────────────────────────── results
show = ["rank", "meter_id", "tier", "probability", "consistency",
        "flagged_readings", "scored", "peak_time",
        "peak_v1", "peak_v2", "peak_v3", "peak_i1", "peak_i2", "peak_i3"]
if "office" in view.columns:
    show.insert(3, "office")

st.dataframe(
    view[[c for c in show if c in view.columns]],
    hide_index=True, height=460,
    column_config={
        "rank": st.column_config.NumberColumn("#", format="%d", width="small"),
        "meter_id": st.column_config.TextColumn("Meter", width="medium"),
        "tier": st.column_config.TextColumn("Confidence", width="small"),
        "office": st.column_config.TextColumn("Office", width="small"),
        "probability": st.column_config.ProgressColumn(
            "Probability", min_value=0.0, max_value=1.0, format="%.3f"),
        "consistency": st.column_config.NumberColumn(
            "Consistency", format="percent",
            help="Share of this meter's scorable readings that were flagged"),
        "flagged_readings": st.column_config.NumberColumn("Flagged",
                                                          format="%d"),
        "scored": st.column_config.NumberColumn("Scored", format="%d"),
        "peak_time": st.column_config.DatetimeColumn("Peak reading",
                                                     format="YYYY-MM-DD HH:mm"),
        **{f"peak_v{n}": st.column_config.NumberColumn(f"V{n}", format="%.1f")
           for n in (1, 2, 3)},
        **{f"peak_i{n}": st.column_config.NumberColumn(f"I{n}", format="%.3f")
           for n in (1, 2, 3)},
    })

st.caption("Voltage and current shown are the single reading that raised the "
           "meter, not its average — an average hides the signature.")

st.download_button(
    f"Download {len(view):,} rows as Excel", export.to_excel(view, up.name),
    file_name=f"screening_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    icon=":material/download:", type="primary")

# ───────────────────────────────────────────────────────────── breakdown
with st.expander("Distribution", icon=":material/bar_chart:"):
    a, b = st.columns(2)
    with a:
        st.caption("Meters by confidence")
        counts = (res["tier"].value_counts()
                  .rename_axis("Confidence").reset_index(name="Meters"))
        st.bar_chart(counts, x="Confidence", y="Meters", horizontal=True)
    with b:
        st.caption("Highest probability per meter")
        p = res["probability"].dropna()
        if len(p):
            hist, edges = np.histogram(p, bins=20, range=(0, 1))
            st.bar_chart(pd.DataFrame(
                {"Probability": np.round(edges[:-1], 2), "Meters": hist}),
                x="Probability", y="Meters")

st.caption(export.NOTE)
