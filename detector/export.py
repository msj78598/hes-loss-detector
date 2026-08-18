"""
Excel export — built in memory and handed straight to the browser.

Nothing touches the filesystem: the workbook is bytes from creation to download.
That is the whole storage policy of this app expressed in one module.

**Dead phases are highlighted, not just listed.** Whoever opens the sheet is
looking for the signature, and a red cell finds it faster than reading six
columns of decimals.
"""
from __future__ import annotations

import io

import pandas as pd

# (source column, sheet heading) — this order is the column order
COLUMNS: list[tuple[str, str]] = [
    ("rank", "#"),
    ("meter_id", "Meter"),
    ("tier", "Confidence"),
    ("probability", "Probability"),
    ("consistency", "Consistency"),
    ("flagged_readings", "Flagged readings"),
    ("scored", "Scored readings"),
    ("readings", "Total readings"),
    ("office", "Office"),
    ("peak_time", "Peak reading time"),
    ("peak_v1", "V1"), ("peak_v2", "V2"), ("peak_v3", "V3"),
    ("peak_i1", "I1"), ("peak_i2", "I2"), ("peak_i3", "I3"),
]

NOTE = ("This is a screening result, not a finding. A raised meter warrants "
        "inspection; only a site visit can confirm a cause.")


def build_table(res: pd.DataFrame) -> pd.DataFrame:
    pairs = [(src, head) for src, head in COLUMNS if src in res.columns]
    out = res[[s for s, _ in pairs]].rename(columns=dict(pairs))
    if "Peak reading time" in out.columns:
        out["Peak reading time"] = (pd.to_datetime(out["Peak reading time"],
                                                   errors="coerce")
                                    .dt.strftime("%Y-%m-%d %H:%M"))
    return out.reset_index(drop=True)


def to_excel(res: pd.DataFrame, source_name: str = "") -> bytes:
    table = build_table(res)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        table.to_excel(xw, sheet_name="Results", index=False, startrow=2)
        wb, ws = xw.book, xw.sheets["Results"]

        title = wb.add_format({"bold": True, "font_size": 13,
                               "font_color": "#0B2A6B"})
        sub = wb.add_format({"font_size": 9, "font_color": "#64748B",
                             "italic": True})
        head = wb.add_format({"bold": True, "bg_color": "#0B2A6B",
                              "font_color": "#FFFFFF", "border": 1,
                              "align": "center", "valign": "vcenter",
                              "text_wrap": True})
        dead = wb.add_format({"bg_color": "#FEE2E2", "font_color": "#B91C1C",
                              "num_format": "0.000"})
        pct = wb.add_format({"num_format": "0.0%"})
        num3 = wb.add_format({"num_format": "0.000"})

        ws.write(0, 0, "Meter loss screening — results", title)
        ws.write(1, 0, f"{len(table)} meters"
                       + (f"  ·  source: {source_name}" if source_name else "")
                       + f"  ·  {NOTE}", sub)

        for c, name in enumerate(table.columns):
            ws.write(2, c, name, head)
            ws.set_column(c, c, max(11, min(len(str(name)) + 4, 22)))
        for name, fmt in (("Probability", num3), ("Consistency", pct)):
            if name in table.columns:
                c = list(table.columns).index(name)
                ws.set_column(c, c, 13, fmt)

        first_v = next((n for n, c in enumerate(table.columns) if c == "V1"),
                       None)
        if first_v is not None and len(table):
            ws.conditional_format(3, first_v, len(table) + 2, first_v + 5,
                                  {"type": "cell", "criteria": "<=",
                                   "value": 1e-6, "format": dead})
        ws.freeze_panes(3, 2)
        if len(table):
            ws.autofilter(2, 0, len(table) + 2, len(table.columns) - 1)
    return buf.getvalue()
