# Meter loss screening

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=msj78598%2Fhes-loss-detector&branch=main&mainModule=streamlit_app.py)

Upload a three-phase readings export. Every reading is scored, rolled up to one
row per meter, and ranked. Download the result as a spreadsheet.

**Nothing is stored.** The uploaded file lives in memory for the length of the
request. No database, no session history, no logs of your data.

---

## What it looks for

Loss shows up as a **contradiction between voltage and current**, and the model
is built around that:

| Signature | What it means |
|---|---|
| Current on a phase whose voltage is zero | The voltage lead is not reading the load it carries |
| Healthy voltage on every phase, no current at all | Nothing metered — legitimate when idle, worth a look when it should not be |
| One phase carrying several times the others | Load bypassing the metered path, or a wiring fault |
| Two live phases, one dead, current on the dead one | The clearest single contradiction a reading can hold |

Twenty-eight features are computed from the six measured numbers of each
reading — per-unit voltage levels, NEMA imbalance, phase agreement, and the
share of load the meter actually measures.

## How a meter is raised

Scoring happens per **reading**; the decision happens per **meter**.

A meter's confidence combines three things:

1. **How high it scored** — the strongest single reading
2. **How consistently** — the share of its readings that were flagged
3. **How many readings** were scorable at all

| Confidence | Probability | Consistency | Readings |
|---|---|---|---|
| Very likely | ≥ 0.90 | ≥ 25 % | ≥ 8 |
| Likely | ≥ 0.70 | ≥ 15 % | ≥ 4 |
| Possible | ≥ 0.40 | ≥ 5 % | ≥ 1 |

One reading of 1.00 out of two hundred is a glitch. The same pattern in a
hundred and ninety is a standing condition. Collapsing the two into a single
probability throws that difference away.

**Idle readings are set aside, not scored.** Healthy voltage with no current is
a closed shop or an off-season pump — a normal operating state, not a finding.

## Input format

Three phase-to-neutral voltages and their currents. Column names are matched on
a normalised form, so both of these work:

```
HES Meter Id | R-N Phase average voltage [V] | ... | B Phase average current[A]
meter_id     | V1 | V2 | V3 | I1 | I2 | I3
```

A meter id column groups the readings. Without one, the whole file is treated as
a single meter. An optional timestamp column gives you the time of the reading
that raised each meter.

CSV and Excel (`.xlsx`, `.xls`) are both accepted. A file without the six
electrical columns is refused with a list of what it *does* contain — so you can
see at a glance whether you exported the wrong report.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy

Click the badge above, or go to [share.streamlit.io](https://share.streamlit.io)
and create an app with:

| Field | Value |
|---|---|
| Repository | `msj78598/hes-loss-detector` |
| Branch | `main` |
| Main file | `streamlit_app.py` |
| Python version | 3.12 |

No secrets to configure — the app has no credentials, no database, and no
outbound calls.

There is a synthetic sample file behind the **Sample file** button — generated
from the physics, not recorded from anywhere — so you can see the output shape
before pointing it at your own data.

## About the model

A k-nearest-neighbours classifier over the 28 features, wrapped in isotonic
calibration so the probabilities mean something. It ships in `model/model.pkl`.

`scikit-learn` is pinned in `requirements.txt` because the model is a pickle:
loading it under a different version can fail or, worse, load with changed
behaviour.

**What the file contains, stated plainly:** k-NN keeps its training set, so the
bundle holds 50,000 rows of *standardised feature vectors* — per-unit voltages,
imbalance ratios, and the like. There are no meter identifiers, no timestamps,
no coordinates, no account or subscriber data, and nothing that ties a row to a
premises. It is a numeric decision surface, not a record set.

## What this is not

This is a **screening** tool. It ranks meters by how strongly their electrical
pattern departs from normal — it does not establish a cause, and it cannot tell
tampering from a genuine fault. Both are loss; which one it is gets decided at
the meter, by someone standing in front of it.

Some patterns it raises are entirely legitimate — an old two-phase supply, or a
direct-connected meter whose loads sit on one phase because that is what the
subscriber needs. Local knowledge belongs in the loop.

## Licence

MIT — see [LICENSE](LICENSE).
