# ============================================
# REAL DATASET BUILDER (Colab): covid19dh -> CSV
# Creates: us_states_covid19dh.csv
# ============================================

!pip -q install covid19dh

import pandas as pd
import numpy as np
from datetime import date
from covid19dh import covid19
from google.colab import files

# ---- Choose time range (edit if you like) ----
START = date(2020, 3, 1)
END   = date(2021, 12, 31)

# ---- Download USA state-level data (level=2) ----
x, src = covid19("USA", level=2, start=START, end=END, raw=False)

print("Downloaded rows:", len(x))
print("Columns:", list(x.columns))
print("\nData sources returned by covid19dh (top):")
display(src.head())

# ---- Harmonize column names safely ----
# covid19dh typically returns columns like: date, iso_alpha_2, administrative_area_level_2,
# confirmed, deaths, recovered, population, etc. (exact names may vary by vintage/source).
df = x.copy()

# Ensure date column
if "date" not in df.columns:
    raise ValueError("No 'date' column found. Please print columns and share here.")

df["date"] = pd.to_datetime(df["date"])

# Region name: try common fields used by covid19dh
region_candidates = [
    "administrative_area_level_2",  # common for states/regions
    "administrative_area_level_1",
    "name",
    "region",
    "state"
]
region_col = next((c for c in region_candidates if c in df.columns), None)
if region_col is None:
    raise ValueError(f"Could not find a region column among: {region_candidates}. Columns are: {list(df.columns)}")

df = df.rename(columns={region_col: "region"})

# Identify confirmed/deaths/recovered/active/population columns if present
def pick_col(cands):
    return next((c for c in cands if c in df.columns), None)

confirmed_col = pick_col(["confirmed", "cases", "cumulative_confirmed", "cum_confirmed"])
deaths_col    = pick_col(["deaths", "cumulative_deaths", "cum_deaths"])
recovered_col = pick_col(["recovered", "cumulative_recovered", "cum_recovered"])
active_col    = pick_col(["active", "active_cases"])
pop_col       = pick_col(["population", "pop"])

if confirmed_col is None:
    raise ValueError("No confirmed/cases column found in covid19dh output. Please share the printed columns.")

# Fill missing numeric columns with 0 where appropriate
for c in [confirmed_col, deaths_col, recovered_col, active_col]:
    if c is not None:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Compute recovered (if absent -> 0)
if recovered_col is None:
    df["recovered"] = 0.0
else:
    df["recovered"] = df[recovered_col].fillna(0.0)

# Compute infected (prefer active if available; else confirmed - deaths - recovered)
if active_col is not None:
    df["infected"] = df[active_col].fillna(0.0)
else:
    d = df[deaths_col].fillna(0.0) if deaths_col is not None else 0.0
    df["infected"] = (df[confirmed_col].fillna(0.0) - d - df["recovered"]).clip(lower=0.0)

# Population (if missing -> 1 so model can run in normalized units)
if pop_col is None:
    df["population"] = 1.0
else:
    df["population"] = pd.to_numeric(df[pop_col], errors="coerce").fillna(1.0)

# Mobility feature: covid19dh is not a mobility database; set a neutral constant (truthful).
df["mobility"] = 1.0

# Keep only needed columns
out = df[["date", "region", "infected", "recovered", "population", "mobility"]].copy()

# Clean region names, drop null region rows
out["region"] = out["region"].astype(str).str.strip()
out = out[out["region"].notna() & (out["region"] != "")]

# Optional: remove aggregates (if any show up)
bad = {"Unknown", "US", "United States", "USA"}
out = out[~out["region"].isin(bad)]

# Ensure sorted
out = out.sort_values(["region", "date"]).reset_index(drop=True)

# Save
csv_path = "/content/us_states_covid19dh.csv"
out.to_csv(csv_path, index=False)
print("Saved:", csv_path, "shape:", out.shape)

display(out.head(10))
display(out.tail(10))

# Download to your computer (so you can upload to your main Colab notebook if you want)
files.download(csv_path)
