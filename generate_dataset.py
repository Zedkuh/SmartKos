"""
generate_dataset.py
-------------------
Generates boarding_house_dataset.csv: 5000 rows of boarding house room states
following the UCI Household Power Consumption schema, augmented with synthetic
boarding house context columns that KosWatt reads directly.

Run once before launching the app:
    python generate_dataset.py

Output: boarding_house_dataset.csv  (same directory as this script)

Schema
------
date                  dd/mm/yyyy
time                  HH:MM:SS
global_active_power   kilowatts (derived from device state + noise)
temperature           °C  [22, 38]   — Singapore boarding house range
time_of_day           0 = daytime (06:00–18:00), 1 = nighttime
occupancy             0 = empty, 1 = occupied
occupancy_confidence  [0.10, 1.00]  — sensor reliability
ac_on                 0/1
ac_eco                0/1
lamp_on               0/1
tv_on                 0/1

Device state generation rationale
----------------------------------
Rows are seeded to include both efficient and wasteful configurations so the
agent has meaningful decisions to make.  Roughly 25–35% of empty-room rows
have at least one device running (the "wasteful" case the agent is designed to
detect and correct).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ── Constants — must match koswatt_agent.py ───────────────────────────────────
STANDBY_WATTS = 10
AC_HOT_WATTS  = 900
AC_COOL_WATTS = 700
AC_ECO_WATTS  = 500
LAMP_WATTS    = 40
TV_WATTS      = 120

N = 5000

# ── Timestamps (10-minute intervals starting 2024-01-01) ─────────────────────
start      = datetime(2024, 1, 1, 0, 0, 0)
timestamps = [start + timedelta(minutes=10 * i) for i in range(N)]
dates      = [t.strftime('%d/%m/%Y') for t in timestamps]
times_str  = [t.strftime('%H:%M:%S') for t in timestamps]
hours      = np.array([t.hour + t.minute / 60.0 for t in timestamps])

# ── Time of day (binary, matching the app radio widget) ──────────────────────
# Daytime:  06:00 – 18:00  →  0
# Nighttime: 18:00 – 06:00 →  1
is_night    = (hours >= 18) | (hours < 6)
time_of_day = is_night.astype(float)

# ── Temperature: Singapore range, peaks in early afternoon ───────────────────
base_temp   = 28 + 4 * np.sin(np.pi * (hours - 6) / 12)   # peaks ~14:00
temperature = (
    base_temp + np.random.normal(0, 1.5, N)
).clip(22, 38).round().astype(int)

# ── Occupancy: boarding house student schedule ────────────────────────────────
def _occ_prob(h):
    if h >= 18 or h < 1:   return 0.85   # evening / late night
    if h < 6:              return 0.60   # sleeping
    if h < 9:              return 0.50   # morning routine
    if h < 17:             return 0.20   # at class / work
    return 0.65                           # 17:00–18:00 returning

occ_probs = np.array([_occ_prob(h) for h in hours])
occupancy = (np.random.rand(N) < occ_probs).astype(int)

# ── Sensor confidence: lower during sleeping hours ────────────────────────────
base_conf = np.where((hours >= 1) & (hours < 7), 0.55, 0.90)
occ_conf  = (base_conf + np.random.normal(0, 0.07, N)).clip(0.10, 1.00).round(2)

# ── Device states ─────────────────────────────────────────────────────────────
rng = np.random.rand

# AC — on when occupied + hot (expected), or left on when empty (wasteful)
ac_on = (
      ((occupancy == 1) & (temperature >= 28) & (rng(N) < 0.80))   # expected use
    | ((occupancy == 1) & (temperature < 28)  & (rng(N) < 0.30))   # mild comfort
    | ((occupancy == 0) & (temperature >= 30) & (rng(N) < 0.30))   # wasteful: hot + empty
    | ((occupancy == 0) & (temperature < 30)  & (rng(N) < 0.12))   # wasteful: cool + empty
).astype(bool)

# AC ECO — student switches to ECO when warm but not scorching
ac_eco = (
    ac_on & (temperature >= 28) & (rng(N) < 0.40)
).astype(bool)

# Lamp — on at night when occupied; occasionally left on (wasteful)
lamp_on = (
      (is_night  & (occupancy == 1) & (rng(N) < 0.80))   # normal night use
    | (is_night  & (occupancy == 0) & (rng(N) < 0.20))   # wasteful: left on at night
    | (~is_night & (occupancy == 1) & (rng(N) < 0.08))   # unusual: daytime lamp
).astype(bool)

# TV — on in evenings when occupied; sometimes forgotten (wasteful)
tv_on = (
      (is_night  & (occupancy == 1) & (rng(N) < 0.65))   # evening TV
    | (is_night  & (occupancy == 0) & (rng(N) < 0.22))   # wasteful: left on
    | (~is_night & (occupancy == 1) & (rng(N) < 0.08))   # daytime TV: rare
).astype(bool)

# ── Global active power (kW): derived from device states + measurement noise ──
def _watts(ac, eco, lamp, tv, temp):
    ac_w = (AC_ECO_WATTS if eco else (AC_HOT_WATTS if temp > 30 else AC_COOL_WATTS)) if ac else 0
    return STANDBY_WATTS + ac_w + (LAMP_WATTS if lamp else 0) + (TV_WATTS if tv else 0)

raw_watts = np.array([
    _watts(bool(ac_on[i]), bool(ac_eco[i]), bool(lamp_on[i]), bool(tv_on[i]), temperature[i])
    for i in range(N)
], dtype=float)

global_active_power = (
    (raw_watts + np.random.normal(0, 15, N)).clip(10) / 1000
).round(3)

# ── Assemble and save ─────────────────────────────────────────────────────────
df = pd.DataFrame({
    'date'                 : dates,
    'time'                 : times_str,
    'global_active_power'  : global_active_power,
    'temperature'          : temperature,
    'time_of_day'          : time_of_day,
    'occupancy'            : occupancy,
    'occupancy_confidence' : occ_conf,
    'ac_on'                : ac_on.astype(int),
    'ac_eco'               : ac_eco.astype(int),
    'lamp_on'              : lamp_on.astype(int),
    'tv_on'                : tv_on.astype(int),
})

out = 'boarding_house_dataset.csv'
df.to_csv(out, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
wasteful_tv   = int(tv_on[occupancy == 0].sum())
wasteful_ac   = int(ac_on[occupancy == 0].sum())
wasteful_lamp = int((lamp_on & is_night & (occupancy == 0)).sum())

print(f"Generated {N} rows → {out}")
print(f"  Occupied rows          : {int(occupancy.sum())} ({100*occupancy.mean():.1f}%)")
print(f"  Wasteful: TV + empty   : {wasteful_tv}")
print(f"  Wasteful: AC + empty   : {wasteful_ac}")
print(f"  Wasteful: lamp+night+empty: {wasteful_lamp}")
