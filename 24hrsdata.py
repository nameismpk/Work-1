import numpy as np
import pandas as pd
from datetime import datetime
from math import sin, pi

# ============================================================
# USER-LOCKED TARGET MIX (must match exactly)
# ============================================================
TARGET_TYPE_COUNTS = {"1BHK": 35, "2BHK": 41, "3BHK": 12, "Commercial": 12}
TARGET_PV_YES_BY_TYPE = {"1BHK": 10, "2BHK": 12, "3BHK": 4, "Commercial": 4}

# ============================================================
# Season labels (TN/Theni) + factors
# ============================================================
def get_season(ts):
    m = ts.month
    if m in [12, 1, 2]:
        return "Winter"       # Dec-Feb
    if m in [3, 4, 5]:
        return "Summer"       # Mar-May
    if m in [6, 7, 8, 9]:
        return "SW_Monsoon"   # Jun-Sep
    return "NE_Monsoon"       # Oct-Nov

def season_factor(ts):
    """
    Returns load_factor, psh (peak sun hours/day), season_name
    """
    season = get_season(ts)
    if season == "Summer":
        return 1.10, 6.0, season
    if season == "SW_Monsoon":
        return 1.03, 5.2, season
    if season == "NE_Monsoon":
        return 1.05, 4.7, season
    return 0.98, 5.0, season  # Winter

def get_day_type(ts):
    return "Weekend" if ts.weekday() >= 5 else "Weekday"

def weekend_factor(building_type, ts):
    is_weekend = (ts.weekday() >= 5)
    if building_type == "Commercial":
        return 0.70 if is_weekend else 1.00
    return 1.05 if is_weekend else 1.00

# ============================================================
# PV season derating (temperature + dust + clouds)
# ============================================================
PV_DERATE_BY_SEASON = {
    "Summer": 0.92,
    "SW_Monsoon": 0.88,
    "NE_Monsoon": 0.86,
    "Winter": 0.97
}

def pv_derate_factor(season_name: str):
    return float(PV_DERATE_BY_SEASON.get(season_name, 0.92))

# ============================================================
# 24h base load shapes (normalized mean=1)
# ============================================================
def base_shape_24(building_type: str):
    if building_type in ["1BHK", "2BHK", "3BHK"]:
        s = np.array([
            0.25, 0.22, 0.20, 0.20, 0.22, 0.30,
            0.55, 0.70, 0.50, 0.35, 0.30, 0.28,
            0.30, 0.32, 0.35, 0.40, 0.55, 0.75,
            0.95, 1.00, 0.85, 0.65, 0.45, 0.32
        ])
        if building_type == "1BHK":
            s *= 0.98
        elif building_type == "3BHK":
            s *= 1.05
        return s / s.mean()

    if building_type == "Commercial":
        s = np.array([
            0.12, 0.10, 0.10, 0.10, 0.12, 0.15,
            0.25, 0.55, 0.85, 1.00, 0.98, 0.95,
            0.92, 0.90, 0.92, 0.98, 1.00, 0.75,
            0.45, 0.30, 0.22, 0.18, 0.15, 0.13
        ])
        return s / s.mean()

    return np.ones(24)

# ============================================================
# PV hourly bell-shape distribution
# ============================================================
def pv_hourly_shape(hour):
    if hour < 6 or hour > 18:
        return 0.0
    x = (hour - 6) / 12.0 * pi
    return sin(x)

BELL_SUM = sum(pv_hourly_shape(h) for h in range(24))

# ============================================================
# Scale day profile to match daily kWh and peak kW
# ============================================================
def enforce_daily_energy_and_peak(profile_24, daily_kwh, peak_kw, max_iter=25):
    p = profile_24.copy().astype(float)

    e = p.sum()
    if e <= 0:
        p = np.ones_like(p)
        e = p.sum()
    p *= (daily_kwh / e)

    for _ in range(max_iter):
        mx = p.max()
        if mx <= peak_kw + 1e-6:
            break

        p = np.minimum(p, peak_kw)
        e2 = p.sum()
        if e2 <= 0:
            break
        p *= (daily_kwh / e2)

        if p.max() > peak_kw * 1.20:
            p = np.minimum(p, peak_kw)
            break

    return p

# ============================================================
# Commercial holiday schedule
# ============================================================
def sample_holiday_dates(start_date, end_date, n_holidays, rng):
    all_days = pd.date_range(start=start_date, end=end_date, freq="D").date
    n_holidays = int(max(0, min(n_holidays, len(all_days))))
    chosen = rng.choice(all_days, size=n_holidays, replace=False)
    return set(chosen)

def holiday_load_factor(building_type, is_holiday, rng):
    if not is_holiday:
        return 1.00
    if building_type == "Commercial":
        return float(rng.uniform(0.30, 0.55))  # closed/partial
    return float(rng.uniform(0.98, 1.06))      # small change for residential

# ============================================================
# IMPORTANT: Enforce exact type and PV yes counts
# ============================================================
def enforce_exact_type_and_pv_counts(bdf, rng):
    # Validate type counts
    actual_type = bdf["Building_Type"].value_counts().to_dict()
    for t, req in TARGET_TYPE_COUNTS.items():
        if actual_type.get(t, 0) != req:
            raise ValueError(f"Type mix mismatch for {t}: actual={actual_type.get(t,0)} expected={req}")

    # Determine PV flag from PV_Type if present else PV_Capacity_kWp
    if "PV_Type" in bdf.columns:
        pv_flag = bdf["PV_Type"].astype(str).str.strip().str.lower().ne("no pv")
    else:
        pv_flag = bdf["PV_Capacity_kWp"].fillna(0).astype(float) > 0

    bdf = bdf.copy()
    bdf["_pv_flag"] = pv_flag

    # Defaults if we need to switch ON PV for some buildings
    default_kwp = {"1BHK": 1.5, "2BHK": 3.0, "3BHK": 5.0, "Commercial": 7.0}

    for t, k in TARGET_PV_YES_BY_TYPE.items():
        idxs = bdf.index[bdf["Building_Type"] == t].tolist()
        current_yes = int(bdf.loc[idxs, "_pv_flag"].sum())

        # Too many PV => turn OFF some
        if current_yes > k:
            yes_idxs = bdf.index[(bdf["Building_Type"] == t) & (bdf["_pv_flag"])].tolist()
            turn_off = rng.choice(yes_idxs, size=int(current_yes - k), replace=False)
            bdf.loc[turn_off, "PV_Capacity_kWp"] = 0.0
            if "PV_Type" in bdf.columns:
                bdf.loc[turn_off, "PV_Type"] = "No PV"
            bdf.loc[turn_off, "_pv_flag"] = False

        # Too few PV => turn ON some
        elif current_yes < k:
            no_idxs = bdf.index[(bdf["Building_Type"] == t) & (~bdf["_pv_flag"])].tolist()
            turn_on = rng.choice(no_idxs, size=int(k - current_yes), replace=False)
            bdf.loc[turn_on, "PV_Capacity_kWp"] = float(default_kwp[t])
            if "PV_Type" in bdf.columns:
                bdf.loc[turn_on, "PV_Type"] = "On-grid"
            bdf.loc[turn_on, "_pv_flag"] = True

    # Final check
    final_yes = {}
    for t in TARGET_PV_YES_BY_TYPE:
        final_yes[t] = int(bdf.loc[bdf["Building_Type"] == t, "_pv_flag"].sum())

    if final_yes != TARGET_PV_YES_BY_TYPE:
        raise ValueError(f"PV enforcement failed: got {final_yes}, expected {TARGET_PV_YES_BY_TYPE}")

    bdf.drop(columns=["_pv_flag"], inplace=True)
    return bdf

# ============================================================
# MAIN GENERATOR
# ============================================================
def generate_one_year_hourly_pv_load(
    input_xlsx,
    buildings_sheet="Buildings",
    start="2025-01-01 00:00:00",
    end="2025-12-31 23:00:00",
    pr=0.75,
    cloud_sigma=0.10,
    load_sigma=0.06,
    n_commercial_holidays=12,
    output_file="theni_100_buildings_1year_hourly_pv_load_FINAL.xlsx",
    seed=None
):
    if seed is None:
        seed = int(datetime.now().timestamp() * 1_000_000)
    rng = np.random.default_rng(seed)

    bdf = pd.read_excel(input_xlsx, sheet_name=buildings_sheet)

    # Ensure needed columns exist
    required = ["Building_ID", "Building_Type", "Peak_Load_kW", "Daily_Energy_kWh", "PV_Capacity_kWp"]
    missing = [c for c in required if c not in bdf.columns]
    if missing:
        raise ValueError(f"Missing columns in Buildings sheet: {missing}")

    # Enforce exact scenario: 35/41/12/12 and PV 10/12/4/4
    bdf = enforce_exact_type_and_pv_counts(bdf, rng=np.random.default_rng(12345))

    # Hourly timeline
    dt_index = pd.date_range(start=start, end=end, freq="h")

    # Holiday dates
    holiday_dates = sample_holiday_dates(pd.to_datetime(start).date(),
                                         pd.to_datetime(end).date(),
                                         n_commercial_holidays, rng)

    # Cache day profiles per building per day
    profile_cache = {}

    records = []

    for _, row in bdf.iterrows():
        bid = row["Building_ID"]
        btype = row["Building_Type"]
        peak_kw = float(row["Peak_Load_kW"])
        daily_kwh_base = float(row["Daily_Energy_kWh"])
        pv_kwp = float(row["PV_Capacity_kWp"])

        base24 = base_shape_24(btype)

        for ts in dt_index:
            season_name = get_season(ts)
            day_type = get_day_type(ts)
            is_holiday = (ts.date() in holiday_dates)

            load_factor, psh, _ = season_factor(ts)
            wfactor = weekend_factor(btype, ts)

            cache_key = (bid, ts.date())
            if cache_key not in profile_cache:
                day_noise = float(np.clip(rng.normal(1.0, 0.03), 0.90, 1.12))
                hfactor = holiday_load_factor(btype, is_holiday, rng)

                daily_kwh = daily_kwh_base * load_factor * wfactor * day_noise * hfactor

                noise24 = rng.normal(1.0, load_sigma, 24)
                noise24 = np.clip(noise24, 0.80, 1.25)

                raw24 = base24 * noise24
                raw24 = np.clip(raw24, 0.03, None)

                day_profile24 = enforce_daily_energy_and_peak(raw24, daily_kwh, peak_kw)

                profile_cache[cache_key] = {
                    "profile24": day_profile24,
                    "psh": float(psh),
                    "season": season_name,
                    "is_holiday": bool(is_holiday),
                    "holiday_factor": float(hfactor),
                    "pv_derate": pv_derate_factor(season_name),
                }

            cached = profile_cache[cache_key]
            load_kw = float(cached["profile24"][ts.hour])

            # PV
            if pv_kwp > 0:
                shape = pv_hourly_shape(ts.hour)
                bell_norm = (shape / BELL_SUM) if BELL_SUM > 0 else 0.0
                cloud = float(np.clip(rng.normal(1.0, cloud_sigma), 0.60, 1.05))
                pv_daily_kwh = pv_kwp * cached["psh"] * pr * cached["pv_derate"] * cloud
                pv_kw = pv_daily_kwh * bell_norm
            else:
                pv_kw = 0.0

            records.append({
                "Datetime": ts,
                "Season": cached["season"],
                "DayType": day_type,
                "IsHoliday": int(cached["is_holiday"]),
                "Month": int(ts.month),
                "Hour": int(ts.hour),
                "Building_ID": bid,
                "Building_Type": btype,
                "Load_kW": round(load_kw, 6),
                "PV_kW": round(float(pv_kw), 6),
                "Net_kW": round(load_kw - float(pv_kw), 6),
                "Load_Holiday_Factor": round(float(cached["holiday_factor"]), 4),
                "PV_Derate_Factor": round(float(cached["pv_derate"]), 4),
            })

    df = pd.DataFrame(records)

    # Annual summary (dt=1h => sum(kW) = kWh)
    summary = df.groupby(["Building_ID", "Building_Type"]).agg(
        Annual_Load_kWh=("Load_kW", "sum"),
        Annual_PV_kWh=("PV_kW", "sum"),
        Annual_Net_kWh=("Net_kW", "sum"),
        Holiday_Hours=("IsHoliday", "sum"),
    ).reset_index()

    meta = pd.DataFrame([
        {"key": "seed", "value": seed},
        {"key": "input_xlsx", "value": input_xlsx},
        {"key": "buildings_sheet", "value": buildings_sheet},
        {"key": "start", "value": start},
        {"key": "end", "value": end},
        {"key": "freq", "value": "H"},
        {"key": "Target_Type_Mix", "value": str(TARGET_TYPE_COUNTS)},
        {"key": "Target_PV_YES_By_Type", "value": str(TARGET_PV_YES_BY_TYPE)},
        {"key": "Seasons", "value": "Winter(Dec-Feb), Summer(Mar-May), SW_Monsoon(Jun-Sep), NE_Monsoon(Oct-Nov)"},
        {"key": "PR", "value": pr},
        {"key": "PV_DERATE_BY_SEASON", "value": str(PV_DERATE_BY_SEASON)},
        {"key": "n_commercial_holidays", "value": n_commercial_holidays},
        {"key": "load_sigma_slot", "value": load_sigma},
        {"key": "cloud_sigma", "value": cloud_sigma},
        {"key": "note", "value": "Annual kWh = sum of hourly kW. PV includes season derate + cloud factor. Commercial holidays reduce load."},
    ])

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Hourly_Long", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        meta.to_excel(writer, sheet_name="Meta", index=False)

    print("✅ FINAL 1-year hourly PV+Load generated!")
    print("Rows:", len(df))  # 100*8760 = 876,000
    print("Output:", output_file)
    print("Seed:", seed)

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    generate_one_year_hourly_pv_load(
        input_xlsx="theni_100_buildings_realistic_pv30.xlsx",  # ✅ your chosen file
        output_file="theni_100_buildings_1year_hourly_pv_load_FINAL.xlsx",
        start="2025-01-01 00:00:00",
        end="2025-12-31 23:00:00",
        n_commercial_holidays=12
    )

