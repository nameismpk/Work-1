import random
from datetime import datetime
import pandas as pd
from math import floor

# -----------------------------
# UPDATED ASSUMPTIONS
# -----------------------------
ASSUMPTIONS = {
    "1BHK": {
        "occupancy": (1, 3),
        "peak_kw": (1.2, 3.0),
        "daily_kwh": (3.0, 8.0),
        "roof_m2": (18, 40),
        "pv_usable_frac": (0.45, 0.70),
        "pv_cap_kwp": 2.0
    },
    "2BHK": {
        "occupancy": (2, 5),
        "peak_kw": (2.0, 5.0),
        "daily_kwh": (6.0, 14.0),
        "roof_m2": (30, 70),
        "pv_usable_frac": (0.50, 0.75),
        "pv_cap_kwp": 4.0
    },
    "3BHK": {
        "occupancy": (3, 7),
        "peak_kw": (3.5, 8.0),
        "daily_kwh": (10.0, 22.0),
        "roof_m2": (60, 120),
        "pv_usable_frac": (0.55, 0.80),
        "pv_cap_kwp": 6.0
    },
    "Commercial": {
        "occupancy": (3, 15),      # ✅ max 15
        "peak_kw": (10.0, 60.0),
        "daily_kwh": (60.0, 350.0),
        "roof_m2": (120, 900),
        "pv_usable_frac": (0.35, 0.65),
        "pv_cap_kwp": 10.0         # ✅ max 10 kWp
    },
}

# PV modeling constants
PV_KW_PER_M2 = 0.18
PERFORMANCE_RATIO = 0.75
PSH_RANGE = (4.5, 6.0)

# Type mix weights
TYPE_WEIGHTS = {
    "1BHK": 0.30,
    "2BHK": 0.45,
    "3BHK": 0.15,
    "Commercial": 0.10
}

PV_SHARE_PER_TYPE = 0.30  # ✅ only 30% PV in each type

def randi(rng, a, b):
    return rng.randint(int(a), int(b))

def randf(rng, a, b, nd=2):
    return round(rng.uniform(float(a), float(b)), nd)

def generate_theni_100_buildings_excel(
    n=100,
    out_xlsx="theni_100_buildings_realistic_pv30.xlsx",
    weights=None,
):
    seed = int(datetime.now().timestamp() * 1_000_000)
    rng = random.Random(seed)

    if weights is None:
        weights = TYPE_WEIGHTS

    types = list(weights.keys())
    probs = [weights[t] for t in types]

    # Step 1: Generate building types first (so we can enforce 30% PV per type)
    building_types = [rng.choices(types, weights=probs, k=1)[0] for _ in range(n)]

    # Step 2: For each type, choose exactly ~30% indices to have PV
    pv_flags = [False] * n
    type_to_indices = {t: [] for t in types}
    for idx, t in enumerate(building_types):
        type_to_indices[t].append(idx)

    pv_counts_by_type = {}
    for t, idxs in type_to_indices.items():
        N = len(idxs)
        k = int(round(PV_SHARE_PER_TYPE * N))  # exactly ~30% per type
        k = max(0, min(k, N))
        chosen = rng.sample(idxs, k) if k > 0 else []
        for c in chosen:
            pv_flags[c] = True
        pv_counts_by_type[t] = k

    # Step 3: Generate per-building parameters
    rows = []
    counts = {t: 0 for t in types}
    pv_yes_total = 0
    pv_no_total = 0

    for i in range(n):
        btype = building_types[i]
        counts[btype] += 1
        a = ASSUMPTIONS[btype]

        occ = randi(rng, *a["occupancy"])
        peak_kw = randf(rng, *a["peak_kw"], 2)
        daily_kwh = randf(rng, *a["daily_kwh"], 2)

        roof_m2 = randi(rng, *a["roof_m2"])
        usable_frac = randf(rng, *a["pv_usable_frac"], 2)
        usable_roof_m2 = round(roof_m2 * usable_frac, 2)

        pv_possible_kwp = round(usable_roof_m2 * PV_KW_PER_M2, 2)
        pv_cap = float(a["pv_cap_kwp"])

        has_pv = pv_flags[i]
        if has_pv:
            pv_yes_total += 1
            pv_kwp = round(min(pv_possible_kwp, pv_cap), 2)
            pv_type = "On-grid"
        else:
            pv_no_total += 1
            pv_kwp = 0.0
            pv_type = "No PV"

        psh = randf(rng, *PSH_RANGE, 2)
        pv_daily_kwh = round(pv_kwp * psh * PERFORMANCE_RATIO, 2)

        rows.append({
            "Building_ID": f"THENI_BLD_{i+1:03d}",
            "District": "Theni",
            "Building_Type": btype,
            "Occupancy_People": occ,
            "Peak_Load_kW": peak_kw,
            "Daily_Energy_kWh": daily_kwh,
            "Roof_Area_m2": roof_m2,
            "PV_Usable_Fraction": usable_frac,
            "PV_Usable_Area_m2": usable_roof_m2,
            "PV_Type": pv_type,
            "PV_Capacity_kWp": pv_kwp,
            "PSH_used_hours": psh,
            "PV_Daily_Gen_kWh": pv_daily_kwh,
        })

    buildings_df = pd.DataFrame(rows)

    # Assumptions sheet
    assumptions_rows = []
    for t, vals in ASSUMPTIONS.items():
        assumptions_rows.append({
            "Type": t,
            "Occupancy_range": f"{vals['occupancy'][0]}–{vals['occupancy'][1]}",
            "Peak_kW_range": f"{vals['peak_kw'][0]}–{vals['peak_kw'][1]}",
            "Daily_kWh_range": f"{vals['daily_kwh'][0]}–{vals['daily_kwh'][1]}",
            "Roof_m2_range": f"{vals['roof_m2'][0]}–{vals['roof_m2'][1]}",
            "PV_usable_fraction_range": f"{vals['pv_usable_frac'][0]}–{vals['pv_usable_frac'][1]}",
            "PV_cap_kWp": vals["pv_cap_kwp"],
        })
    assumptions_df = pd.DataFrame(assumptions_rows)

    # Summary sheet
    summary_rows = [
        {"Metric": "Seed_used", "Value": seed},
        {"Metric": "Total_buildings", "Value": n},
        {"Metric": "PV_share_per_type", "Value": PV_SHARE_PER_TYPE},
        {"Metric": "PV_yes_total", "Value": pv_yes_total},
        {"Metric": "PV_no_total", "Value": pv_no_total},
        {"Metric": "Total_Daily_Load_kWh (sum)", "Value": round(buildings_df["Daily_Energy_kWh"].sum(), 2)},
        {"Metric": "Total_PV_Capacity_kWp (sum)", "Value": round(buildings_df["PV_Capacity_kWp"].sum(), 2)},
        {"Metric": "Total_PV_Daily_Gen_kWh (sum)", "Value": round(buildings_df["PV_Daily_Gen_kWh"].sum(), 2)},
        {"Metric": "PV_KW_PER_M2", "Value": PV_KW_PER_M2},
        {"Metric": "Performance_Ratio", "Value": PERFORMANCE_RATIO},
        {"Metric": "PSH_range_hours", "Value": f"{PSH_RANGE[0]}–{PSH_RANGE[1]}"},
    ]

    # Add counts per type + PV per type
    for t in types:
        summary_rows.append({"Metric": f"Count_{t}", "Value": counts.get(t, 0)})
        summary_rows.append({"Metric": f"PV_yes_{t}", "Value": pv_counts_by_type.get(t, 0)})

    summary_df = pd.DataFrame(summary_rows)

    # Write Excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        buildings_df.to_excel(writer, sheet_name="Buildings", index=False)
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"✅ Excel created: {out_xlsx}")
    print("Seed:", seed)
    print("Type mix:", counts)
    print("PV yes by type:", pv_counts_by_type)
    print("PV yes/no total:", pv_yes_total, "/", pv_no_total)

if __name__ == "__main__":
    generate_theni_100_buildings_excel()
