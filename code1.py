# ===========================================================
# Virtual Power Plant Scheduling with PV + EV + Demand Response
# (TNEB / TNERC Time-of-Day EV Tariff Integrated)
# -----------------------------------------------------------
# Features:
# - PV + Load + EV fleet aggregation
# - V2G operation with SOC & DOD limits
# - Demand Response event and revenue
# - Objective: Minimize (energy cost + peak demand + degradation) - DR revenue
#
# Run directly  -> uses synthetic example data with TN ToD prices
# ===========================================================

import math
from dataclasses import dataclass
from typing import List, Dict

import pulp
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# EV data structure
# -------------------------------------------------------------------
@dataclass
class EV:
    name: str
    battery_kwh: float
    p_ch_max: float
    eta_ch: float
    eta_dis: float
    soc_arr: float
    soc_dep_req: float
    dod_max: float
    t_arr: int
    t_dep: int  # exclusive
    c_degradation: float


# -------------------------------------------------------------------
# Example profiles (with TNEB / TNERC ToD prices)
# -------------------------------------------------------------------
def generate_example_profiles(T: int = 24):
    """
    Generate profiles for:
    - load_profile[t]  (kW)
    - pv_profile[t]    (kW)
    - price_buy[t]     (₹/kWh)  -> TNERC EV ToD-style
    - price_sell[t]    (₹/kWh)
    - baseline_net[t]  (kW)
    - dr_window        list of DR time indices
    """

    # 1) Example load profile
    load_profile = []
    for t in range(T):
        if 0 <= t < 6:
            load_profile.append(40.0)      # night low
        elif 6 <= t < 9:
            load_profile.append(80.0)      # morning ramp
        elif 9 <= t < 17:
            load_profile.append(120.0)     # working hours
        elif 17 <= t < 22:
            load_profile.append(100.0)     # evening
        else:
            load_profile.append(60.0)      # late night

    # 2) PV profile (bell around noon)
    pv_profile = []
    pv_peak = 150.0  # kW
    for t in range(T):
        pv = pv_peak * math.exp(-((t - 12) ** 2) / (2 * (3.0 ** 2)))
        pv_profile.append(pv)

    # 3) TNEB / TNERC EV ToD-like price (approx structure)
    #    - Peak      : 06–09, 18–22  ->  ₹9.75 / kWh
    #    - Solar/off : 09–16        ->  ₹6.50 / kWh
    #    - Shoulder  : others       ->  ₹8.10 / kWh
    price_buy = []
    for t in range(T):
        if (6 <= t < 9) or (18 <= t < 22):
            price_buy.append(9.75)     # peak
        elif 9 <= t < 16:
            price_buy.append(6.50)     # solar / off-peak
        else:
            price_buy.append(8.10)     # shoulder / night

    # 4) Export tariff (simple assumption: 70% of min import)
    min_import = min(price_buy)
    export_tariff = 0.7 * min_import
    price_sell = [export_tariff for _ in range(T)]

    # 5) Baseline net load (no EV, no optimization)
    baseline_net = []
    for t in range(T):
        net = load_profile[t] - pv_profile[t]
        baseline_net.append(max(net, 0.0))

    # 6) DR window aligned with evening peak 18–22
    dr_window = list(range(18, 22))

    return load_profile, pv_profile, price_buy, price_sell, baseline_net, dr_window


# -------------------------------------------------------------------
# Example EV fleet
# -------------------------------------------------------------------
def generate_example_evs() -> List[EV]:
    evs = [
        # Office EV – parked 09–17
        EV(
            name="Office_EV_1",
            battery_kwh=50.0,
            p_ch_max=11.0,
            eta_ch=0.95,
            eta_dis=0.95,
            soc_arr=0.4,
            soc_dep_req=0.8,
            dod_max=0.8,
            t_arr=9,
            t_dep=17,
            c_degradation=2.0,
        ),
        # Residential EV – parked 19–24
        EV(
            name="Res_EV_1",
            battery_kwh=40.0,
            p_ch_max=7.4,
            eta_ch=0.95,
            eta_dis=0.95,
            soc_arr=0.3,
            soc_dep_req=0.7,
            dod_max=0.8,
            t_arr=19,
            t_dep=24,
            c_degradation=2.5,
        ),
        # Another office EV – parked 08–18
        EV(
            name="Office_EV_2",
            battery_kwh=60.0,
            p_ch_max=22.0,
            eta_ch=0.95,
            eta_dis=0.95,
            soc_arr=0.5,
            soc_dep_req=0.9,
            dod_max=0.8,
            t_arr=8,
            t_dep=18,
            c_degradation=1.8,
        ),
    ]
    return evs


# -------------------------------------------------------------------
# MILP builder & solver
# -------------------------------------------------------------------
def build_and_solve_vpp(
    load,
    pv,
    price_buy,
    price_sell,
    baseline_net,
    evs: List[EV],
    dr_window,
    delta_t: float = 1.0,
    pcc_limit_kw: float = 500.0,
    k_peak: float = 145.0 / 30.0,  # ~ ₹/kW per day (approx from monthly demand charge)
    lambda_dr: float = 10.0,       # ₹/kWh DR incentive
) -> Dict:
    T = len(load)
    time = range(T)

    prob = pulp.LpProblem("PV_EV_VPP_with_DR_TNEB", pulp.LpMinimize)

    # Grid & PV
    p_im = pulp.LpVariable.dicts("Grid_IM", time, lowBound=0)
    p_ex = pulp.LpVariable.dicts("Grid_EX", time, lowBound=0)
    p_pv = pulp.LpVariable.dicts("PV_used", time, lowBound=0)

    # DR reduction (can be +/- : positive = reduction, negative = over-consumption)
    r = pulp.LpVariable.dicts("DR_delta", time)

    # Peak demand variable
    P_peak = pulp.LpVariable("P_peak", lowBound=0)

    # EV variables
    p_ch, p_dis, soc = {}, {}, {}
    y_ch, y_dis = {}, {}

    for ev in evs:
        p_ch[ev.name] = pulp.LpVariable.dicts(f"p_ch_{ev.name}", time, lowBound=0)
        p_dis[ev.name] = pulp.LpVariable.dicts(f"p_dis_{ev.name}", time, lowBound=0)
        soc[ev.name] = pulp.LpVariable.dicts(f"soc_{ev.name}", time, lowBound=0, upBound=1)
        y_ch[ev.name] = pulp.LpVariable.dicts(f"y_ch_{ev.name}", time, cat="Binary")
        y_dis[ev.name] = pulp.LpVariable.dicts(f"y_dis_{ev.name}", time, cat="Binary")

    # ------------------ Constraints ------------------

    for t in time:
        # PCC limit and peak
        prob += p_im[t] <= pcc_limit_kw
        prob += p_ex[t] <= pcc_limit_kw
        prob += P_peak >= p_im[t]

        # PV availability
        prob += p_pv[t] <= pv[t]

        # Power balance
        prob += (
            p_pv[t] + sum(p_dis[ev.name][t] for ev in evs) + p_im[t]
            == load[t] + sum(p_ch[ev.name][t] for ev in evs) + p_ex[t]
        )

        # DR delta definition
        if t in dr_window:
            # r[t] = baseline_net - net_import
            prob += r[t] == baseline_net[t] - (p_im[t] - p_ex[t])
        else:
            prob += r[t] == 0

    # EV constraints
    for ev in evs:
        min_soc = 1.0 - ev.dod_max

        for t in time:
            # SOC evolution
            if t == 0:
                if t < ev.t_arr:
                    prob += soc[ev.name][t] == ev.soc_arr
                elif t == ev.t_arr:
                    prob += soc[ev.name][t] == ev.soc_arr
                else:  # t > arr, but t==0 can't happen, safe
                    prob += soc[ev.name][t] == ev.soc_arr
            else:
                if t < ev.t_arr:
                    prob += soc[ev.name][t] == ev.soc_arr
                elif ev.t_arr < t < ev.t_dep:
                    prob += soc[ev.name][t] == soc[ev.name][t - 1] \
                        + (ev.eta_ch * p_ch[ev.name][t - 1] * delta_t) / ev.battery_kwh \
                        - (p_dis[ev.name][t - 1] * delta_t) / (ev.eta_dis * ev.battery_kwh)
                elif t == ev.t_arr:
                    prob += soc[ev.name][t] == ev.soc_arr
                else:  # t >= ev.t_dep
                    prob += soc[ev.name][t] == soc[ev.name][t - 1]

            # SOC lower bound
            prob += soc[ev.name][t] >= min_soc

            # Connected or not
            if ev.t_arr <= t < ev.t_dep:
                prob += p_ch[ev.name][t] <= ev.p_ch_max * y_ch[ev.name][t]
                prob += p_dis[ev.name][t] <= ev.p_ch_max * y_dis[ev.name][t]
                prob += y_ch[ev.name][t] + y_dis[ev.name][t] <= 1
            else:
                prob += p_ch[ev.name][t] == 0
                prob += p_dis[ev.name][t] == 0
                prob += y_ch[ev.name][t] == 0
                prob += y_dis[ev.name][t] == 0

        # Departure SOC requirement
        dep_index = min(ev.t_dep - 1, T - 1)
        prob += soc[ev.name][dep_index] >= ev.soc_dep_req

    # ------------------ Objective ------------------

    energy_cost = sum(
        price_buy[t] * p_im[t] * delta_t - price_sell[t] * p_ex[t] * delta_t
        for t in time
    )

    peak_cost = k_peak * P_peak

    degradation_cost = 0
    for ev in evs:
        degradation_cost += ev.c_degradation * sum(
            p_dis[ev.name][t] * delta_t for t in time
        )

    dr_revenue = lambda_dr * sum(r[t] * delta_t for t in time)  # positive r => revenue; negative r => penalty

    prob += energy_cost + peak_cost + degradation_cost - dr_revenue

    # ------------------ Solve ------------------
    print("Solving MILP with TNEB-style ToD tariff...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print("Status:", pulp.LpStatus[prob.status])

    # ------------------ Collect Results ------------------
    results: Dict = {
        "load": load,
        "pv": pv,
        "price_buy": price_buy,
        "price_sell": price_sell,
        "p_im": [p_im[t].value() for t in time],
        "p_ex": [p_ex[t].value() for t in time],
        "p_pv": [p_pv[t].value() for t in time],
        "r": [r[t].value() for t in time],
        "dr_window": dr_window,
        "soc": {ev.name: [soc[ev.name][t].value() for t in time] for ev in evs},
    }

    total_obj = pulp.value(prob.objective)
    print("\n===== RESULT SUMMARY =====")
    print(f"Total objective (₹): {total_obj:,.2f}")
    print(f"Energy cost (₹): {pulp.value(energy_cost):,.2f}")
    print(f"Peak demand cost (₹): {pulp.value(peak_cost):,.2f}")
    print(f"Degradation cost (₹): {pulp.value(degradation_cost):,.2f}")
    print(f"DR revenue (₹): {pulp.value(dr_revenue):,.2f}")

    return results


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------
def plot_results(results: Dict, evs: List[EV]):
    T = len(results["load"])
    t = list(range(T))

    # Power flows
    plt.figure(figsize=(10, 5))
    plt.plot(t, results["load"], label="Load (kW)")
    plt.plot(t, results["pv"], label="PV available (kW)")
    plt.plot(t, results["p_pv"], label="PV used (kW)")
    plt.plot(t, results["p_im"], label="Grid import (kW)")
    plt.plot(t, results["p_ex"], label="Grid export (kW)")
    plt.xlabel("Time index (hour)")
    plt.ylabel("Power (kW)")
    plt.title("Power Flow Profile (Load, PV, Grid)")
    plt.grid(True)
    plt.legend()

    # DR reduction
    plt.figure(figsize=(8, 4))
    plt.step(t, results["r"], where="mid", label="DR delta (kW)")
    for dw in results["dr_window"]:
        plt.axvspan(dw, dw+1, color="orange", alpha=0.1)
    plt.xlabel("Time index (hour)")
    plt.ylabel("DR delta (kW)")
    plt.title("Demand Response Effect (Positive = Reduction)")
    plt.grid(True)
    plt.legend()

    # SOCs
    plt.figure(figsize=(10, 5))
    for ev in evs:
        plt.plot(t, results["soc"][ev.name], label=f"SOC {ev.name}")
    plt.xlabel("Time index (hour)")
    plt.ylabel("SOC (p.u.)")
    plt.title("EV SOC Trajectories")
    plt.grid(True)
    plt.legend()

    plt.show()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    load, pv, price_buy, price_sell, baseline, dr_window = generate_example_profiles(T=24)
    evs = generate_example_evs()

    res = build_and_solve_vpp(
        load=load,
        pv=pv,
        price_buy=price_buy,
        price_sell=price_sell,
        baseline_net=baseline,
        evs=evs,
        dr_window=dr_window,
        delta_t=1.0,
        pcc_limit_kw=500.0,
        k_peak=145.0 / 30.0,
        lambda_dr=10.0,
    )

    plot_results(res, evs)
