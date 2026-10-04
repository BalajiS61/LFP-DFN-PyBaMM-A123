"""
step3b_ocv_analysis.py - Task 3: measured OCV of the cell vs the Prada2013 OCV.

Measured OCV sources
  1. HPPC OCV points over the FULL test (primary):
       - start point = rested, fully charged cell (SOC 1)
       - end-of-rest points; each rest (1800 s) is NOT fully relaxed, so an
         exponential relaxation V_inf + B exp(-t/tau) is fitted to the tail and
         extrapolated to equilibrium. The correction is reported per point.
     SOC by coulomb counting (within 6 mAh of the cycler counter).
  2. Final rests of the 1C-4C charge tests: OCV at full charge on the CHARGE
     branch -> hysteresis check. A segment held at 3.6 V with tapering current
     is the CV phase of the CC-CV charge, not a rest, and is rejected.
  3. 1C pseudo-OCV = average of 1C charge (CC) and 1C discharge (shape only).

Prada2013 OCV
  U_p(y) - U_n(x) along Prada's own stoichiometry window, compared vs SOC
  (shape) and vs Ah discharged (shape + capacity).

Run from the project root:
    python src\\step3b_ocv_analysis.py
Outputs -> results/step3b/
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import RESULTS_DIR, load_dataset
from model_setup import PARAM_SET, base_parameter_values
from ocv_tools import (end_rest_after_charge, eval_ocp, full_cell_ocv,
                       hppc_rest_points, pseudo_ocv_1c, stoich_window)

OUT = RESULTS_DIR / "step3b"
OUT.mkdir(parents=True, exist_ok=True)


def rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def main():
    print("=" * 66)
    print("TASK 3b - MEASURED OCV vs PRADA2013 OCV")
    print("=" * 66)
    hppc = load_dataset("HPPC")
    c1 = load_dataset("C1")
    d1 = load_dataset("D1")

    # ---------------- 1. HPPC OCV points with relaxation correction
    pts, q_total = hppc_rest_points(hppc)
    rp = pd.DataFrame(pts)
    print(f"\n1) HPPC OCV POINTS (full test, capacity reference {q_total:.3f} Ah)")
    print(f"  {'source':<11}{'SOC':>6}{'V_end':>9}{'slope':>8}{'tau[s]':>8}"
          f"{'V_inf':>9}{'corr':>7}{'fit':>5}{'V_ocv':>9}")
    print(f"  {'':<11}{'':>6}{'[V]':>9}{'[mV/h]':>8}{'':>8}{'[V]':>9}{'[mV]':>7}{'':>5}{'[V]':>9}")
    for _, r in rp.iterrows():
        tau = f"{r['tau_s']:>8.0f}" if np.isfinite(r["tau_s"]) else f"{'-':>8}"
        sl = f"{r['slope_mV_per_h']:>8.1f}" if np.isfinite(r["slope_mV_per_h"]) else f"{'-':>8}"
        print(f"  {r['source']:<11}{r['soc']:>6.3f}{r['V_end']:>9.4f}{sl}{tau}"
              f"{r['V_inf']:>9.4f}{r['relax_correction_mV']:>7.1f}"
              f"{('ok' if r['fit_ok'] else '-'):>5}{r['V_ocv']:>9.4f}")
    rests = rp[rp["source"] == "HPPC rest"]
    print(f"  relaxation fits valid: {int(rests['fit_ok'].sum())} of {len(rests)}; "
          f"correction range {rests['relax_correction_mV'].min():.1f} to "
          f"{rests['relax_correction_mV'].max():.1f} mV")

    # ---------------- 2. charge-branch OCV at full charge (hysteresis check)
    print("\n2) OCV AT FULL CHARGE AFTER CHARGE TESTS (charge branch)")
    chg = {}
    for k in ["C1", "C2", "C3", "C4"]:
        r = end_rest_after_charge(load_dataset(k))
        chg[k] = r
        line = (f"  {k}: last 600 s mean current {1000 * r['tail_current_mean_A']:.1f} mA, "
                f"voltage range {r['tail_V_range_mV']:.1f} mV, V_end {r['V_end']:.4f} V -> {r['reason']}")
        print(line)
    vals = [(r["V_inf"] if r.get("fit_ok") else r["V_end"]) for r in chg.values() if r["valid"]]
    hyst = None
    if vals:
        hyst = 1000 * (np.mean(vals) - rp.iloc[0]["V_ocv"])
        print(f"  charge-branch full-charge OCV {np.mean(vals):.4f} V vs HPPC start "
              f"{rp.iloc[0]['V_ocv']:.4f} V -> difference {hyst:.1f} mV")
    else:
        print("  -> no true rest after charging in any file: charge-branch OCV and hysteresis "
              "cannot be measured from this data")

    # ---------------- 3. 1C pseudo-OCV (shape reference)
    ps = pseudo_ocv_1c(c1, d1)
    v_ps = np.interp(rp["soc"], ps["soc"], ps["V_avg"], left=np.nan, right=np.nan)
    d_ps = (v_ps - rp["V_ocv"]) * 1000
    print(f"\n3) 1C PSEUDO-OCV: SOC {ps['soc'][0]:.3f}-{ps['soc'][-1]:.3f}; "
          f"minus HPPC OCV: mean {np.nanmean(d_ps):.1f} mV, max |diff| {np.nanmax(np.abs(d_ps)):.1f} mV; "
          f"half charge/discharge gap median {1000 * np.median(ps['half_gap']):.1f} mV")
    print("  -> used as a shape reference only (polarization + hysteresis too large for an OCV)")

    # ---------------- 4. Prada2013 OCV
    pv = base_parameter_values()
    w = stoich_window(pv)
    print(f"\n4) {PARAM_SET} STOICHIOMETRY WINDOW (own 2.0-3.6 V limits)")
    print(f"  negative x: {w['x0']:.4f} (SOC 0) -> {w['x100']:.4f} (SOC 1)")
    print(f"  positive y: {w['y0']:.4f} (SOC 0) -> {w['y100']:.4f} (SOC 1)")
    print(f"  Q_n {w['Q_n_Ah']:.3f} Ah, Q_p {w['Q_p_Ah']:.3f} Ah -> window capacity "
          f"{w['capacity_Ah']:.3f} Ah (measured {q_total:.3f} Ah, ratio {w['capacity_Ah'] / q_total:.3f})")

    soc_grid = np.linspace(0, 1, 201)
    ocv_prada = full_cell_ocv(pv, w, soc_grid)
    rp["Prada_same_SOC"] = np.interp(rp["soc"], soc_grid, ocv_prada)
    soc_ah = 1 - rp["Ah_discharged"] / w["capacity_Ah"]
    rp["Prada_same_Ah"] = np.where(soc_ah >= 0, np.interp(soc_ah.clip(0, 1), soc_grid, ocv_prada), np.nan)
    rp["err_same_SOC_mV"] = (rp["Prada_same_SOC"] - rp["V_ocv"]) * 1000
    rp["err_same_Ah_mV"] = (rp["Prada_same_Ah"] - rp["V_ocv"]) * 1000

    print(f"\n5) {PARAM_SET} OCV - MEASURED OCV (relaxation-corrected) [mV]")
    print(f"  {'SOC':>6}{'V_ocv':>9}{'same SOC':>10}{'same Ah':>10}")
    for _, r in rp.iterrows():
        sa = f"{r['err_same_Ah_mV']:>10.1f}" if np.isfinite(r["err_same_Ah_mV"]) else f"{'beyond':>10}"
        print(f"  {r['soc']:>6.3f}{r['V_ocv']:>9.4f}{r['err_same_SOC_mV']:>10.1f}{sa}")
    mid = (rp["soc"] >= 0.3) & (rp["soc"] <= 0.95)
    print(f"  RMSE same SOC: all points {rms(rp['err_same_SOC_mV']):.1f} mV, "
          f"SOC 30-95 % {rms(rp.loc[mid, 'err_same_SOC_mV']):.1f} mV")

    # ---------------- save
    rp.to_csv(OUT / "measured_ocv_points.csv", index=False)
    pd.DataFrame({k: ps[k] for k in ["soc", "V_charge", "V_discharge", "V_avg", "half_gap"]}
                 ).to_csv(OUT / "pseudo_ocv_1C.csv", index=False)
    pd.DataFrame({"soc": soc_grid, "OCV_prada": ocv_prada}).to_csv(OUT / "prada_ocv.csv", index=False)
    summary = {"capacity_measured_Ah": q_total, "prada_window": w,
               "n_ocv_points": int(len(rp)),
               "n_relaxation_fits_ok": int(rests["fit_ok"].sum()),
               "relax_correction_mV_range": [float(rests["relax_correction_mV"].min()),
                                             float(rests["relax_correction_mV"].max())],
               "charge_branch_full_charge": chg, "hysteresis_full_charge_mV": hyst,
               "rmse_same_soc_all_mV": rms(rp["err_same_SOC_mV"]),
               "rmse_same_soc_30_95_mV": rms(rp.loc[mid, "err_same_SOC_mV"])}
    (OUT / "ocv_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---------------- plots
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    a.plot(ps["soc"] * 100, ps["V_charge"], color="tab:green", lw=1, label="1C charge (CC)")
    a.plot(ps["soc"] * 100, ps["V_discharge"], color="tab:red", lw=1, label="1C discharge")
    a.plot(ps["soc"] * 100, ps["V_avg"], "k--", lw=1, label="1C average")
    a.plot(rests["soc"] * 100, rests["V_end"], "x", color="tab:gray", label="HPPC end of rest (raw)")
    a.plot(rp["soc"] * 100, rp["V_ocv"], "o", color="tab:blue", label="HPPC OCV (relaxation-corrected)")
    if vals:
        a.plot([100], [np.mean(vals)], "^", color="tab:green", ms=9, label="full charge, charge branch")
    a.set_xlabel("SOC [%]")
    a.set_ylabel("Voltage [V]")
    a.set_title("Measured OCV sources")
    a.set_ylim(2.8, 3.65)
    a.legend(fontsize=8)

    a = ax[0, 1]
    ah_grid = (1 - soc_grid) * w["capacity_Ah"]
    a.plot(rp["Ah_discharged"], rp["V_ocv"], "o-", color="tab:blue", label="Measured OCV")
    a.plot(ah_grid, ocv_prada, "--", color="tab:orange", label=f"{PARAM_SET} OCV")
    a.axvline(w["capacity_Ah"], color="tab:orange", ls=":", lw=1,
              label=f"{PARAM_SET} capacity {w['capacity_Ah']:.3f} Ah")
    a.axvline(q_total, color="tab:blue", ls=":", lw=1, label=f"measured {q_total:.3f} Ah")
    a.set_xlabel("Charge discharged from full [Ah]")
    a.set_ylabel("OCV [V]")
    a.set_title("OCV vs discharged charge: shape + capacity")
    a.set_ylim(2.8, 3.65)
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(rp["soc"] * 100, rp["err_same_SOC_mV"], "o-", label="at same SOC")
    a.plot(rp["soc"] * 100, rp["err_same_Ah_mV"], "s--", label="at same Ah discharged")
    a.bar(rests["soc"] * 100, rests["relax_correction_mV"], width=1.2, color="tab:gray",
          alpha=0.5, label="relaxation correction applied")
    a.axhline(0, color="k", lw=0.5)
    a.invert_xaxis()
    a.set_xlabel("SOC [%] (measured)")
    a.set_ylabel(f"{PARAM_SET} - measured [mV]")
    a.set_ylim(max(-300, np.nanmin(rp["err_same_SOC_mV"]) - 20), 60)
    a.set_title("OCV error (points below -300 mV off-scale)")
    a.legend(fontsize=8)

    a = ax[1, 1]
    sto = np.linspace(0.001, 0.999, 300)
    a.plot(sto, eval_ocp(pv, "positive", sto), color="tab:red", label="positive (LFP) U_p")
    a.plot(sto, eval_ocp(pv, "negative", sto), color="tab:gray", label="negative (graphite) U_n")
    a.axvspan(min(w["y0"], w["y100"]), max(w["y0"], w["y100"]), color="tab:red", alpha=0.1,
              label="positive window used")
    a.axvspan(min(w["x0"], w["x100"]), max(w["x0"], w["x100"]), color="tab:gray", alpha=0.1,
              label="negative window used")
    a.set_xlabel("Stoichiometry [-]")
    a.set_ylabel("OCP vs Li/Li+ [V]")
    a.set_ylim(0, 4.0)
    a.set_title(f"{PARAM_SET} electrode OCPs")
    a.legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("Task 3b - measured OCV vs Prada2013 OCV", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "step3b_ocv_analysis.png", dpi=150)
    print(f"\nSaved: {OUT / 'step3b_ocv_analysis.png'}")
    print("Saved: measured_ocv_points.csv, pseudo_ocv_1C.csv, prada_ocv.csv, ocv_summary.json")


if __name__ == "__main__":
    main()