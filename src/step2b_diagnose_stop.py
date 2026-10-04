"""
step2b_diagnose_stop.py - Find out why the baseline HPPC simulation stopped early.

Checks, at the moment the solver stopped:
  A) capacity      : charge removed (measured vs model) and electrode stoichiometries
  B) electrolyte   : minimum electrolyte concentration (depletion -> voltage collapse)
  C) LFP particles : positive / negative particle surface stoichiometry (saturation)

Also checks saturation: negative surface full (x -> 1) / positive surface empty (y -> 0),
which ends a run on the MAXIMUM voltage during charge pulses.

Run from the project root:
    python src\\step2b_diagnose_stop.py                      (baseline, Task 2)
    python src\\step2b_diagnose_stop.py --model calibrated   (Step 3c calibration)
Outputs -> results/step2/ or results/step3d/ : diagnose_stop.png, diagnose_stop.json
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import RESULTS_DIR, load_dataset
from model_setup import simulate



def get_var(sol, names):
    """Return (name, entries) for the first variable name that exists."""
    for n in names:
        try:
            return n, np.asarray(sol[n].entries)
        except Exception:  # noqa: BLE001
            continue
    return None, None


def spatial_min_max(arr):
    """Min and max over space for each time (works for 1D or 2D entries)."""
    if arr is None:
        return None, None
    if arr.ndim == 1:
        return arr, arr
    return arr.min(axis=0), arr.max(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "calibrated"], default="baseline")
    args = ap.parse_args()
    OUT = RESULTS_DIR / ("step2" if args.model == "baseline" else "step3d")
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"WHY DID THE {args.model.upper()} HPPC SIMULATION STOP EARLY?")
    print("=" * 66)
    data = load_dataset("HPPC")
    if args.model == "baseline":
        res = simulate(data, initial_soc=f"{float(data.V[0]):.4f} V")
    else:
        from calibration import (calibrated_parameter_values, load_ocv_calibration,
                                 set_state_from_full)
        calib = load_ocv_calibration()
        pv = set_state_from_full(calibrated_parameter_values(calib), calib, 0.0)
        res = simulate(data, parameter_values=pv, initial_soc=None)
    sol = res["sol"]
    pv = res["sim"].parameter_values
    t = res["t_sim"]
    V = res["V_sim"]
    t_stop = float(t[-1])

    report = {"t_stop_s": t_stop, "test_duration_s": float(data.t[-1])}

    # termination reason
    term = getattr(sol, "termination", "unknown")
    print(f"\nSolver termination reason : {term}")
    report["termination"] = str(term)

    # A) capacity
    q_meas = -np.interp(t_stop, data.t, data.ah_model())  # discharged, Ah (>0)
    _, q_model = get_var(sol, ["Discharge capacity [A.h]"])
    q_model_end = float(np.ravel(q_model)[-1]) if q_model is not None else float("nan")
    try:
        nominal = float(pv["Nominal cell capacity [A.h]"])
    except Exception:  # noqa: BLE001
        nominal = float("nan")
    print("\nA) CAPACITY")
    print(f"  Model nominal capacity       : {nominal:.3f} Ah")
    print(f"  Net discharged (measured)    : {q_meas:.3f} Ah at stop")
    print(f"  Net discharged (model)       : {q_model_end:.3f} Ah at stop")
    print(f"  Measured total in full test  : {-data.ah_model()[-1]:.3f} Ah")
    report.update({"nominal_capacity_Ah": nominal, "discharged_measured_Ah": float(q_meas),
                   "discharged_model_Ah": q_model_end})

    # B) electrolyte
    nm_ce, ce = get_var(sol, ["Electrolyte concentration [mol.m-3]"])
    ce_min, _ = spatial_min_max(ce)
    try:
        ce0 = float(pv["Initial concentration in electrolyte [mol.m-3]"])
    except Exception:  # noqa: BLE001
        ce0 = float("nan")
    print("\nB) ELECTROLYTE")
    if ce_min is not None:
        print(f"  Initial concentration        : {ce0:.0f} mol/m3")
        print(f"  Minimum at stop              : {ce_min[-1]:.1f} mol/m3")
        print(f"  Minimum over whole run       : {ce_min.min():.1f} mol/m3")
        report.update({"ce_initial": ce0, "ce_min_at_stop": float(ce_min[-1]),
                       "ce_min_overall": float(ce_min.min())})
    else:
        print("  (variable not available)")

    # C) particle surface stoichiometries
    print("\nC) PARTICLE SURFACE STOICHIOMETRY (0 = empty, 1 = full)")
    stoich = {}
    for elec in ["Negative", "Positive"]:
        nm, arr = get_var(sol, [f"{elec} particle surface stoichiometry",
                                f"{elec} electrode surface stoichiometry"])
        if arr is None:  # fall back to concentration / max concentration
            nm, arr = get_var(sol, [f"{elec} particle surface concentration [mol.m-3]"])
            if arr is not None:
                cmax = float(pv[f"Maximum concentration in {elec.lower()} electrode [mol.m-3]"])
                arr = arr / cmax
        lo, hi = spatial_min_max(arr)
        if lo is not None:
            stoich[elec] = (lo, hi)
            print(f"  {elec:<9} at stop: min {lo[-1]:.4f}, max {hi[-1]:.4f}   "
                  f"(whole run: {lo.min():.4f} - {hi.max():.4f})")
            report[f"{elec.lower()}_surface_stoich_at_stop"] = [float(lo[-1]), float(hi[-1])]
        else:
            print(f"  {elec}: (variable not available)")

    # bulk (average) stoichiometry at the stop from charge conservation
    bal = res.get("balance")
    if bal:
        q_net = float(np.interp(t_stop, data.t, data.ah_model()))   # Ah, charge > 0
        x_bulk = bal["sto_n_init"] + q_net / bal["Q_n_Ah"]
        y_bulk = bal["sto_p_init"] - q_net / bal["Q_p_Ah"]
        print(f"\nBULK (average) stoichiometry at stop from charge conservation: "
              f"negative {x_bulk:.4f}, positive {y_bulk:.4f}")
        print("  surface far from bulk -> solid-diffusion limitation; surface close to bulk -> capacity limit")
        report.update({"negative_bulk_sto_at_stop": float(x_bulk), "positive_bulk_sto_at_stop": float(y_bulk)})

    # automatic verdict
    print("\nVERDICT")
    verdict = []
    if ce_min is not None and ce_min[-1] < 0.05 * ce0:
        verdict.append("B) electrolyte depletion in the electrode during the high-rate pulse")
    if "Positive" in stoich and stoich["Positive"][1][-1] > 0.98:
        verdict.append("positive (LFP) particle surface saturated: fully lithiated (discharge limit)")
    if "Positive" in stoich and stoich["Positive"][0][-1] < 0.02:
        verdict.append("positive (LFP) particle surface emptied: fully delithiated (charge limit)")
    if "Negative" in stoich and stoich["Negative"][0][-1] < 0.02:
        verdict.append("negative (graphite) particle surface emptied (discharge limit)")
    if "Negative" in stoich and stoich["Negative"][1][-1] > 0.98:
        verdict.append("negative (graphite) particle surface saturated: full (charge limit)")
    if not verdict:
        verdict.append("no single limit reached - check the plots and the termination reason")
    for v in verdict:
        print(f"  -> {v}")
    report["verdict"] = verdict
    (OUT / "diagnose_stop.json").write_text(json.dumps(report, indent=2))

    # plot the last 1500 s before the stop
    w = t >= t_stop - 1500
    dm = (data.t >= t_stop - 1500) & (data.t <= t_stop)
    n_rows = 2 + (ce_min is not None) + len(stoich)
    fig, ax = plt.subplots(n_rows, 1, figsize=(11, 2.4 * n_rows), sharex=True)
    r = 0
    ax[r].plot(data.t[dm], data.V[dm], label="Measured")
    ax[r].plot(t[w], V[w], "--", label="DFN")
    ax[r].set_ylabel("Voltage [V]")
    ax[r].legend()
    ax[r].set_title(f"{args.model}: last 1500 s before the simulation stopped")
    r += 1
    ax[r].plot(data.t[dm], data.I_raw[dm], color="tab:gray")
    ax[r].set_ylabel("Current [A]")
    r += 1
    if ce_min is not None:
        ax[r].plot(t[w], ce_min[w], color="tab:green")
        ax[r].set_ylabel("min c_e\n[mol/m3]")
        r += 1
    for elec, (lo, hi) in stoich.items():
        ax[r].plot(t[w], lo[w], label="min over x")
        ax[r].plot(t[w], hi[w], label="max over x")
        ax[r].set_ylabel(f"{elec[:3]}. surface\nstoichiometry")
        ax[r].legend(fontsize=8)
        r += 1
    for a in ax:
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(OUT / "diagnose_stop.png", dpi=150)
    print(f"\nSaved: {OUT / 'diagnose_stop.png'}")
    print(f"Saved: {OUT / 'diagnose_stop.json'}")


if __name__ == "__main__":
    main()