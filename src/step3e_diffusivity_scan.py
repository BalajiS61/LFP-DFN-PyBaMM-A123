"""
step3e_diffusivity_scan.py - Task 3: sensitivity of the HPPC result to the graphite
solid diffusivity (manual tuning of the transport parameter that ends both runs).

Evidence (step2b diagnosis):
  baseline   : graphite surface EMPTY (0.005 vs bulk 0.254) on discharge pulses, low SOC
  calibrated : graphite surface FULL  (0.9999 vs bulk 0.928) on charge pulses, high SOC
  -> graphite solid diffusion too slow in Prada2013.

For each factor k, D_n -> k * D_n(Prada2013) on the Step 3c calibrated model, and
the full HPPC test is simulated. Reported per factor:
  coverage, termination, RMSE over a COMMON window (fair), rest / load RMSE,
  and the resulting D_n value (for a literature-range check).

Run from the project root:
    python src\\step3e_diffusivity_scan.py
    python src\\step3e_diffusivity_scan.py --factors 1 3 10 30 100 300
Outputs -> results/step3e/
"""
import argparse
import functools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

from calibration import (calibrated_parameter_values, load_ocv_calibration,
                         set_state_from_full)
from data_loader import RESULTS_DIR, load_dataset
from model_setup import compare_on_data_grid, simulate, voltage_metrics

OUT = RESULTS_DIR / "step3e"
OUT.mkdir(parents=True, exist_ok=True)
DN = "Negative particle diffusivity [m2.s-1]"


def scale_parameter(pv, name, k):
    """Multiply a parameter (constant or function) by k, keeping a function's signature."""
    v = pv[name]
    if callable(v):
        @functools.wraps(v)
        def scaled(*args, _f=v, _k=k):
            return _k * _f(*args)
        pv.update({name: scaled})
    else:
        pv.update({name: k * float(v)})
    return pv


def parameter_value(pv, name, sto=0.5, T=298.15):
    """Numeric value of a constant or function parameter (function at sto, T)."""
    v = pv[name]
    if not callable(v):
        return float(v)
    try:
        out = v(pybamm.Scalar(sto), pybamm.Scalar(T))
    except TypeError:
        out = v(pybamm.Scalar(sto))
    return float(out.evaluate()) if hasattr(out, "evaluate") else float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=float, nargs="+", default=[1, 3, 10, 30, 100, 300])
    args = ap.parse_args()

    print("=" * 66)
    print("TASK 3e - GRAPHITE DIFFUSIVITY SENSITIVITY (HPPC, calibrated thermodynamics)")
    print("=" * 66)
    calib = load_ocv_calibration()
    data = load_dataset("HPPC")
    pv0 = calibrated_parameter_values(calib)
    dn0 = parameter_value(pv0, DN)
    print(f"Prada2013 graphite diffusivity: {dn0:.3e} m2/s "
          f"({'function of stoichiometry, value at x = 0.5' if callable(pv0[DN]) else 'constant'})")

    runs = []
    for k in args.factors:
        print(f"\n--- D_n x {k:g} ({k * dn0:.2e} m2/s) ---")
        pv = scale_parameter(calibrated_parameter_values(calib), DN, k)
        pv = set_state_from_full(pv, calib, 0.0)
        try:
            res = simulate(data, parameter_values=pv, initial_soc=None)
        except Exception as e:  # noqa: BLE001
            print(f"  solver failed: {type(e).__name__}: {e}")
            runs.append({"factor": k, "Dn": k * dn0, "failed": True})
            continue
        mask, V_sim = compare_on_data_grid(data, res)
        t, Vm, I = data.t[mask], data.V[mask], data.I_raw[mask]
        err = V_sim - Vm
        valid_end = t[-1] - 60 if res["stopped_early"] else t[-1]
        pd.DataFrame({"t_s": t, "I_A_charge_pos": I, "V_meas": Vm, "V_sim": V_sim,
                      "error_V": err}).to_csv(OUT / f"hppc_Dn_x{k:g}.csv", index=False)
        runs.append({"factor": k, "Dn": k * dn0, "failed": False,
                     "coverage": float(t[-1] / data.t[-1]),
                     "stopped_early": res["stopped_early"], "termination": res["termination"],
                     "valid_end_s": float(valid_end), "runtime_s": res["runtime_s"],
                     "_t": t, "_err": err, "_I": I, "_V": V_sim})

    ok = [r for r in runs if not r["failed"]]
    if not ok:
        raise RuntimeError("all runs failed")
    t_common = min(r["valid_end_s"] for r in ok)
    print(f"\nCommon window for a fair comparison: 0-{t_common / 3600:.2f} h")
    print(f"\n{'Dn factor':>9}{'Dn [m2/s]':>11}{'cover':>8}{'RMSE*':>8}{'rest*':>8}{'load*':>8}"
          f"{'RMSE all':>10}{'max all':>9}  end")
    for r in runs:
        if r["failed"]:
            print(f"{r['factor']:>9g}{r['Dn']:>11.2e}  solver failed")
            continue
        w = r["_t"] <= t_common
        m_c = voltage_metrics(r["_err"][w], r["_I"][w])
        keep = r["_t"] <= r["valid_end_s"]
        m_a = voltage_metrics(r["_err"][keep], r["_I"][keep])
        r.update({"common": m_c, "all": m_a})
        end = "complete" if not r["stopped_early"] else r["termination"].replace("event: ", "")
        print(f"{r['factor']:>9g}{r['Dn']:>11.2e}{100 * r['coverage']:>7.1f}%"
              f"{1000 * m_c['RMSE_V']:>8.1f}{1000 * m_c.get('RMSE_rest_V', np.nan):>8.1f}"
              f"{1000 * m_c.get('RMSE_load_V', np.nan):>8.1f}{1000 * m_a['RMSE_V']:>10.1f}"
              f"{1000 * m_a['MaxAbs_V']:>9.1f}  {end}")
    print("  * = over the common window; 'all' = over each run's own simulated part [mV]")

    complete = [r for r in ok if not r["stopped_early"]]
    if complete:
        print(f"\nSmallest factor completing the full test: x{min(r['factor'] for r in complete):g}")
    else:
        print("\nNo factor completes the full test - another limit is active (see terminations)")

    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
    (OUT / "diffusivity_scan.json").write_text(json.dumps(
        {"Dn_prada_m2s": dn0, "common_window_s": t_common, "runs": clean}, indent=2, default=str))

    # ---------------- plots
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    f = [r["factor"] for r in ok]
    ax[0, 0].semilogx(f, [100 * r["coverage"] for r in ok], "o-")
    ax[0, 0].set_xlabel("D_n factor (x Prada2013)")
    ax[0, 0].set_ylabel("HPPC coverage [%]")
    ax[0, 0].set_title("Test coverage vs graphite diffusivity")
    ax[0, 1].semilogx(f, [1000 * r["common"]["RMSE_V"] for r in ok], "o-", label="all")
    ax[0, 1].semilogx(f, [1000 * r["common"].get("RMSE_rest_V", np.nan) for r in ok], "s--", label="rest")
    ax[0, 1].semilogx(f, [1000 * r["common"].get("RMSE_load_V", np.nan) for r in ok], "^--", label="load")
    ax[0, 1].set_xlabel("D_n factor (x Prada2013)")
    ax[0, 1].set_ylabel(f"RMSE over 0-{t_common / 3600:.1f} h [mV]")
    ax[0, 1].set_title("Error on the common window")
    ax[0, 1].legend()
    ax[1, 0].plot(data.t / 3600, data.V, "k", lw=0.8, label="Measured")
    for r in ok:
        ax[1, 0].plot(r["_t"] / 3600, r["_V"], lw=0.6, label=f"x{r['factor']:g}")
    ax[1, 0].set_ylim(data.V.min() - 0.15, data.V.max() + 0.2)
    ax[1, 0].set_xlabel("Time [h]")
    ax[1, 0].set_ylabel("Voltage [V]")
    ax[1, 0].set_title("HPPC voltage")
    ax[1, 0].legend(fontsize=7, ncol=2)
    for r in ok:
        ax[1, 1].plot(r["_t"] / 3600, r["_err"] * 1000, lw=0.5, label=f"x{r['factor']:g}")
    ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[1, 1].set_ylim(-250, 250)
    ax[1, 1].set_xlabel("Time [h]")
    ax[1, 1].set_ylabel("Error [mV]")
    ax[1, 1].set_title("Voltage error")
    ax[1, 1].legend(fontsize=7, ncol=2)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("Task 3e - graphite diffusivity sensitivity (calibrated thermodynamics)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "step3e_diffusivity_scan.png", dpi=150)
    print(f"\nSaved: {OUT / 'step3e_diffusivity_scan.png'}, diffusivity_scan.json, hppc_Dn_x*.csv")


if __name__ == "__main__":
    main()