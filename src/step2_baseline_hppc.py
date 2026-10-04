"""
step2_baseline_hppc.py - Task 2: simulate HPPC with the un-tuned Prada2013 DFN.

- applies the measured HPPC current (backward-ZOH step profile)
- compares simulated vs measured terminal voltage
- computes RMSE, MAE, max error, bias, and RMSE split into rest / load
- records the electrode balance and why the simulation ended

Initial state:
    default ("auto"): from the measured initial rest voltage  -> results/step2/
    --soc0 1.0      : forced fully charged start             -> results/step2_soc0_1.0/
(each mode writes to its own folder so runs never overwrite each other)

Run from the project root:
    python src\\step2_baseline_hppc.py
    python src\\step2_baseline_hppc.py --soc0 1.0
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import RESULTS_DIR, load_dataset
from model_setup import (PARAM_SET, compare_on_data_grid, simulate,
                         voltage_metrics)


def run(data, soc0_arg):
    if soc0_arg == "auto":
        v0 = float(data.V[0])
        try:
            print(f"Initial state from measured rest voltage: {v0:.4f} V")
            return simulate(data, initial_soc=f"{v0:.4f} V")
        except Exception as e:  # noqa: BLE001
            print(f"  voltage-based initial state not possible ({type(e).__name__}: {e})")
            print("  -> falling back to initial SOC = 1.0 (cell fully charged before HPPC)")
            return simulate(data, initial_soc=1.0)
    print(f"Initial SOC forced to {float(soc0_arg)}")
    return simulate(data, initial_soc=float(soc0_arg))


def pulse_zoom_window(data, frac=0.4, width_before=200, width_after=700):
    """Window around the first large discharge pulse after `frac` of the test."""
    cand = np.where((data.I_raw < -30) & (data.t > frac * data.t[-1]))[0]
    if cand.size == 0:
        cand = np.where(np.abs(data.I_raw) > 1)[0]
    ts = data.t[cand[0]]
    return ts - width_before, ts + width_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soc0", default="auto", help="'auto' or a number, e.g. 1.0")
    args = ap.parse_args()

    out = RESULTS_DIR / ("step2" if args.soc0 == "auto" else f"step2_soc0_{float(args.soc0)}")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"TASK 2 - BASELINE DFN ({PARAM_SET}) ON HPPC")
    print("=" * 66)
    data = load_dataset("HPPC")
    print(f"HPPC: {data.t.size} samples, {data.t[-1]:.0f} s, "
          f"{-data.ah_model()[-1]:.3f} Ah net discharged in the test")

    res = run(data, args.soc0)
    mask, V_sim = compare_on_data_grid(data, res)
    t, V_meas, I = data.t[mask], data.V[mask], data.I_raw[mask]
    err = V_sim - V_meas

    m = voltage_metrics(err, I)
    i_max = int(np.argmax(np.abs(err)))
    m.update({
        "parameter_set": PARAM_SET,
        "initial_soc_used": str(res["initial_soc_used"]),
        "electrode_balance": res["balance"],
        "simulated_until_s": float(res["t_sim"][-1]),
        "test_duration_s": float(data.t[-1]),
        "stopped_early": res["stopped_early"],
        "termination": res["termination"],
        "discharged_at_end_Ah": float(-np.interp(res["t_sim"][-1], data.t, data.ah_model())),
        "fraction_of_test_compared": float(t[-1] / data.t[-1]),
        "max_error_time_s": float(t[i_max]),
        "max_error_current_A": float(I[i_max]),
        "measured_V_at_max_error": float(V_meas[i_max]),
        "simulated_V_at_max_error": float(V_sim[i_max]),
        "runtime_s": float(res["runtime_s"]),
    })
    keep = t < t[-1] - 60 if res["stopped_early"] else np.ones_like(t, bool)
    if res["stopped_early"]:
        m["excluding_final_60s"] = voltage_metrics(err[keep], I[keep])

    print("\nBASELINE HPPC METRICS (error = simulated - measured)")
    print("-" * 50)
    for k in ["RMSE_V", "MAE_V", "MaxAbs_V", "MeanError_V", "StdError_V",
              "RMSE_rest_V", "RMSE_load_V"]:
        if k in m:
            print(f"  {k:<14} = {m[k] * 1000:9.1f} mV")
    print(f"  Max error at t = {m['max_error_time_s']:.1f} s, I = {m['max_error_current_A']:.2f} A "
          f"(meas {m['measured_V_at_max_error']:.3f} V, sim {m['simulated_V_at_max_error']:.3f} V)")
    print(f"  Compared {100 * m['fraction_of_test_compared']:.1f} % of the test, "
          f"{m['discharged_at_end_Ah']:.3f} Ah discharged at the end of the simulation")
    if "excluding_final_60s" in m:
        ex = m["excluding_final_60s"]
        print(f"  Without the final 60 s (end-of-run collapse): RMSE {ex['RMSE_V'] * 1000:.1f} mV, "
              f"MAE {ex['MAE_V'] * 1000:.1f} mV, max {ex['MaxAbs_V'] * 1000:.1f} mV")

    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    pd.DataFrame({"t_s": t, "I_A_charge_pos": I, "V_meas": V_meas,
                  "V_sim": V_sim, "error_V": err}).to_csv(out / "hppc_baseline.csv", index=False)

    th = t / 3600
    t_stop_h = th[-1]
    note = (f"simulation stopped at {t_stop_h:.1f} h ({m['discharged_at_end_Ah']:.2f} Ah):\n"
            f"model voltage collapses to {V_sim[-1]:.2f} V (off-scale)")

    # Plot 1: measured vs simulated (y-range set by the measured data)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(data.t / 3600, data.V, lw=0.8, label="Measured (full test)")
    ax[0].plot(th, V_sim, lw=0.8, ls="--", label=f"DFN ({PARAM_SET}, un-tuned)")
    ylo, yhi = data.V.min() - 0.15, max(data.V.max(), np.percentile(V_sim[keep], 99.9)) + 0.1
    ax[0].set_ylim(ylo, yhi)
    if res["stopped_early"]:
        ax[0].axvline(t_stop_h, color="tab:red", ls=":", lw=1.2, label="simulation stopped")
        ax[0].text(t_stop_h + 0.2, ylo + 0.05, note, fontsize=8, color="tab:red", va="bottom")
    ax[0].set_ylabel("Voltage [V]")
    rm = m["excluding_final_60s"]["RMSE_V"] if "excluding_final_60s" in m else m["RMSE_V"]
    ax[0].set_title(f"Task 2 - HPPC: measured vs simulated voltage "
                    f"(RMSE {rm * 1000:.1f} mV on simulated part)")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)
    ax[1].plot(data.t / 3600, data.I_raw, lw=0.6, color="tab:gray")
    ax[1].set_ylabel("Current [A]\n(charge +)")
    ax[1].set_xlabel("Time [h]")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "step2_measured_vs_simulated.png", dpi=150)

    # Plot 2: voltage error (y-range set by the 99.9th percentile, collapse annotated)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7),
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(th, err * 1000, lw=0.6, color="tab:red")
    ax[0].axhline(0, color="k", lw=0.5)
    elim = np.percentile(np.abs(err[keep]), 99.9) * 1000 * 1.15
    ax[0].set_ylim(-elim, elim)
    if res["stopped_early"]:
        ax[0].axvline(t_stop_h, color="tab:red", ls=":", lw=1.2)
        ax[0].text(t_stop_h, -elim * 0.95, note, fontsize=8, color="tab:red",
                   ha="right", va="bottom")
    ax[0].set_xlabel("Time [h]")
    ax[0].set_ylabel("Error (sim - meas) [mV]")
    ax[0].set_title(f"Task 2 - HPPC voltage estimation error "
                    f"(MAE {m.get('excluding_final_60s', m)['MAE_V'] * 1000:.1f} mV)")
    ax[0].grid(alpha=0.3)
    ax[1].hist(np.clip(err[keep] * 1000, -elim, elim), bins=200, color="tab:red", alpha=0.7)
    ax[1].set_xlabel("Error [mV] (end-of-run collapse excluded)")
    ax[1].set_ylabel("Count")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "step2_voltage_error.png", dpi=150)

    # Plot 3: zoom on one pulse set
    a, b = pulse_zoom_window(data)
    z = (t >= a) & (t <= b)
    if z.any():
        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        ax[0].plot(t[z], V_meas[z], label="Measured")
        ax[0].plot(t[z], V_sim[z], "--", label="DFN")
        ax[0].set_ylabel("Voltage [V]")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[0].set_title("Task 2 - zoom on one HPPC pulse set")
        ax[1].plot(t[z], err[z] * 1000, color="tab:red")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set_ylabel("Error [mV]")
        ax[1].grid(alpha=0.3)
        ax[2].plot(t[z], I[z], color="tab:gray")
        ax[2].set_ylabel("Current [A]")
        ax[2].set_xlabel("Time [s]")
        ax[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "step2_pulse_zoom.png", dpi=150)

    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()