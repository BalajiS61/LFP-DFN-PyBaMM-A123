"""
step3d_hppc_calibrated.py - Task 3: effect of the manual (thermodynamic) calibration.

Runs HPPC with Prada2013 + Step 3c calibration (graphite OCP, electrode capacities,
stoichiometry window; kinetics and transport still un-tuned) and compares it with
the Task 2 baseline:
  - on the SAME time window (the part the baseline could simulate) -> fair comparison
  - over the full test (if the calibrated model completes it)
Initial state: fully charged, taken directly from the calibrated lithium line.

Run from the project root:
    python src\\step3d_hppc_calibrated.py
Outputs -> results/step3d/
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration import (calibrated_parameter_values, load_ocv_calibration,
                         set_state_from_full)
from data_loader import RESULTS_DIR, load_dataset
from model_setup import compare_on_data_grid, simulate, voltage_metrics

BASE_CSV = RESULTS_DIR / "step2" / "hppc_baseline.csv"
OUT = RESULTS_DIR / "step3d"
OUT.mkdir(parents=True, exist_ok=True)


def fmt(m):
    keys = ["RMSE_V", "MAE_V", "MaxAbs_V", "MeanError_V", "RMSE_rest_V", "RMSE_load_V"]
    return "".join(f"{1000 * m[k]:>9.1f}" if k in m else f"{'-':>9}" for k in keys)


def main():
    print("=" * 66)
    print("TASK 3d - HPPC WITH CALIBRATED THERMODYNAMICS (Step 3c)")
    print("=" * 66)
    calib = load_ocv_calibration()
    print(f"Calibration: graphite OCP {calib['negative_ocp_source']}, "
          f"x100 {calib['x100']:.4f}, y100 {calib['y100']:.4f}, "
          f"Q_n {calib['Q_n_Ah']:.3f} Ah, Q_p {calib['Q_p_Ah']:.3f} Ah")

    data = load_dataset("HPPC")
    pv = calibrated_parameter_values(calib)
    pv = set_state_from_full(pv, calib, 0.0)
    res = simulate(data, parameter_values=pv, initial_soc=None)

    mask, V_sim = compare_on_data_grid(data, res)
    t, V_meas, I = data.t[mask], data.V[mask], data.I_raw[mask]
    err = V_sim - V_meas
    stop_cut = res["stopped_early"]
    keep = t < t[-1] - 60 if stop_cut else np.ones_like(t, bool)

    m_full = voltage_metrics(err[keep], I[keep])
    pd.DataFrame({"t_s": t, "I_A_charge_pos": I, "V_meas": V_meas, "V_sim": V_sim,
                  "error_V": err}).to_csv(OUT / "hppc_calibrated.csv", index=False)

    # ---------------- fair comparison with the baseline on the COMMON window
    # (both runs' final 60 s excluded if they stopped early)
    base = pd.read_csv(BASE_CSV)
    t_b_end = base["t_s"].iloc[-1] - 60
    t_c_end = t[-1] - 60 if stop_cut else t[-1]
    t_b_end = min(t_b_end, t_c_end)
    w_cal = t <= t_b_end
    w_base = base["t_s"] <= t_b_end
    m_cal_win = voltage_metrics(err[w_cal], I[w_cal])
    m_base_win = voltage_metrics(base.loc[w_base, "error_V"].to_numpy(),
                                 base.loc[w_base, "I_A_charge_pos"].to_numpy())

    frac = t[-1] / data.t[-1]
    print(f"\nCoverage: {100 * frac:.1f} % of the HPPC test "
          f"({'complete' if not res['stopped_early'] else 'stopped early: ' + res['termination']})")
    print(f"\n{'':<34}{'RMSE':>9}{'MAE':>9}{'max':>9}{'mean':>9}{'rest':>9}{'load':>9}  [mV]")
    print(f"{'baseline, common window 0-' + f'{t_b_end / 3600:.1f} h':<34}{fmt(m_base_win)}")
    print(f"{'calibrated, common window':<34}{fmt(m_cal_win)}")
    print(f"{'calibrated, full simulated part':<34}{fmt(m_full)}")
    imp = 100 * (1 - m_cal_win["RMSE_V"] / m_base_win["RMSE_V"])
    print(f"\nRMSE change on the common window (0-{t_b_end / 3600:.1f} h): {1000 * m_base_win['RMSE_V']:.1f} -> "
          f"{1000 * m_cal_win['RMSE_V']:.1f} mV ({imp:+.0f} % {'improvement' if imp >= 0 else 'worse'})")

    summary = {"calibration": {k: calib[k] for k in ["negative_ocp_source", "x100", "y100",
                                                     "Q_n_Ah", "Q_p_Ah"]},
               "coverage_fraction": float(frac), "stopped_early": res["stopped_early"],
               "termination": res["termination"], "window_end_s": float(t_b_end),
               "baseline_common_window": m_base_win, "calibrated_common_window": m_cal_win,
               "calibrated_full": m_full, "rmse_improvement_common_window_pct": float(imp),
               "runtime_s": res["runtime_s"]}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))

    # ---------------- plots
    th = t / 3600
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 1]})
    ax[0].plot(data.t / 3600, data.V, lw=0.8, color="k", label="Measured")
    ax[0].plot(base["t_s"] / 3600, base["V_sim"], lw=0.7, ls=":", color="tab:gray",
               label="Baseline (Prada2013, un-tuned)")
    ax[0].plot(th, V_sim, lw=0.8, ls="--", color="tab:orange",
               label="Calibrated thermodynamics (3c)")
    ax[0].set_ylim(data.V.min() - 0.15, data.V.max() + 0.15)
    ax[0].set_ylabel("Voltage [V]")
    ax[0].set_title(f"Task 3d - HPPC after OCV / capacity calibration "
                    f"(RMSE {1000 * m_full['RMSE_V']:.1f} mV over {100 * frac:.0f} % of the test)")
    ax[0].legend(loc="lower left", fontsize=8)
    lim = max(np.percentile(np.abs(err[keep]), 99.9), np.percentile(np.abs(base["error_V"]), 99.0)) * 1000 * 1.1
    ax[1].plot(base["t_s"] / 3600, base["error_V"] * 1000, lw=0.6, color="tab:gray", label="baseline")
    ax[1].plot(th, err * 1000, lw=0.6, color="tab:red", label="calibrated")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_ylabel("Error (sim - meas) [mV]")
    ax[1].legend(fontsize=8)
    ax[2].plot(data.t / 3600, data.I_raw, lw=0.6, color="tab:gray")
    ax[2].set_ylabel("Current [A]")
    ax[2].set_xlabel("Time [h]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "step3d_hppc_calibrated.png", dpi=150)
    print(f"\nSaved: {OUT / 'step3d_hppc_calibrated.png'}, hppc_calibrated.csv, metrics.json")
    print("Next: python src\\step3a_residual_analysis.py --input results\\step3d\\hppc_calibrated.csv "
          "--out step3d_residuals --label \"calibrated (3c)\"")


if __name__ == "__main__":
    main()