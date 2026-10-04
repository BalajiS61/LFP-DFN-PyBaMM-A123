"""
step4a_training_check.py - Task 4 preparation: check the training set-up before optimizing.

Training data (assignment): Charge 1C/2C/3C/4C (CC-CV to 3.6 V) + Discharge 1C.
Model: Task 3 manually tuned model (Step 3c thermodynamics + D_n from Step 3e).

1. Initial state of each test
     charge tests end at the CV cut-off (full) -> start = full - charge put in
     discharge test starts full (rested at 3.428 V) -> start = full
   Cross-check: state implied by the initial voltage via the calibrated OCV
   (only indicative: the cell may not be relaxed at the start).
2. Training error of the Task 3 model per test (the number the optimizer must beat)
3. Runtime per evaluation (plans the optimization)
4. Mesh check: coarse mesh vs PyBaMM default mesh (voltage difference, speed-up)

Run from the project root:
    python src\\step4a_training_check.py
Outputs -> results/step4a/
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration import (TASK3_DN, lithium_line, load_ocv_calibration, set_state_from_full,
                         task3_parameter_values)
from data_loader import DATASETS, RESULTS_DIR, TRAINING_KEYS, load_dataset
from model_setup import COARSE_MESH, compare_on_data_grid, simulate, voltage_metrics
from ocv_tools import eval_ocp

OUT = RESULTS_DIR / "step4a"
OUT.mkdir(parents=True, exist_ok=True)


def start_from_full(data, key):
    """Ah below full at the start of a training test (from charge counting)."""
    if key.startswith("C"):
        return float(data.ah_model()[-1])      # charge put in until the CV cut-off (full)
    return 0.0                                 # discharge test starts full


def ocv_inverse(calib, pv, v0):
    """Ah below full whose calibrated OCV equals v0 (indicative)."""
    ah = np.linspace(0, 2.62, 1500)
    x, y = lithium_line(calib, ah)
    ok = (x > 0) & (x < 1) & (y > 0) & (y < 1)
    ah = ah[ok]
    ocv = eval_ocp(pv, "positive", y[ok]) - eval_ocp(pv, "negative", x[ok])
    idx = np.argmin(np.abs(ocv - v0))
    return float(ah[idx]), float(ocv[idx])


def run(data, pv_base, calib, ah0, mesh=None):
    pv = set_state_from_full(pv_base.copy(), calib, ah0)
    t0 = time.perf_counter()
    res = simulate(data, parameter_values=pv, initial_soc=None, var_pts=mesh, verbose=False)
    wall = time.perf_counter() - t0
    mask, V_sim = compare_on_data_grid(data, res)
    return res, mask, V_sim, wall


def main():
    print("=" * 66)
    print("TASK 4a - TRAINING DATA CHECK (Task 3 model)")
    print("=" * 66)
    calib = load_ocv_calibration()
    pv_base = task3_parameter_values(calib)
    print(f"Model: Step 3c thermodynamics + D_n {TASK3_DN:.1e} m2/s; training sets: {TRAINING_KEYS}")

    # ---------------- 1. initial states
    print("\n1) INITIAL STATES")
    print(f"  {'set':<4}{'V0 [V]':>8}{'I0 [A]':>8}{'Ah below full':>15}{'x0':>8}{'y0':>8}"
          f"{'from V0 (OCV)':>15}")
    data, states = {}, {}
    for k in TRAINING_KEYS:
        d = load_dataset(k)
        data[k] = d
        ah0 = start_from_full(d, k)
        x0, y0 = lithium_line(calib, ah0)
        ah_v, _ = ocv_inverse(calib, pv_base, float(d.V[0]))
        states[k] = {"ah_below_full": ah0, "x0": float(x0), "y0": float(y0),
                     "V0": float(d.V[0]), "ah_from_V0_indicative": ah_v}
        print(f"  {k:<4}{d.V[0]:>8.4f}{d.I_raw[0]:>8.3f}{ah0:>15.3f}{x0:>8.4f}{y0:>8.4f}{ah_v:>15.3f}")
    print("  (charge tests: start = full - charge put in until the CV cut-off; the voltage-based value is"
          " only indicative because the cell is not relaxed after the preceding discharge)")

    # ---------------- 2-4. simulate with default and coarse mesh
    print("\n2) TRAINING ERROR OF THE TASK 3 MODEL + 3) RUNTIME + 4) MESH CHECK")
    print(f"  {'set':<4}{'cover':>7}{'RMSE':>8}{'MAE':>8}{'max':>8}{'mean':>8}"
          f"{'t_def[s]':>9}{'t_coarse':>9}{'dV rms':>8}{'dV max':>8}  [mV]")
    results, curves = {}, {}
    for k in TRAINING_KEYS:
        d = data[k]
        ah0 = states[k]["ah_below_full"]
        try:
            res_f, mask_f, Vf, wf = run(d, pv_base, calib, ah0)
            res_c, mask_c, Vc, wc = run(d, pv_base, calib, ah0, COARSE_MESH)
        except Exception as e:  # noqa: BLE001
            print(f"  {k:<4} simulation failed: {type(e).__name__}: {e}")
            results[k] = {"failed": True}
            continue
        err = Vf - d.V[mask_f]
        keep = np.ones(err.size, bool)
        if res_f["stopped_early"]:
            keep = d.t[mask_f] < d.t[mask_f][-1] - 60
        m = voltage_metrics(err[keep], d.I_raw[mask_f][keep])
        n = min(mask_f.sum(), mask_c.sum())
        dV = (Vc[:n] - Vf[:n]) * 1000
        results[k] = {"failed": False, "coverage": float(d.t[mask_f][-1] / d.t[-1]),
                      "termination": res_f["termination"], "metrics": m,
                      "runtime_default_s": wf, "runtime_coarse_s": wc,
                      "mesh_dV_rms_mV": float(np.sqrt(np.mean(dV ** 2))),
                      "mesh_dV_max_mV": float(np.max(np.abs(dV)))}
        curves[k] = (d.t[mask_f], d.V[mask_f], Vf, d.t[mask_c], Vc)
        print(f"  {k:<4}{100 * results[k]['coverage']:>6.1f}%{1000 * m['RMSE_V']:>8.1f}"
              f"{1000 * m['MAE_V']:>8.1f}{1000 * m['MaxAbs_V']:>8.1f}{1000 * m['MeanError_V']:>8.1f}"
              f"{wf:>9.1f}{wc:>9.1f}{results[k]['mesh_dV_rms_mV']:>8.2f}{results[k]['mesh_dV_max_mV']:>8.1f}"
              f"{'' if results[k]['coverage'] > 0.999 else '  ' + res_f['termination']}")

    ok = [r for r in results.values() if not r["failed"]]
    if ok:
        cost = float(np.mean([r["metrics"]["RMSE_V"] for r in ok]))
        t_def = sum(r["runtime_default_s"] for r in ok)
        t_c = sum(r["runtime_coarse_s"] for r in ok)
        dv_max = max(r["mesh_dV_max_mV"] for r in ok)
        dv_rms = max(r["mesh_dV_rms_mV"] for r in ok)
        print(f"\n  Training cost (mean RMSE over the {len(ok)} sets): {1000 * cost:.1f} mV")
        print(f"  One evaluation of all sets: default mesh {t_def:.0f} s, coarse mesh {t_c:.0f} s "
              f"(speed-up x{t_def / max(t_c, 1e-9):.1f})")
        print(f"  Coarse vs default mesh: worst RMS difference {dv_rms:.2f} mV, worst max {dv_max:.1f} mV")
        verdict = ("coarse mesh acceptable for optimization" if dv_rms < 0.1 * 1000 * cost
                   else "coarse mesh NOT accurate enough - use the default mesh")
        print(f"  -> {verdict} (criterion: mesh RMS difference < 10 % of the training error)")
        est = [200, 400]
        print(f"  Estimated optimization time ({est[0]}-{est[1]} evaluations, coarse mesh): "
              f"{est[0] * t_c / 3600:.1f}-{est[1] * t_c / 3600:.1f} h")
        summary = {"model": "task3", "Dn": TASK3_DN, "initial_states": states, "results": results,
                   "training_cost_mean_rmse_V": cost, "eval_time_default_s": t_def,
                   "eval_time_coarse_s": t_c, "mesh_verdict": verdict, "coarse_mesh": COARSE_MESH}
        (OUT / "training_check.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---------------- plot
    keys = [k for k in TRAINING_KEYS if k in curves]
    fig, ax = plt.subplots(len(keys), 2, figsize=(14, 3.0 * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        t, Vm, Vf, tc, Vc = curves[k]
        ax[i, 0].plot(t / 60, Vm, "k", lw=1, label="Measured")
        ax[i, 0].plot(t / 60, Vf, "--", color="tab:orange", lw=1, label="Task 3 model (default mesh)")
        ax[i, 0].plot(tc / 60, Vc, ":", color="tab:blue", lw=1, label="coarse mesh")
        ax[i, 0].set_ylabel("Voltage [V]")
        ax[i, 0].set_title(f"{k}: {DATASETS[k]} (RMSE {1000 * results[k]['metrics']['RMSE_V']:.1f} mV)")
        ax[i, 0].legend(fontsize=7)
        ax[i, 1].plot(t / 60, (Vf - Vm) * 1000, color="tab:red", lw=0.8)
        ax[i, 1].axhline(0, color="k", lw=0.5)
        ax[i, 1].set_ylabel("Error [mV]")
        for a in ax[i]:
            a.grid(alpha=0.3)
    ax[-1, 0].set_xlabel("Time [min]")
    ax[-1, 1].set_xlabel("Time [min]")
    fig.suptitle("Task 4a - training data with the Task 3 model", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "step4a_training_check.png", dpi=130)
    print(f"\nSaved: {OUT / 'step4a_training_check.png'}, training_check.json")


if __name__ == "__main__":
    main()