"""
step5_validate.py - Task 5: validation of the optimized parameter set.

1. HPPC (not used by the optimizer) with the final model, default (fine) mesh:
     measured vs simulated voltage, voltage error, pulse zoom
     comparison: baseline (Task 2) / manually tuned (Task 3) / optimized (Task 4)
2. Error metrics, with the choice explained in the report:
     RMSE (primary, per sample), MAE, max, mean (bias),
     time-weighted RMSE (HPPC is sampled every 0.1 s in pulses but 10 s at rest),
     split: long rest (OCV-like; partly calibrated on HPPC rests in Step 3c, so not
     fully independent) vs dynamic (pulses, relaxation, low-rate steps: independent,
     the optimizer only saw the charge / discharge data)
3. Training data (Charge 1C-4C, Discharge 1C) with the final model, default mesh
4. Physical variables: anode potential vs Li/Li+ (at the separator and x-averaged),
   cathode potential, particle surface / average stoichiometries, electrolyte
   concentration - for HPPC and for the 4C charge (lithium-plating relevance)

SOC axis: referenced to the deepest discharge of the HPPC test (about 2.48 Ah).

Run from the project root (after step4_optimize.py has finished):
    python src\\step5_validate.py
    python src\\step5_validate.py --params results\\step4_cc\\optimized_parameters.json --tag cc
Outputs -> results/step5/
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import DATASETS, RESULTS_DIR, TRAINING_KEYS, load_dataset
from final_model import load_optimized, simulate_optimized
from model_setup import voltage_metrics

OUT = RESULTS_DIR / "step5"
BASE_CSV = RESULTS_DIR / "step2" / "hppc_baseline.csv"
TASK3_CSV = RESULTS_DIR / "step3e" / "hppc_Dn_x10.csv"
REST_A, LONG_REST_S = 0.05, 300.0

PHYS = {
    "anode_sep": ["Negative electrode surface potential difference at separator interface [V]"],
    "anode_xavg": ["X-averaged negative electrode surface potential difference [V]"],
    "anode_ocp": ["X-averaged negative electrode open-circuit potential [V]"],
    "cathode_xavg": ["X-averaged positive electrode surface potential difference [V]"],
    "cathode_ocp": ["X-averaged positive electrode open-circuit potential [V]"],
    "neg_surf": ["X-averaged negative particle surface stoichiometry"],
    "neg_avg": ["Average negative particle stoichiometry", "X-averaged negative particle stoichiometry"],
    "pos_surf": ["X-averaged positive particle surface stoichiometry"],
    "pos_avg": ["Average positive particle stoichiometry", "X-averaged positive particle stoichiometry"],
    "ce": ["Electrolyte concentration [mol.m-3]"],
}


# ------------------------------------------------------------------ helpers
def get_var(sol, names):
    for n in names:
        try:
            return n, np.asarray(sol[n].entries)
        except Exception:  # noqa: BLE001
            continue
    return None, None


def physical_variables(res):
    sol = res["sol"]
    out = {"t": np.asarray(sol["Time [s]"].entries).ravel()}
    for key, names in PHYS.items():
        name, arr = get_var(sol, names)
        if arr is None:
            continue
        if arr.ndim == 2:                                    # (x, t): keep min / max over x
            out[key + "_min"], out[key + "_max"] = arr.min(axis=0), arr.max(axis=0)
        else:
            out[key] = arr.ravel()
    return out


def phase_masks(t, I):
    """long rest (OCV-like) vs dynamic (everything else)."""
    rest = np.abs(I) < REST_A
    idx = np.where(~rest, np.arange(t.size), -1)
    last = np.maximum.accumulate(idx)
    since = np.where(last >= 0, t - t[np.clip(last, 0, None)], np.inf)
    long_rest = rest & (since > LONG_REST_S)
    return long_rest, ~long_rest


def metrics_full(t, I, err):
    """Per-sample metrics + time-weighted RMSE + rest / dynamic split."""
    m = voltage_metrics(err, I)
    w = np.diff(t, prepend=t[0])                             # backward interval length
    w[0] = w[1] if w.size > 1 else 1.0
    m["RMSE_time_weighted_V"] = float(np.sqrt(np.sum(w * err ** 2) / np.sum(w)))
    lr, dyn = phase_masks(t, I)
    m["RMSE_long_rest_V"] = float(np.sqrt(np.mean(err[lr] ** 2))) if lr.any() else float("nan")
    m["RMSE_dynamic_V"] = float(np.sqrt(np.mean(err[dyn] ** 2))) if dyn.any() else float("nan")
    m["share_long_rest_samples"] = float(lr.mean())
    return m


def row(label, m):
    keys = ["RMSE_V", "RMSE_time_weighted_V", "MAE_V", "MaxAbs_V", "MeanError_V",
            "RMSE_long_rest_V", "RMSE_dynamic_V"]
    return f"  {label:<26}" + "".join(f"{1000 * m.get(k, np.nan):>9.1f}" for k in keys)


HEADER = (f"  {'model':<26}{'RMSE':>9}{'RMSE_tw':>9}{'MAE':>9}{'max':>9}{'mean':>9}"
          f"{'rest':>9}{'dynamic':>9}   [mV]")


def plot_physical(pvars, data, V_sim, mask, title, fname):
    rows = [("Terminal voltage [V]", None)]
    if "anode_sep" in pvars or "anode_xavg" in pvars or "anode_sep_min" in pvars:
        rows.append(("Anode potential vs Li/Li+ [V]", "anode"))
    if "cathode_xavg" in pvars:
        rows.append(("Cathode potential vs Li/Li+ [V]", "cathode"))
    if any(k in pvars for k in ["neg_surf", "neg_avg", "pos_surf", "pos_avg"]):
        rows.append(("Stoichiometry [-]", "sto"))
    if "ce_min" in pvars:
        rows.append(("Electrolyte conc. [mol/m3]", "ce"))
    fig, ax = plt.subplots(len(rows), 1, figsize=(12, 2.6 * len(rows)), sharex=True, squeeze=False)
    ax = ax[:, 0]
    th = pvars["t"] / 3600
    for a, (lab, kind) in zip(ax, rows):
        if kind is None:
            a.plot(data.t / 3600, data.V, "k", lw=0.8, label="Measured")
            a.plot(data.t[mask] / 3600, V_sim, "--", color="tab:orange", lw=0.8, label="Optimized model")
        elif kind == "anode":
            if "anode_sep" in pvars:
                a.plot(th, pvars["anode_sep"], color="tab:red", lw=0.8, label="at separator interface")
            if "anode_xavg" in pvars:
                a.plot(th, pvars["anode_xavg"], color="tab:blue", lw=0.8, label="x-averaged")
            if "anode_ocp" in pvars:
                a.plot(th, pvars["anode_ocp"], ":", color="tab:gray", lw=0.8, label="x-averaged OCP")
            a.axhline(0, color="k", lw=0.6, ls="--")
        elif kind == "cathode":
            a.plot(th, pvars["cathode_xavg"], color="tab:green", lw=0.8, label="x-averaged")
            if "cathode_ocp" in pvars:
                a.plot(th, pvars["cathode_ocp"], ":", color="tab:gray", lw=0.8, label="x-averaged OCP")
        elif kind == "sto":
            for k, c, ls, l in [("neg_surf", "tab:blue", "-", "graphite surface"),
                                ("neg_avg", "tab:blue", ":", "graphite average"),
                                ("pos_surf", "tab:red", "-", "LFP surface"),
                                ("pos_avg", "tab:red", ":", "LFP average")]:
                if k in pvars:
                    a.plot(th, pvars[k], ls, color=c, lw=0.8, label=l)
        elif kind == "ce":
            a.plot(th, pvars["ce_min"], color="tab:purple", lw=0.8, label="min over x")
            a.plot(th, pvars["ce_max"], color="tab:olive", lw=0.8, label="max over x")
        a.set_ylabel(lab)
        a.legend(fontsize=7, loc="best")
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("Time [h]")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(RESULTS_DIR / "step4" / "optimized_parameters.json"),
                    help="optimized_parameters.json to validate")
    ap.add_argument("--tag", default="", help="results go to results/step5<_tag>/")
    args = ap.parse_args()
    OUT = RESULTS_DIR / ("step5" + (f"_{args.tag}" if args.tag else ""))
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("TASK 5 - VALIDATION OF THE OPTIMIZED PARAMETER SET")
    print("=" * 66)
    opt = load_optimized(Path(args.params))
    print(f"Parameter file: {args.params}")
    print("Optimized parameters: " + ", ".join(f"{k}={v:.4g}" for k, v in opt["optimized"].items())
          + f", R_c={1000 * opt['contact_resistance_Ohm']:.2f} mOhm"
          + (f", dh={1000 * opt['hysteresis_dh_V']:.1f} mV" if opt.get("hysteresis_fitted") else ""))

    # ---------------- 1. HPPC with the final model (fine mesh)
    hppc = load_dataset("HPPC")
    q_ref = float(-hppc.ah_model().min())
    print(f"\nHPPC: {hppc.t.size} samples, {hppc.t[-1] / 3600:.1f} h; SOC reference (deepest "
          f"discharge) {q_ref:.3f} Ah")
    res, mask, V = simulate_optimized(hppc, "HPPC", opt, mesh=None)
    t, Vm, I = hppc.t[mask], hppc.V[mask], hppc.I_raw[mask]
    err = V - Vm
    cover = t[-1] / hppc.t[-1]
    print(f"Coverage {100 * cover:.1f} %  ({res['termination']}), runtime {res['runtime_s']:.0f} s")
    m_opt = metrics_full(t, I, err)
    pd.DataFrame({"t_s": t, "I_A_charge_pos": I, "V_meas": Vm, "V_sim": V, "error_V": err}
                 ).to_csv(OUT / "hppc_optimized.csv", index=False)

    comp = {"optimized_full": m_opt}
    print("\n1) HPPC VALIDATION - full test")
    print(HEADER)
    if TASK3_CSV.exists():
        d3 = pd.read_csv(TASK3_CSV)
        comp["task3_full"] = metrics_full(d3["t_s"].to_numpy(), d3["I_A_charge_pos"].to_numpy(),
                                          d3["error_V"].to_numpy())
        print(row("Task 3 (manual tuning)", comp["task3_full"]))
    print(row("Task 4 (optimized)", m_opt))

    if BASE_CSV.exists():
        b = pd.read_csv(BASE_CSV)
        t_end = b["t_s"].iloc[-1] - 60
        print(f"\n   on the baseline window 0-{t_end / 3600:.1f} h (the un-tuned model stopped there)")
        print(HEADER)
        wb = b["t_s"] <= t_end
        comp["baseline_window"] = metrics_full(b.loc[wb, "t_s"].to_numpy(), b.loc[wb, "I_A_charge_pos"].to_numpy(),
                                               b.loc[wb, "error_V"].to_numpy())
        print(row("Task 2 baseline", comp["baseline_window"]))
        if TASK3_CSV.exists():
            w3 = d3["t_s"] <= t_end
            comp["task3_window"] = metrics_full(d3.loc[w3, "t_s"].to_numpy(), d3.loc[w3, "I_A_charge_pos"].to_numpy(),
                                                d3.loc[w3, "error_V"].to_numpy())
            print(row("Task 3 (manual tuning)", comp["task3_window"]))
        wo = t <= t_end
        comp["optimized_window"] = metrics_full(t[wo], I[wo], err[wo])
        print(row("Task 4 (optimized)", comp["optimized_window"]))
    if opt.get("hppc_split"):
        from step4_optimize import hppc_split_masks
        tr, ho = hppc_split_masks(hppc)
        tr, ho = tr[mask], ho[mask]
        print("\n   HPPC split used in the optimization: training = odd SOC levels, HELD-OUT = even levels")
        print(HEADER)
        comp["optimized_heldout"] = metrics_full(t[ho], I[ho], err[ho])
        comp["optimized_trainlevels"] = metrics_full(t[tr], I[tr], err[tr])
        if TASK3_CSV.exists():
            e3 = np.interp(t, d3["t_s"].to_numpy(), d3["error_V"].to_numpy())
            comp["task3_heldout"] = metrics_full(t[ho], I[ho], e3[ho])
            print(row("Task 3, held-out levels", comp["task3_heldout"]))
        print(row("optimized, HELD-OUT levels", comp["optimized_heldout"]))
        print(row("optimized, training levels", comp["optimized_trainlevels"]))
    print("  RMSE_tw = time-weighted; rest = long rests (OCV-like, partly calibrated on HPPC in 3c);")
    print("  dynamic = pulses, relaxation, low-rate steps (independent of the optimizer's training data)")

    # ---------------- 2. training data with the final model (fine mesh)
    print("\n2) TRAINING DATA with the final model (default mesh), full curves")
    print(HEADER)
    train = {}
    sims = {}
    for k in TRAINING_KEYS:
        d = load_dataset(k)
        r, mk, Vk = simulate_optimized(d, k, opt, mesh=None)
        sims[k] = (d, r, mk, Vk)
        train[k] = metrics_full(d.t[mk], d.I_raw[mk], Vk - d.V[mk])
        print(row(f"{k} ({DATASETS[k]})", train[k]))
    comp["training"] = train

    # ---------------- 3. physical variables
    phys_h = physical_variables(res)
    d4, r4, m4, V4 = sims["C4"]
    phys_c4 = physical_variables(r4)
    summary_phys = {}
    for lab, pv_ in [("HPPC", phys_h), ("C4", phys_c4)]:
        key = "anode_sep" if "anode_sep" in pv_ else ("anode_xavg" if "anode_xavg" in pv_ else None)
        if key:
            a = pv_[key]
            summary_phys[lab] = {"variable": key, "min_anode_potential_V": float(np.min(a)),
                                 "time_below_0V_s": float(np.sum(np.diff(pv_["t"], prepend=pv_["t"][0])[a < 0]))}
    print("\n3) PHYSICAL VARIABLES")
    for lab, s in summary_phys.items():
        print(f"  {lab}: minimum anode potential vs Li/Li+ ({s['variable']}) {1000 * s['min_anode_potential_V']:.1f} mV, "
              f"time below 0 V {s['time_below_0V_s']:.0f} s"
              + ("  -> lithium-plating risk region (plating is not modelled)" if s["min_anode_potential_V"] < 0 else ""))
    missing = [k for k in PHYS if not any(k == x or x.startswith(k + "_") for x in phys_h)]
    if missing:
        print(f"  (not available in this PyBaMM version: {missing})")
    comp["physical"] = summary_phys
    (OUT / "validation_metrics.json").write_text(json.dumps(comp, indent=2, default=float))

    # ---------------- plots
    th = t / 3600
    soc = 1 + np.interp(t, hppc.t, hppc.ah_model()) / q_ref
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(hppc.t / 3600, hppc.V, "k", lw=0.8, label="Measured")
    if TASK3_CSV.exists():
        ax[0].plot(d3["t_s"] / 3600, d3["V_sim"], ":", color="tab:gray", lw=0.7, label="Task 3 (manual tuning)")
    ax[0].plot(th, V, "--", color="tab:orange", lw=0.8, label="Optimized (Task 4)")
    ax[0].set_ylim(hppc.V.min() - 0.15, hppc.V.max() + 0.15)
    ax[0].set_ylabel("Voltage [V]")
    ax[0].set_title(f"Task 5 - HPPC validation: measured vs simulated (RMSE {1000 * m_opt['RMSE_V']:.1f} mV)")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[1].plot(hppc.t / 3600, hppc.I_raw, color="tab:gray", lw=0.6)
    ax[1].set_ylabel("Current [A]")
    ax[1].set_xlabel("Time [h]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "step5_measured_vs_simulated.png", dpi=150)

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 2, 1.3]})
    lim = np.percentile(np.abs(err), 99.9) * 1000 * 1.1
    if TASK3_CSV.exists():
        ax[0].plot(d3["t_s"] / 3600, d3["error_V"] * 1000, color="tab:gray", lw=0.5, label="Task 3")
    ax[0].plot(th, err * 1000, color="tab:red", lw=0.5, label="Optimized")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel("Time [h]")
    ax[0].set_ylabel("Error (sim - meas) [mV]")
    ax[0].set_title(f"Task 5 - HPPC voltage error (RMSE {1000 * m_opt['RMSE_V']:.1f} mV, "
                    f"time-weighted {1000 * m_opt['RMSE_time_weighted_V']:.1f} mV)")
    ax[0].legend(fontsize=8)
    ax[1].scatter(soc * 100, err * 1000, s=1, c="tab:red", alpha=0.4)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylim(-lim, lim)
    ax[1].invert_xaxis()
    ax[1].set_xlabel(f"SOC [%] (reference: deepest discharge {q_ref:.2f} Ah)")
    ax[1].set_ylabel("Error [mV]")
    ax[2].hist(np.clip(err * 1000, -lim, lim), bins=200, color="tab:red", alpha=0.7)
    ax[2].set_xlabel("Error [mV]")
    ax[2].set_ylabel("Count")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "step5_voltage_error.png", dpi=150)

    cand = np.where((hppc.I_raw < -30) & (hppc.t > 0.4 * hppc.t[-1]))[0]
    if cand.size:
        ts = hppc.t[cand[0]]
        z = (t >= ts - 200) & (t <= ts + 700)
        zb = None
        if BASE_CSV.exists():
            zb = (b["t_s"] >= ts - 200) & (b["t_s"] <= ts + 700)
        fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        ax[0].plot(t[z], Vm[z], "k", label="Measured")
        if zb is not None and zb.any():
            ax[0].plot(b.loc[zb, "t_s"], b.loc[zb, "V_sim"], ":", color="tab:gray", label="Baseline (Task 2)")
        ax[0].plot(t[z], V[z], "--", color="tab:orange", label="Optimized")
        ax[0].set_ylabel("Voltage [V]")
        ax[0].legend(fontsize=8)
        ax[0].set_title("Task 5 - zoom on one HPPC pulse set (same window as Task 2)")
        ax[1].plot(t[z], err[z] * 1000, color="tab:red", label="Optimized")
        if zb is not None and zb.any():
            ax[1].plot(b.loc[zb, "t_s"], b.loc[zb, "error_V"] * 1000, ":", color="tab:gray", label="Baseline")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set_ylabel("Error [mV]")
        ax[1].set_xlabel("Time [s]")
        ax[1].legend(fontsize=8)
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "step5_pulse_zoom.png", dpi=150)

    fig, ax = plt.subplots(len(TRAINING_KEYS), 1, figsize=(12, 2.6 * len(TRAINING_KEYS)), squeeze=False)
    for i, k in enumerate(TRAINING_KEYS):
        d, r, mk, Vk = sims[k]
        a = ax[i, 0]
        a.plot(d.t / 60, d.V, "k", lw=1, label="Measured")
        a.plot(d.t[mk] / 60, Vk, "--", color="tab:orange", lw=1, label="Optimized")
        a.set_title(f"{DATASETS[k]} (RMSE {1000 * train[k]['RMSE_V']:.1f} mV)")
        a.set_ylabel("Voltage [V]")
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    ax[-1, 0].set_xlabel("Time [min]")
    fig.tight_layout()
    fig.savefig(OUT / "step5_training_data.png", dpi=130)

    plot_physical(phys_h, hppc, V, mask, "Task 5 - physical variables, HPPC (optimized model)",
                  "step5_physical_variables_hppc.png")
    plot_physical(phys_c4, d4, V4, m4, "Task 5 - physical variables, 4C charge (optimized model)",
                  "step5_physical_variables_4C_charge.png")
    print(f"\nSaved to {OUT}: step5_measured_vs_simulated.png, step5_voltage_error.png, step5_pulse_zoom.png,")
    print("  step5_training_data.png, step5_physical_variables_hppc.png, step5_physical_variables_4C_charge.png,")
    print("  validation_metrics.json, hppc_optimized.csv")


if __name__ == "__main__":
    main()