"""
step3a_residual_analysis.py - Task 3: when and why do the voltage residuals appear?

Uses the Task 2 baseline result (results/step2/hppc_baseline.csv), no new simulation.

1. Error by operating phase:
      long rest (OCV-like), relaxation (< 300 s after load), low-rate discharge,
      discharge pulse, charge pulse
2. Error by SOC (10 % bins, SOC from measured charge throughput)
3. End-of-rest offset vs SOC  -> thermodynamic (OCV / capacity) mismatch
4. Pulse resistance, measured vs model:
      R_inst = dV/dI at the first sample of a pulse   (~0.1 s: ohmic part)
      R_end  = dV/dI at the end of the pulse          (+ charge transfer + diffusion)
   Only constant-current pulses >= 2 s are used; voltage-limited charge pulses
   (current reduced by the cycler at 3.6 V) are flagged and excluded.
   -> tells which parameter group (ohmic / kinetics / diffusion) is wrong

Run from the project root:
    python src\\step3a_residual_analysis.py                       (baseline, default)
    python src\\step3a_residual_analysis.py --input results\\step3d\\hppc_calibrated.csv --out step3d_residuals --label "calibrated (3c)"
Outputs -> results/step3a/ (or results/<out>/)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import NOMINAL_CAPACITY_AH, RESULTS_DIR, load_dataset

IN_CSV = RESULTS_DIR / "step2" / "hppc_baseline.csv"
OUT = RESULTS_DIR / "step3a"
LABEL = "un-tuned Prada2013 baseline"

REST_A = 0.05          # |I| below this = rest
PULSE_A = 5.0          # |I| above this = pulse
RELAX_S = 300.0        # rest within this time after load = relaxation
LONG_REST_S = 600.0    # rest segments at least this long give OCV-like end points
MIN_PULSE_S = 2.0      # shorter "pulses" are current transitions, not pulses


def segments(mask):
    """Start/end indices (inclusive) of contiguous True runs."""
    m = np.concatenate(([False], mask, [False])).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def main():
    global IN_CSV, OUT, LABEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="CSV with t_s, I_A_charge_pos, V_meas, V_sim, error_V")
    ap.add_argument("--out", default=None, help="output folder name inside results/")
    ap.add_argument("--label", default=None, help="model label for the plot title")
    args = ap.parse_args()
    if args.input:
        from pathlib import Path
        IN_CSV = Path(args.input)
    if args.out:
        OUT = RESULTS_DIR / args.out
    if args.label:
        LABEL = args.label
    OUT.mkdir(parents=True, exist_ok=True)
    if not IN_CSV.exists():
        raise FileNotFoundError(f"{IN_CSV} not found - run step2_baseline_hppc.py first")

    print("=" * 66)
    print(f"RESIDUAL ANALYSIS - {LABEL}")
    print(f"input: {IN_CSV}")
    print("=" * 66)
    df = pd.read_csv(IN_CSV)
    data = load_dataset("HPPC")

    # drop the end-of-run collapse (final 60 s) only if the run stopped before the test end
    if df["t_s"].iloc[-1] < data.t[-1] - 1.0:
        df = df[df["t_s"] < df["t_s"].iloc[-1] - 60].reset_index(drop=True)
    t = df["t_s"].to_numpy()
    I = df["I_A_charge_pos"].to_numpy()
    Vm = df["V_meas"].to_numpy()
    Vs = df["V_sim"].to_numpy()
    e = df["error_V"].to_numpy()

    # SOC from measured charge throughput (backward ZOH, same as model input)
    q_total = -data.ah_model()[-1]                       # Ah discharged over full test
    q = np.interp(t, data.t, data.ah_model())            # Ah (negative = discharged)
    soc = 1.0 + q / q_total
    print(f"Samples analysed: {t.size} (to {t[-1]:.0f} s), SOC range {soc.min():.2f}-{soc.max():.2f}")
    print(f"SOC reference capacity (measured HPPC throughput): {q_total:.3f} Ah")

    # ---------------- 1. phase classification
    rest = np.abs(I) < REST_A
    phase = np.full(t.size, "low-rate discharge", dtype=object)
    phase[I < -PULSE_A] = "discharge pulse"
    phase[I > PULSE_A] = "charge pulse"
    phase[(I > REST_A) & (I <= PULSE_A)] = "low-rate charge"
    load_idx = np.where(~rest)[0]
    last_load_t = np.full(t.size, -np.inf)
    if load_idx.size:
        last = np.maximum.accumulate(np.where(~rest, np.arange(t.size), -1))
        valid = last >= 0
        last_load_t[valid] = t[last[valid]]
    since_load = t - last_load_t
    phase[rest & (since_load <= RELAX_S)] = "relaxation (<300 s)"
    phase[rest & (since_load > RELAX_S)] = "long rest"

    print("\n1) ERROR BY PHASE (sim - meas)")
    print(f"  {'phase':<22}{'share':>7}{'RMSE':>9}{'MAE':>8}{'mean':>8}{'max':>8}  [mV]")
    phase_stats = {}
    order = ["long rest", "relaxation (<300 s)", "low-rate discharge", "low-rate charge",
             "discharge pulse", "charge pulse"]
    for p in order:
        sel = phase == p
        if not sel.any():
            continue
        s = {"share_pct": float(100 * sel.mean()), "RMSE_mV": 1000 * rms(e[sel]),
             "MAE_mV": float(1000 * np.mean(np.abs(e[sel]))),
             "mean_mV": float(1000 * np.mean(e[sel])),
             "max_mV": float(1000 * np.max(np.abs(e[sel])))}
        phase_stats[p] = s
        print(f"  {p:<22}{s['share_pct']:>6.1f}%{s['RMSE_mV']:>9.1f}{s['MAE_mV']:>8.1f}"
              f"{s['mean_mV']:>8.1f}{s['max_mV']:>8.1f}")

    # ---------------- 2. error by SOC bin
    print("\n2) ERROR BY SOC BIN")
    print(f"  {'SOC bin':<12}{'RMSE all':>10}{'RMSE rest':>11}{'RMSE pulse':>12}{'mean rest':>11}  [mV]")
    bins = np.arange(0.0, 1.0001, 0.1)
    soc_stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (soc >= lo) & (soc < hi if hi < 1 else soc <= 1.0001)
        if sel.sum() < 10:
            continue
        r = sel & rest
        pl = sel & (np.abs(I) > PULSE_A)
        row = {"soc_lo": float(lo), "soc_hi": float(hi),
               "RMSE_all_mV": 1000 * rms(e[sel]), "RMSE_rest_mV": 1000 * rms(e[r]),
               "RMSE_pulse_mV": 1000 * rms(e[pl]),
               "mean_rest_mV": float(1000 * np.mean(e[r])) if r.any() else float("nan")}
        soc_stats.append(row)
        print(f"  {lo:.1f}-{hi:.1f}     {row['RMSE_all_mV']:>10.1f}{row['RMSE_rest_mV']:>11.1f}"
              f"{row['RMSE_pulse_mV']:>12.1f}{row['mean_rest_mV']:>11.1f}")

    # ---------------- 3. end-of-rest (OCV-like) offsets
    rest_pts = []
    for a, b in segments(rest):
        if t[b] - t[a] >= LONG_REST_S:
            rest_pts.append({"t_s": float(t[b]), "soc": float(soc[b]),
                             "rest_duration_s": float(t[b] - t[a]),
                             "V_meas": float(Vm[b]), "V_sim": float(Vs[b]),
                             "error_mV": float(1000 * e[b])})
    rp = pd.DataFrame(rest_pts)
    rp.to_csv(OUT / "end_of_rest_points.csv", index=False)
    print(f"\n3) END-OF-REST POINTS (rest >= {LONG_REST_S:.0f} s): {len(rp)}")
    if len(rp):
        print(f"  {'SOC':>6}{'V_meas':>9}{'V_sim':>9}{'error':>9}  [V, V, mV]")
        for _, r in rp.iterrows():
            print(f"  {r['soc']:>6.3f}{r['V_meas']:>9.4f}{r['V_sim']:>9.4f}{r['error_mV']:>9.1f}")

    # ---------------- 4. pulse resistance analysis
    # A valid pulse: |I| > PULSE_A for at least MIN_PULSE_S, current steady within 10 %.
    # Charge pulses whose current is reduced by the cycler's 3.6 V limit are flagged
    # "voltage-limited" and excluded from the resistance statistics.
    pulses = []
    n_short = 0
    for a, b in segments(np.abs(I) > PULSE_A):
        if a == 0:
            continue
        dur = t[b] - t[a - 1]
        if dur < MIN_PULSE_S or b == a:
            n_short += 1
            continue
        i0, v0m, v0s = I[a - 1], Vm[a - 1], Vs[a - 1]
        Ip = I[a:b + 1]
        i_med = float(np.median(Ip))
        # steadiness on the 10-90th percentile, so single ramp/edge samples do not count
        p10, p90 = np.percentile(Ip, [10, 90])
        steady = (p90 - p10) <= 0.10 * abs(i_med)
        # first sample that is actually at pulse current (skips edge/ramp samples)
        at_level = np.where(np.abs(Ip - i_med) <= 0.10 * abs(i_med))[0]
        if at_level.size == 0:
            n_short += 1
            continue
        a_s = a + int(at_level[0])
        dI_inst, dI_end = I[a_s] - i0, I[b] - i0
        c_rate = round(abs(i_med) / NOMINAL_CAPACITY_AH * 2) / 2
        pulses.append({
            "t_start_s": float(t[a - 1]), "duration_s": float(dur),
            "soc": float(soc[a - 1]), "I_A": i_med, "C_rate": float(c_rate),
            "type": "discharge" if i_med < 0 else "charge",
            "steady_current": bool(steady),
            "t_inst_s": float(t[a_s] - t[a - 1]),
            "R_inst_meas_mOhm": float(1000 * (Vm[a_s] - v0m) / dI_inst),
            "R_inst_sim_mOhm": float(1000 * (Vs[a_s] - v0s) / dI_inst),
            "R_end_meas_mOhm": float(1000 * (Vm[b] - v0m) / dI_end),
            "R_end_sim_mOhm": float(1000 * (Vs[b] - v0s) / dI_end),
        })
    pu = pd.DataFrame(pulses)
    pu.to_csv(OUT / "pulse_resistance.csv", index=False)

    print(f"\n4) PULSE RESISTANCE")
    res_summary = {}
    if len(pu):
        ok = pu[pu["steady_current"]]
        print(f"  valid pulses: {len(ok)} | voltage-limited (excluded): "
              f"{int((~pu['steady_current']).sum())} | too short (< {MIN_PULSE_S:.0f} s, excluded): {n_short}")
        print(f"  R_inst taken at the first sample at pulse current "
              f"(median {ok['t_inst_s'].median():.2f} s after pulse start)" if len(ok) else "")
        print(f"  {'type':<10}{'C-rate':>7}{'n':>4}{'dur[s]':>8}{'R_inst meas':>13}{'R_inst sim':>12}"
              f"{'R_end meas':>12}{'R_end sim':>11}  [mOhm, median]")
        for (typ, cr), g in ok.groupby(["type", "C_rate"]):
            row = {k: float(g[k].median()) for k in
                   ["R_inst_meas_mOhm", "R_inst_sim_mOhm", "R_end_meas_mOhm", "R_end_sim_mOhm"]}
            row["n"] = int(len(g))
            row["median_current_A"] = float(g["I_A"].median())
            row["median_duration_s"] = float(g["duration_s"].median())
            row["median_t_inst_s"] = float(g["t_inst_s"].median())
            res_summary[f"{typ}_{cr:g}C"] = row
            print(f"  {typ:<10}{cr:>6g}C{len(g):>4}{row['median_duration_s']:>8.1f}"
                  f"{row['R_inst_meas_mOhm']:>13.2f}{row['R_inst_sim_mOhm']:>12.2f}"
                  f"{row['R_end_meas_mOhm']:>12.2f}{row['R_end_sim_mOhm']:>11.2f}")
        if len(ok):
            ratio_inst = (ok["R_inst_sim_mOhm"] / ok["R_inst_meas_mOhm"]).median()
            ratio_end = (ok["R_end_sim_mOhm"] / ok["R_end_meas_mOhm"]).median()
            print(f"\n  Model/measured ratio (median over valid pulses): "
                  f"instantaneous {ratio_inst:.2f}, end-of-pulse {ratio_end:.2f}")
            res_summary["ratio_inst_sim_over_meas"] = float(ratio_inst)
            res_summary["ratio_end_sim_over_meas"] = float(ratio_end)
        res_summary["n_valid"] = int(len(ok))
        res_summary["n_voltage_limited"] = int((~pu["steady_current"]).sum())
        res_summary["n_too_short"] = int(n_short)

    summary = {"capacity_reference_Ah": float(q_total), "phase": phase_stats,
               "soc_bins": soc_stats, "end_of_rest_points": rest_pts,
               "pulse_resistance": res_summary}
    (OUT / "residual_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- plots
    colors = {"long rest": "tab:blue", "relaxation (<300 s)": "tab:cyan",
              "low-rate discharge": "tab:olive", "low-rate charge": "tab:purple",
              "discharge pulse": "tab:red", "charge pulse": "tab:green"}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for p in order:
        sel = phase == p
        if sel.any():
            ax[0, 0].scatter(soc[sel] * 100, e[sel] * 1000, s=2, c=colors[p], label=p, alpha=0.6)
    ax[0, 0].axhline(0, color="k", lw=0.5)
    ax[0, 0].invert_xaxis()
    ax[0, 0].set_xlabel("SOC [%] (measured)")
    ax[0, 0].set_ylabel("Error (sim - meas) [mV]")
    ax[0, 0].set_title("Error vs SOC, by phase")
    ax[0, 0].legend(markerscale=5, fontsize=8)

    names = [p for p in order if p in phase_stats]
    ax[0, 1].bar(range(len(names)), [phase_stats[p]["RMSE_mV"] for p in names],
                 color=[colors[p] for p in names])
    ax[0, 1].set_xticks(range(len(names)))
    ax[0, 1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax[0, 1].set_ylabel("RMSE [mV]")
    ax[0, 1].set_title("RMSE by phase")

    if len(rp):
        ax[1, 0].plot(rp["soc"] * 100, rp["error_mV"], "o-", color="tab:blue")
        ax[1, 0].axhline(0, color="k", lw=0.5)
        ax[1, 0].invert_xaxis()
    ax[1, 0].set_xlabel("SOC [%] (measured)")
    ax[1, 0].set_ylabel("End-of-rest error [mV]")
    ax[1, 0].set_title("Rest (OCV-like) offset vs SOC: thermodynamic mismatch")

    if len(pu):
        for typ, mk in [("discharge", "v"), ("charge", "^")]:
            g = pu[(pu["type"] == typ) & pu["steady_current"]]
            ax[1, 1].scatter(g["soc"] * 100, g["R_end_meas_mOhm"], marker=mk, c="k",
                             label=f"{typ} measured (end)")
            ax[1, 1].scatter(g["soc"] * 100, g["R_end_sim_mOhm"], marker=mk, c="tab:orange",
                             label=f"{typ} model (end)")
            ax[1, 1].scatter(g["soc"] * 100, g["R_inst_meas_mOhm"], marker=mk, facecolors="none",
                             edgecolors="k", label=f"{typ} measured (inst.)")
            ax[1, 1].scatter(g["soc"] * 100, g["R_inst_sim_mOhm"], marker=mk, facecolors="none",
                             edgecolors="tab:orange", label=f"{typ} model (inst.)")
        ax[1, 1].invert_xaxis()
        ax[1, 1].legend(fontsize=7, ncol=2)
    ax[1, 1].set_xlabel("SOC [%] (measured)")
    ax[1, 1].set_ylabel("Pulse resistance [mOhm]")
    ax[1, 1].set_title("Pulse resistance (valid constant-current pulses)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(f"Residual analysis - {LABEL}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "residual_analysis.png", dpi=150)

    print(f"\nSaved: {OUT / 'residual_analysis.png'}")
    print(f"Saved: {OUT / 'residual_summary.json'}, end_of_rest_points.csv, pulse_resistance.csv")


if __name__ == "__main__":
    main()