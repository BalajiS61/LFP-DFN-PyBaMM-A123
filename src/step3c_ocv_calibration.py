"""
step3c_ocv_calibration.py - Task 3: manual tuning of capacity and OCP-related parameters.

Electrode balancing (OCV fitting). Discharging from full along the lithium-conserving line:
    x(Ah) = x100 - Ah / Q_n        (negative / graphite stoichiometry)
    y(Ah) = y100 + Ah / Q_p        (positive / LFP stoichiometry)
    OCV(Ah) = U_p(y) - U_n(x)
Fitted parameters: x100, y100, Q_n.  Q_p is FIXED at the Prada2013 value:
the LFP OCP is flat on its plateau, so the OCV carries almost no information on
Q_p (a free fit drifts to the geometry bound). This is demonstrated by a profile
scan over Q_p, saved as a separate result.

Data (from step3b, results/step3b/measured_ocv_points.csv):
    12 relaxation-corrected HPPC rest points   sigma = 3 mV
    HPPC start (SOC 1, rest time unknown)      sigma = 20 mV
    (no end-of-discharge OCV constraint: the rest at 2.371 Ah relaxes to 3.176 V, so the
     final drop to 2.5 V under load is dynamic polarization, not an OCV feature)
Physical constraint: neither electrode may be exhausted before 2.46 Ah
(largest measured discharge capacity, 1C discharge 2.451 Ah).

OCP candidates (positive = Prada2013 LFP in all cases):
    negative from Prada2013, Chen2020 and Ecker2015 (external literature graphite
    OCPs). Identical functions are detected and skipped (PyBaMM's Prada2013 set
    uses the Chen2020 graphite OCP).
The candidate with the lowest error at the rest points is selected.

Translation to PyBaMM parameters:
    active material fraction eps_new = eps_Prada * Q_fit / Q_Prada (each electrode)
    initial state of each test from the lithium line (x, y at Ah from full)

Run from the project root:
    python src\\step3c_ocv_calibration.py
Outputs -> results/step3c/ocv_calibration.json (+ plots, tables)
"""
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
from scipy.optimize import least_squares

from data_loader import RESULTS_DIR, load_dataset
from model_setup import PARAM_SET, base_parameter_values, electrode_balance
from ocv_tools import eval_ocp, full_cell_ocv, stoich_window

IN_CSV = RESULTS_DIR / "step3b" / "measured_ocv_points.csv"
OUT = RESULTS_DIR / "step3c"
OUT.mkdir(parents=True, exist_ok=True)

SIGMA_REST, SIGMA_TOP = 0.003, 0.020
AH_MIN_CAPACITY = 2.46          # electrodes must not run out before this
NEG_SOURCES = ["Prada2013", "Chen2020", "Ecker2015"]
STO = np.linspace(0.0005, 0.9995, 4000)


def tabulate(pv, electrode):
    return eval_ocp(pv, electrode, STO)


def make_model(Up_tab, Un_tab):
    def ocv(p, ah):
        x100, y100, qn, qp = p
        x = x100 - ah / qn
        y = y100 + ah / qp
        return np.interp(y, STO, Up_tab) - np.interp(x, STO, Un_tab), x, y
    return ocv


def fit_candidate(ocv, ah, v, sig, ah_knee, q_max, qp_fixed, starts=None):
    """Fit x100, y100, Q_n with Q_p fixed. Returns (params[4], cost) or (None, inf)."""
    def resid(p3):
        p = [p3[0], p3[1], p3[2], qp_fixed]
        vm, _, _ = ocv(p, ah)
        r = (vm - v) / sig
        x_end = p3[0] - AH_MIN_CAPACITY / p3[2]
        y_end = p3[1] + AH_MIN_CAPACITY / qp_fixed
        pen = np.array([max(0.0, 0.002 - x_end), max(0.0, y_end - 0.998)]) * 1e4
        return np.concatenate([r, pen])

    # each electrode must at least hold the measured capacity; geometry upper bound
    lo = [0.40, 0.0005, 0.98 * AH_MIN_CAPACITY]
    hi = [0.99, 0.30, q_max[0]]
    if starts is None:
        starts = list(itertools.product([0.6, 0.8, 0.95], [0.005, 0.03, 0.1],
                                        [2.6, 3.0, 0.99 * q_max[0]]))
    best = None
    for st in starts:
        p0 = np.clip(st, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), x_scale=[0.1, 0.01, 0.5], max_nfev=4000)
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None, np.inf
    return np.array([best.x[0], best.x[1], best.x[2], qp_fixed]), best.cost


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"{IN_CSV} not found - run step3b_ocv_analysis.py first")
    print("=" * 66)
    print("TASK 3c - OCV CALIBRATION (stoichiometry window + capacity)")
    print("=" * 66)
    pts = pd.read_csv(IN_CSV)
    ah = pts["Ah_discharged"].to_numpy()
    v = pts["V_ocv"].to_numpy()
    is_rest = (pts["source"] == "HPPC rest").to_numpy()
    sig = np.where(is_rest, SIGMA_REST, SIGMA_TOP)
    hppc = load_dataset("HPPC")
    ah_knee = float(-hppc.ah_model()[-1])
    print(f"OCV points: {is_rest.sum()} rests (sigma {1000 * SIGMA_REST:.0f} mV) + "
          f"{(~is_rest).sum()} start point (sigma {1000 * SIGMA_TOP:.0f} mV)")
    print(f"Constraint: no electrode exhausted before {AH_MIN_CAPACITY} Ah "
          f"(measured HPPC capacity {ah_knee:.3f} Ah)")

    pv_prada = base_parameter_values()
    w_prada = stoich_window(pv_prada)
    Up_tab = tabulate(pv_prada, "positive")
    bal_prada = electrode_balance(pv_prada)
    eps_n0 = float(pv_prada["Negative electrode active material volume fraction"])
    eps_p0 = float(pv_prada["Positive electrode active material volume fraction"])
    por_n = float(pv_prada["Negative electrode porosity"])
    por_p = float(pv_prada["Positive electrode porosity"])
    # largest capacity the Prada electrode geometry can hold (active + porosity = 1)
    q_max = (bal_prada["Q_n_Ah"] * (1 - por_n) / eps_n0, bal_prada["Q_p_Ah"] * (1 - por_p) / eps_p0)
    print(f"Geometry limit (active + porosity <= 1): Q_n <= {q_max[0]:.3f} Ah, Q_p <= {q_max[1]:.3f} Ah")
    if min(q_max) <= 0.98 * AH_MIN_CAPACITY:
        raise RuntimeError(
            f"The {PARAM_SET} electrode geometry cannot hold {AH_MIN_CAPACITY} Ah even at 100 % "
            f"solid volume (Q_n <= {q_max[0]:.3f}, Q_p <= {q_max[1]:.3f} Ah). "
            "Electrode area or thickness would have to be changed.")

    # Prada original on the same metric
    soc_grid = np.linspace(0, 1, 401)
    ocv_prada = full_cell_ocv(pv_prada, w_prada, soc_grid)
    ah_grid_prada = (1 - soc_grid) * w_prada["capacity_Ah"]
    o = np.argsort(ah_grid_prada)
    v_prada_pts = np.interp(ah, ah_grid_prada[o], ocv_prada[o], right=np.nan)
    e0 = (v_prada_pts - v)[is_rest] * 1000
    rmse_prada = float(np.sqrt(np.nanmean(e0 ** 2)))
    print(f"\nUn-tuned {PARAM_SET}: RMSE at rest points (same Ah) {rmse_prada:.1f} mV "
          f"(points beyond its capacity excluded)")

    qp_fixed = bal_prada["Q_p_Ah"]
    print(f"Q_p fixed at the {PARAM_SET} value {qp_fixed:.3f} Ah (not identifiable from OCV, see profile scan)")

    results = []
    tables = {}
    print(f"\n{'negative OCP':<14}{'RMSE rest':>10}{'max rest':>10}{'top':>8}{'OCVend':>8}"
          f"{'x100':>8}{'y100':>8}{'Q_n':>7}{'Q_p':>7}  [mV, mV, mV, V; -, -, Ah, Ah]")
    for src in NEG_SOURCES:
        try:
            pv_src = pybamm.ParameterValues(src)
            Un_tab = tabulate(pv_src, "negative")
        except Exception as e:  # noqa: BLE001
            print(f"{src:<14} not available ({type(e).__name__})")
            continue
        same_as = [k for k, tab in tables.items() if np.max(np.abs(tab - Un_tab)) < 1e-9]
        tables[src] = Un_tab
        if same_as:
            print(f"{src:<14} identical OCP function to {same_as[0]} -> skipped (same result)")
            continue
        ocv = make_model(Up_tab, Un_tab)
        p, _ = fit_candidate(ocv, ah, v, sig, ah_knee, q_max, qp_fixed)
        if p is None:
            print(f"{src:<14} fit failed")
            continue
        vm, x, y = ocv(p, ah)
        err = (vm - v) * 1000
        vk, _, _ = ocv(p, np.array([ah_knee]))
        rmse = float(np.sqrt(np.mean(err[is_rest] ** 2)))
        res = {"negative_ocp_source": src, "x100": p[0], "y100": p[1], "Q_n_Ah": p[2],
               "Q_p_Ah": p[3], "rmse_rest_mV": rmse,
               "max_abs_rest_mV": float(np.max(np.abs(err[is_rest]))),
               "top_error_mV": float(err[~is_rest][0]), "knee_ocv_V": float(vk[0]),
               "Q_n_at_bound": bool(p[2] > 0.995 * q_max[0]),
               "errors_mV": err.tolist(), "Un_tab": Un_tab}
        results.append(res)
        print(f"{src:<14}{rmse:>10.1f}{res['max_abs_rest_mV']:>10.1f}{res['top_error_mV']:>8.1f}"
              f"{vk[0]:>8.3f}{p[0]:>8.4f}{p[1]:>8.4f}{p[2]:>7.3f}{p[3]:>7.3f}"
              f"{'  (Q_n at bound)' if res['Q_n_at_bound'] else ''}")

    if not results:
        raise RuntimeError("no OCP candidate could be fitted")
    best = min(results, key=lambda d: d["rmse_rest_mV"])
    src = best["negative_ocp_source"]
    x100, y100, qn, qp = best["x100"], best["y100"], best["Q_n_Ah"], best["Q_p_Ah"]

    # ---------------- identifiability: profile scan over Q_p (best OCP source)
    ocv_best = make_model(Up_tab, best["Un_tab"])
    qp_scan = np.linspace(0.98 * AH_MIN_CAPACITY, q_max[1], 12)
    prof = []
    st = [[best["x100"], best["y100"], best["Q_n_Ah"]]]
    print(f"\nIDENTIFIABILITY - profile scan over Q_p (graphite {src}; x100, y100, Q_n refitted)")
    print(f"  {'Q_p [Ah]':>9}{'RMSE rest [mV]':>16}{'N/P':>7}")
    for qpv in qp_scan:
        p, _ = fit_candidate(ocv_best, ah, v, sig, ah_knee, q_max, qpv,
                             starts=st + list(itertools.product([0.7, 0.95], [0.005, 0.05], [2.7, 3.1])))
        if p is None:
            continue
        vm, _, _ = ocv_best(p, ah)
        r = float(np.sqrt(np.mean(((vm - v) * 1000)[is_rest] ** 2)))
        prof.append({"Q_p_Ah": float(qpv), "rmse_rest_mV": r, "N_P": float(p[2] / qpv)})
        print(f"  {qpv:>9.3f}{r:>16.2f}{p[2] / qpv:>7.3f}")
    if prof:
        rr = np.array([d["rmse_rest_mV"] for d in prof])
        qq = np.array([d["Q_p_Ah"] for d in prof])
        ok = rr <= rr.min() + 1.0                      # within 1 mV of the best fit
        q_lo = float(qq[ok].min())
        upper_open = bool(ok[-1])
        print(f"  Q_p values fitting within 1 mV of the best: {q_lo:.2f} Ah"
              f"{' up to the geometry limit' if upper_open else f' to {qq[ok].max():.2f} Ah'}")
        print(f"  -> OCV gives a LOWER bound for Q_p (about {q_lo:.2f} Ah) but "
              f"{'no upper bound' if upper_open else 'a narrow range'}; "
              f"{PARAM_SET} value {qp_fixed:.3f} Ah is "
              f"{'inside' if qp_fixed >= q_lo else 'OUTSIDE'} the data-consistent range")
    pd.DataFrame(prof).to_csv(OUT / "qp_profile_scan.csv", index=False)

    # window at the measured end and the limiting electrode
    x_end, y_end = x100 - ah_knee / qn, y100 + ah_knee / qp
    eps_n = eps_n0 * qn / bal_prada["Q_n_Ah"]
    eps_p = eps_p0 * qp / bal_prada["Q_p_Ah"]
    cyc_li_Ah = x100 * qn + y100 * qp

    print(f"\nSELECTED: negative OCP from {src} (lowest rest-point RMSE)")
    print(f"  RMSE at rest points {best['rmse_rest_mV']:.1f} mV (un-tuned {PARAM_SET}: {rmse_prada:.1f} mV)")
    print(f"  negative x: {x100:.4f} (full) -> {x_end:.4f} at {ah_knee:.3f} Ah")
    print(f"  positive y: {y100:.4f} (full) -> {y_end:.4f} at {ah_knee:.3f} Ah")
    print(f"  Q_n {qn:.3f} Ah (Prada {bal_prada['Q_n_Ah']:.3f}), Q_p {qp:.3f} Ah "
          f"(Prada {bal_prada['Q_p_Ah']:.3f}), N/P ratio {qn / qp:.3f}")
    print(f"  active material fraction: negative {eps_n0:.4f} -> {eps_n:.4f} (x{eps_n / eps_n0:.3f}), "
          f"positive {eps_p0:.4f} -> {eps_p:.4f} (x{eps_p / eps_p0:.3f})")
    print(f"  volume check: negative active {eps_n:.4f} + porosity {por_n:.4f} = {eps_n + por_n:.4f}, "
          f"positive active {eps_p:.4f} + porosity {por_p:.4f} = {eps_p + por_p:.4f} (must be <= 1)")
    at_bound = [n for n, val, lim in [("Q_n", qn, q_max[0]), ("Q_p", qp, q_max[1])] if val > 0.995 * lim]
    if at_bound:
        print(f"  NOTE: {', '.join(at_bound)} at the geometry limit")
    print(f"  cyclable lithium (x100 Q_n + y100 Q_p) {cyc_li_Ah:.3f} Ah")
    print(f"  remaining before an electrode is exhausted at {ah_knee:.3f} Ah: "
          f"negative {x_end * qn * 1000:.0f} mAh, positive {(1 - y_end) * qp * 1000:.0f} mAh")
    print("\n  per-point error after calibration [mV]:")
    for s, e in zip(pts["soc"], best["errors_mV"]):
        print(f"    SOC {s:.3f}: {e:+.1f}")

    calib = {
        "method": "OCV fit of the electrode stoichiometry window and capacities",
        "positive_ocp_source": PARAM_SET, "negative_ocp_source": src,
        "x100": x100, "y100": y100, "Q_n_Ah": qn, "Q_p_Ah": qp,
        "Negative electrode active material volume fraction": eps_n,
        "Positive electrode active material volume fraction": eps_p,
        "prada_active_material_fraction": {"negative": eps_n0, "positive": eps_p0},
        "porosity": {"negative": por_n, "positive": por_p},
        "geometry_capacity_limit_Ah": {"negative": q_max[0], "positive": q_max[1]},
        "reference_capacity_Ah": ah_knee,
        "rmse_rest_mV": best["rmse_rest_mV"], "rmse_rest_untuned_mV": rmse_prada,
        "Q_p_fixed": True,
        "Q_p_fixed_reason": "LFP OCP flat on its plateau: Q_p not identifiable from OCV "
                            "(profile scan in qp_profile_scan.csv); kept at the Prada2013 value",
        "qp_profile_scan": prof,
        "candidates": [{k: d[k] for k in d if k not in ("Un_tab", "errors_mV")} for d in results],
        "sigma_mV": {"rest": 1000 * SIGMA_REST, "start": 1000 * SIGMA_TOP},
        "ocv_at_measured_end_V": best["knee_ocv_V"],
        "note_end_of_discharge": "OCV at the measured end stays near 3.15 V (rest at 2.371 Ah relaxes "
                                 "to 3.176 V): the final drop to 2.5 V under load is dynamic polarization, "
                                 "not thermodynamics",
    }
    (OUT / "ocv_calibration.json").write_text(json.dumps(calib, indent=2))
    pts.assign(**{f"err_{d['negative_ocp_source']}_mV": d["errors_mV"] for d in results}
               ).to_csv(OUT / "ocv_fit_points.csv", index=False)

    # ---------------- plots
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ah_f = np.linspace(0, ah_knee, 400)
    a = ax[0, 0]
    a.plot(ah, v, "o", color="k", label="Measured OCV")
    a.plot(ah_grid_prada, ocv_prada, ":", color="tab:gray", label=f"{PARAM_SET} un-tuned")
    for d in results:
        vm, _, _ = make_model(Up_tab, d["Un_tab"])([d["x100"], d["y100"], d["Q_n_Ah"], d["Q_p_Ah"]], ah_f)
        a.plot(ah_f, vm, lw=1.5 if d is best else 1,
               label=f"fit, graphite {d['negative_ocp_source']} ({d['rmse_rest_mV']:.1f} mV)")
    a.set_ylim(2.6, 3.65)
    a.set_xlabel("Charge discharged from full [Ah]")
    a.set_ylabel("OCV [V]")
    a.set_title("OCV calibration")
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(ah, v, "o", color="k", label="Measured OCV")
    for d in results:
        vm, _, _ = make_model(Up_tab, d["Un_tab"])([d["x100"], d["y100"], d["Q_n_Ah"], d["Q_p_Ah"]], ah_f)
        a.plot(ah_f, vm, lw=1.5 if d is best else 1, label=d["negative_ocp_source"])
    a.set_xlim(0.05, 2.40)
    a.set_ylim(3.15, 3.36)
    a.set_xlabel("Charge discharged from full [Ah]")
    a.set_ylabel("OCV [V]")
    a.set_title("Zoom on the plateau region")
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(ah[is_rest], e0, "x:", color="tab:gray", label=f"{PARAM_SET} un-tuned")
    for d in results:
        a.plot(ah, d["errors_mV"], "o-", label=d["negative_ocp_source"])
    a.axhline(0, color="k", lw=0.5)
    a.set_ylim(-60, 60)
    a.set_xlabel("Charge discharged from full [Ah]")
    a.set_ylabel("Model - measured OCV [mV]")
    a.set_title("OCV error at the measured points (off-scale points clipped)")
    a.legend(fontsize=8)

    a = ax[1, 1]
    a.plot(STO, Up_tab, color="tab:red", label="positive LFP (Prada2013)")
    a.plot(STO, best["Un_tab"], color="tab:gray", label=f"negative graphite ({src})")
    a.axvspan(y100, y_end, color="tab:red", alpha=0.12, label="positive window (calibrated)")
    a.axvspan(x_end, x100, color="tab:gray", alpha=0.12, label="negative window (calibrated)")
    a.set_ylim(0, 4.0)
    a.set_xlabel("Stoichiometry [-]")
    a.set_ylabel("OCP vs Li/Li+ [V]")
    a.set_title("Calibrated electrode windows")
    a.legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("Task 3c - OCV calibration: stoichiometry window and electrode capacities", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "step3c_ocv_calibration.png", dpi=150)
    if prof:
        fig2, a2 = plt.subplots(figsize=(7, 4.5))
        a2.plot([d["Q_p_Ah"] for d in prof], [d["rmse_rest_mV"] for d in prof], "o-")
        a2.axvline(qp_fixed, color="tab:orange", ls="--", label=f"{PARAM_SET} value {qp_fixed:.3f} Ah (used)")
        a2.set_xlabel("Positive electrode capacity Q_p [Ah] (fixed in each refit)")
        a2.set_ylabel("RMSE at rest points [mV]")
        a2.set_title("Identifiability of Q_p from OCV (profile scan)")
        a2.grid(alpha=0.3)
        a2.legend()
        fig2.tight_layout()
        fig2.savefig(OUT / "step3c_qp_profile.png", dpi=150)
    print(f"\nSaved: {OUT / 'ocv_calibration.json'}")
    print(f"Saved: {OUT / 'step3c_ocv_calibration.png'}, step3c_qp_profile.png, "
          "ocv_fit_points.csv, qp_profile_scan.csv")


if __name__ == "__main__":
    main()