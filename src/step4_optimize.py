"""
step4_optimize.py - Task 4: parameter optimization on the training data
(Charge 1C/2C/3C/4C + Discharge 1C), starting from the Task 3 manually tuned model.

Model voltage
    V = V_DFN(k_n, k_p, D_n, D_p) + R_c * I + dh * eta(t)
      V_DFN : PyBaMM DFN (Step 3c thermodynamics), coarse mesh (checked in step4a)
      R_c   : lumped contact / ohmic resistance (series resistance -> exactly additive)
      dh    : optional LFP OCV hysteresis between discharge and charge branch (kept as a free
              linear term; the --diagnose run found no support for it on the CC parts). A uniform shift of
              the positive OCP shifts the terminal voltage by exactly that amount, so this
              equals a two-branch positive OCP. eta(t) in [0, 1] is a one-state hysteresis
              state: d(eta)/dt = kappa |I| / (3600 Q) (u - eta), u = 1 on charge, 0 on
              discharge, held at rest. Initial branch from the test history: charge tests
              start after a discharge (eta = 0); D1 and HPPC start after a full charge (eta = 1).
      I     : current, charge positive

Optimizer
    outer: 4 DFN parameters in log10 space (k_n, k_p factors on the exchange-current
           densities, D_n absolute, D_p factor), scipy least_squares 'trf' (gradient-based,
           finite-difference Jacobian), bounds from Steps 3a-3e
    inner: R_c and dh solved exactly by bounded linear least squares for every DFN run
           (variable projection) - no extra simulations
    cost : each training set weighted equally (residuals scaled by 1/sqrt(N_set))
    speed: the 5 training sets are simulated in parallel processes

Run from the project root:
    python src\\step4_optimize.py --check            (1 evaluation: tests the hysteresis term)
    python src\\step4_optimize.py --diagnose         (error by phase; hysteresis test on CC parts)
    python src\\step4_optimize.py                    (full optimization, default kappa 30)
    python src\\step4_optimize.py --kappa 10         (choose kappa from --check)
    python src\\step4_optimize.py --resume           (continue from the best saved point)
    python src\\step4_optimize.py --cost-window cc --tag cc   (fit CC + discharge, CV excluded)
    python src\\step4_optimize.py --fit k_n,k_p,D_n,D_p,b_bot --bounds D_n=3e-14:3e-13 --hppc-split
                                    --cost-window cc --tag split   (final: + HPPC odd levels, even held out)
Outputs -> results/step4/  (optimized_parameters.json, history, plots)
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybamm
from scipy.optimize import least_squares, lsq_linear

from calibration import (calibrated_parameter_values, load_ocv_calibration,
                         scale_parameter, set_state_from_full)
from data_loader import DATASETS, RESULTS_DIR, TRAINING_KEYS, load_dataset
from model_setup import COARSE_MESH, compare_on_data_grid, simulate, voltage_metrics

OUT = RESULTS_DIR / "step4"
HISTORY = OUT / "history.jsonl"
BEST = OUT / "best_so_far.json"
COST_WINDOW = "all"                    # "all": everything except the initial history window
                                       # "cc" : charge tests constant-current part only (CV phase excluded)

# name, kind, PyBaMM parameter, initial, lower, upper
#   kind: "factor" (x Prada2013, log10-fitted), "absolute" (log10-fitted), "linear" (shape exponent)
ALL_PARAMS = {
    "k_n": ("factor", "Negative electrode exchange-current density [A.m-2]", 1.0, 0.1, 100.0),
    "k_p": ("factor", "Positive electrode exchange-current density [A.m-2]", 1.0, 0.1, 100.0),
    "D_n": ("absolute", "Negative particle diffusivity [m2.s-1]", 3.0e-14, 1.0e-14, 3.0e-13),
    "D_p": ("factor", "Positive particle diffusivity [m2.s-1]", 1.0, 0.001, 1000.0),
    "a_top": ("linear", "LFP diffusivity drop near y=0 (charged) [decades]", 1.0, 0.0, 4.0),
    "a_bot": ("linear", "LFP diffusivity drop near y=1 (discharged) [decades]", 1.0, 0.0, 4.0),
    "b_bot": ("linear", "graphite diffusivity drop near x=0 (empty) [decades]", 1.0, 0.0, 4.0),
}
BASE_SET = ["k_n", "k_p", "D_n", "D_p"]
SHAPE_SET = ["k_n", "D_n", "D_p", "a_top", "b_bot"]      # k_p fixed (not identifiable in Run A);
                                                         # a_bot dropped: no effect (LFP never reaches y~1, probe 4b)
DP_WIDTH = 0.05                                          # width of the end regions in stoichiometry
LIN_SCALE = 3.0                                          # z units -> shape exponent (step 0.03 -> 0.09)


def make_params(names, init=None):
    init = init or {}
    return [(n, ALL_PARAMS[n][0], ALL_PARAMS[n][1], init.get(n, ALL_PARAMS[n][2]),
             ALL_PARAMS[n][3], ALL_PARAMS[n][4]) for n in names]


def bounds_for(params):
    lo, hi = [], []
    for name, kind, _, ini, low, up in params:
        if kind == "linear":
            lo.append((low - ini) / LIN_SCALE)
            hi.append((up - ini) / LIN_SCALE)
        else:
            lo.append(np.log10(low) - np.log10(ini))
            hi.append(np.log10(up) - np.log10(ini))
    return np.array(lo), np.array(hi)


PARAMS = make_params(BASE_SET)
RC_MAX, DH_MAX = 0.020, 0.060          # Ohm, V
Q_REF = 2.45                           # Ah, for the hysteresis rate
ETA0 = {"C1": 0.0, "C2": 0.0, "C3": 0.0, "C4": 0.0, "D1": 1.0, "HPPC": 1.0}
PENALTY_V = 0.3                        # residual for samples a failed / stopped run cannot cover
EXCLUDE_AH = 0.15                      # charge tests: initial history window excluded from the cost
JAC_STEP = 0.03                        # finite-difference step in log10 units (about 7 % change)
LO, HI = bounds_for(PARAMS)


# ------------------------------------------------------------------ model pieces
def start_from_full(data, key):
    """Ah below full at the start: charge tests end full (CV cut-off); D1 / HPPC start full."""
    return float(data.ah_model()[-1]) if key.startswith("C") else 0.0


def training_mask(data, key, window=None):
    """
    Samples used in the cost. Charge tests start while the cell is still relaxing from the
    preceding discharge (voltage below OCV); the model starts in a uniform relaxed state and
    cannot represent that history. Their first EXCLUDE_AH of charge (incl. the initial rest)
    is therefore excluded. Everything else - CC, CV phase, end of discharge, rests - is used.
    """
    window = window or COST_WINDOW
    if key == "HPPC":
        return hppc_split_masks(data)[0]
    if key.startswith("C"):
        m = data.ah_model() >= EXCLUDE_AH
        if window == "cc":
            # CV phase excluded: the cell is held at 3.6 V while the current tapers; this needs a
            # slow top-of-charge relaxation that a constant-diffusivity DFN cannot represent
            m &= data.I_raw > 0.95 * data.I_raw.max()
        return m
    return np.ones(data.t.size, bool)


def hppc_levels(data, min_rest_s=600.0):
    """
    SOC level index per HPPC sample: number of long rests (>= min_rest_s) that have ended before it.
    Each pulse set follows a long rest, so samples after rest k belong to level k.
    Returns (level array, long-rest mask).
    """
    rest = np.abs(data.I_raw) < 0.05
    m = np.concatenate(([False], rest, [False])).astype(int)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    level = np.zeros(data.t.size, int)
    long_rest = np.zeros(data.t.size, bool)
    k = 0
    for a, b in zip(starts, ends):
        if data.t[b] - data.t[a] >= min_rest_s:
            long_rest[a:b + 1] = True
            k += 1
            level[b + 1:] = k
    return level, long_rest


def hppc_split_masks(data):
    """Training = dynamic samples (no long rests) of odd SOC levels; held-out = even levels."""
    level, long_rest = hppc_levels(data)
    train = (level % 2 == 1) & ~long_rest
    held = (level % 2 == 0) & (level > 0) & ~long_rest
    return train, held


def values_from_z(z):
    out = {}
    for i, (name, kind, _, ini, _, _) in enumerate(PARAMS):
        out[name] = float(ini + LIN_SCALE * z[i]) if kind == "linear" else float(ini * 10 ** z[i])
    return out


def apply_dp_shape(pv, a_top, a_bot):
    """State-dependent LFP diffusivity: D_p(y) = D_p,0(y) * 10^-(a_top e^(-y/w) + a_bot e^(-(1-y)/w))."""
    name = ALL_PARAMS["D_p"][1]
    base = pv[name]
    ln10 = float(np.log(10.0))

    a_t, a_b = float(a_top), float(a_bot)

    def dp_shaped(sto, T):
        b = base(sto, T) if callable(base) else base
        g = a_t * pybamm.exp(-sto / DP_WIDTH) + a_b * pybamm.exp(-(1 - sto) / DP_WIDTH)
        return b * pybamm.exp(-ln10 * g)
    pv.update({name: dp_shaped})
    return pv


def apply_dn_shape(pv, b_bot):
    """State-dependent graphite diffusivity: D_n(x) = D_n,0(x) * 10^-(b_bot e^(-x/w))  (slow near empty)."""
    name = ALL_PARAMS["D_n"][1]
    base = pv[name]
    ln10 = float(np.log(10.0))
    b_b = float(b_bot)

    def dn_shaped(sto, T):
        b = base(sto, T) if callable(base) else base
        return b * pybamm.exp(-ln10 * b_b * pybamm.exp(-sto / DP_WIDTH))
    pv.update({name: dn_shaped})
    return pv


def build_parameter_values(values, calib=None):
    """Step 3c calibrated parameters + optimized DFN parameters."""
    calib = calib or load_ocv_calibration()
    pv = calibrated_parameter_values(calib)
    for name, (kind, pname, *_) in ALL_PARAMS.items():
        if name not in values or kind == "linear":
            continue                               # not fitted -> Prada2013 / Task 3 value
        if kind == "absolute":
            pv.update({pname: values[name]})
        else:
            scale_parameter(pv, pname, values[name])
    if "D_n" not in values:
        pv.update({ALL_PARAMS["D_n"][1]: ALL_PARAMS["D_n"][2]})
    if "a_top" in values or "a_bot" in values:
        apply_dp_shape(pv, values.get("a_top", 0.0), values.get("a_bot", 0.0))
    if "b_bot" in values:
        apply_dn_shape(pv, values["b_bot"])
    return pv, calib


def hysteresis_state(data, eta0, kappa):
    """One-state hysteresis eta(t) in [0, 1] driven by charge throughput (exact for ZOH current)."""
    I, t = data.I_raw, data.t
    eta = np.empty(t.size)
    eta[0] = eta0
    dt = np.diff(t)
    for i in range(1, t.size):
        if abs(I[i]) < 1e-3:
            eta[i] = eta[i - 1]
            continue
        u = 1.0 if I[i] > 0 else 0.0
        eta[i] = u + (eta[i - 1] - u) * np.exp(-kappa * abs(I[i]) * dt[i - 1] / (3600 * Q_REF))
    return eta


_CACHE = {}


def _simulate_set(args):
    """Worker: simulate one dataset; returns (key, V on the data grid with NaN where not covered, info)."""
    key, values, mesh = args
    try:
        if key not in _CACHE:
            _CACHE[key] = load_dataset(key)
        d = _CACHE[key]
        pv, calib = build_parameter_values(values)
        pv = set_state_from_full(pv, calib, start_from_full(d, key))
        res = simulate(d, parameter_values=pv, initial_soc=None, var_pts=mesh, verbose=False)
        mask, Vs = compare_on_data_grid(d, res)
        V = np.full(d.t.size, np.nan)
        V[mask] = Vs
        return key, V, res["termination"]
    except Exception as e:  # noqa: BLE001
        return key, None, f"failed: {type(e).__name__}: {e}"


# ------------------------------------------------------------------ evaluator
class Evaluator:
    def __init__(self, keys, mesh, kappa, pool):
        self.keys, self.mesh, self.kappa, self.pool = keys, mesh, kappa, pool
        self.data = {k: load_dataset(k) for k in keys}
        self.w = {k: 1.0 / np.sqrt(self.data[k].t.size) for k in keys}
        self.masks = {k: training_mask(self.data[k], k) for k in keys}
        self.masks_all = {k: training_mask(self.data[k], k, "all") for k in keys}
        self.set_kappa(kappa)
        self.n_eval = 0
        self.best = None
        self.t_start = time.perf_counter()
        self.use_dh = True
        self._cache = {}

    def set_kappa(self, kappa):
        self.kappa = kappa
        self.eta = {k: hysteresis_state(self.data[k], ETA0[k], kappa) for k in self.keys}

    def simulate_all(self, values, mesh=None):
        jobs = [(k, values, mesh if mesh is not None else self.mesh) for k in self.keys]
        return {k: (V, info) for k, V, info in self.pool.map(_simulate_set, jobs)}

    def fit_linear(self, sims, use_rc=True, use_dh=True, masks=None):
        """Best R_c, dh for given DFN voltages; returns (Rc, dh, residual vector, per-set metrics).
        masks: optional {key: bool array} restricting which samples are used."""
        A, b, pen = [], [], []
        for k in self.keys:
            d, V = self.data[k], sims[k][0]
            sel = masks[k] if masks is not None else np.ones(d.t.size, bool)
            w = 1.0 / np.sqrt(max(sel.sum(), 1))
            if V is None:
                pen.append(np.full(sel.sum(), PENALTY_V * w))
                continue
            ok = np.isfinite(V) & sel
            cols = [d.I_raw[ok] if use_rc else np.zeros(ok.sum()),
                    self.eta[k][ok] if use_dh else np.zeros(ok.sum())]
            A.append(np.column_stack(cols) * w)
            b.append((d.V[ok] - V[ok]) * w)
            miss = sel & ~np.isfinite(V)
            if miss.any():
                pen.append(np.full(miss.sum(), PENALTY_V * w))
        A = np.vstack(A) if A else np.zeros((0, 2))
        b = np.concatenate(b) if b else np.zeros(0)
        x = np.zeros(2)
        if A.shape[0] and (use_rc or use_dh):
            hi = [RC_MAX if use_rc else 1e-12, DH_MAX if use_dh else 1e-12]
            x = lsq_linear(A, b, bounds=([0.0, 0.0], hi)).x
        r = np.concatenate([A @ x - b] + pen) if (A.shape[0] or pen) else np.zeros(1)
        per = {}
        for k in self.keys:
            d, V = self.data[k], sims[k][0]
            if V is None:
                per[k] = {"RMSE_V": float("nan"), "MAE_V": float("nan"), "MaxAbs_V": float("nan"),
                          "MeanError_V": float("nan"), "covered": 0.0}
                continue
            sel = masks[k] if masks is not None else np.ones(d.t.size, bool)
            ok = np.isfinite(V) & sel
            e = V[ok] + x[0] * d.I_raw[ok] + x[1] * self.eta[k][ok] - d.V[ok]
            m = voltage_metrics(e, d.I_raw[ok])
            m["covered"] = float(ok.mean())
            per[k] = m
        return x[0], x[1], r, per

    def jacobian(self, z):
        """Forward differences with an ABSOLUTE step of JAC_STEP decades (scipy's relative step
        degenerates to ~1e-8 at z = 0, far below solver tolerance)."""
        r0 = self.residuals(z)
        J = np.empty((r0.size, z.size))
        for i in range(z.size):
            h = JAC_STEP if z[i] + JAC_STEP <= HI[i] else -JAC_STEP
            zi = np.array(z, dtype=float)
            zi[i] += h
            J[:, i] = (self.residuals(zi, tag=f"jac {PARAMS[i][0]}") - r0) / h
        return J

    def residuals(self, z, tag=""):
        key = tuple(np.round(np.asarray(z, dtype=float), 12))
        if key in self._cache:
            return self._cache[key]
        values = values_from_z(z)
        t0 = time.perf_counter()
        sims = self.simulate_all(values)
        rc, dh, r, per = self.fit_linear(sims, use_dh=self.use_dh, masks=self.masks)
        cost = float(np.sum(r ** 2) / len(self.keys))          # mean of per-set MSE
        self.n_eval += 1
        rec = {"eval": self.n_eval, "z": list(map(float, z)), "values": values, "R_c": rc, "dh": dh,
               "kappa": self.kappa, "cost": cost, "rms_mV": 1000 * np.sqrt(cost),
               "per_set_rmse_mV": {k: 1000 * v["RMSE_V"] for k, v in per.items()},
               "info": {k: sims[k][1] for k in self.keys},
               "eval_s": time.perf_counter() - t0, "elapsed_min": (time.perf_counter() - self.t_start) / 60}
        with open(HISTORY, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.best is None or cost < self.best["cost"]:
            self.best = rec
            BEST.write_text(json.dumps(rec, indent=2))
        print(f"  eval {self.n_eval:>3}{'' if not tag else ' (' + tag + ')':<12} rms {rec['rms_mV']:6.1f} mV  "
              + "  ".join(f"{p[0]}={values[p[0]]:.3g}" for p in PARAMS)
              + f"  R_c={1000 * rc:.2f} mOhm  dh={1000 * dh:.1f} mV  ({rec['eval_s']:.0f} s, "
              f"{rec['elapsed_min']:.1f} min)", flush=True)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = r
        return r


# ------------------------------------------------------------------ modes
def run_check(ev):
    print("\nCHECK MODE - Task 3 model, no DFN change: does a hysteresis term explain the offsets?")
    sims = ev.simulate_all(values_from_z(np.zeros(len(PARAMS))))
    failed = [k for k in ev.keys if sims[k][0] is None]
    for k in ev.keys:
        if sims[k][0] is None or not np.isfinite(sims[k][0]).all():
            print(f"  {k}: {sims[k][1]}")
    if failed:
        print(f"\n  STOP: simulation failed for {failed} - fix this before optimizing (see messages above)")
        return
    rows = []
    for label, kap, rc_on, dh_on in [("(a) no correction", 30, False, False),
                                     ("(b) + R_c", 30, True, False),
                                     ("(c) + R_c + hysteresis, kappa 10", 10, True, True),
                                     ("(c) + R_c + hysteresis, kappa 30", 30, True, True),
                                     ("(c) + R_c + hysteresis, kappa 100", 100, True, True)]:
        ev.set_kappa(kap)
        rc, dh, r, per = ev.fit_linear(sims, rc_on, dh_on)
        cost = np.sum(r ** 2) / len(ev.keys)
        rows.append((label, 1000 * np.sqrt(cost), rc, dh, per))
    print(f"\n  {'variant':<36}{'cost rms':>9}{'R_c':>8}{'dh':>7}" + "".join(f"{k:>7}" for k in ev.keys)
          + "   [mV, mOhm, mV, per-set RMSE mV]")
    for label, c, rc, dh, per in rows:
        print(f"  {label:<36}{c:>9.1f}{1000 * rc:>8.2f}{1000 * dh:>7.1f}"
              + "".join(f"{1000 * per[k]['RMSE_V']:>7.1f}" for k in ev.keys))
    print("\n  mean error per set with (b) vs best (c) [mV] (charge sets should move toward 0):")
    best_c = min(rows[2:], key=lambda r: r[1])
    print("   " + "".join(f"{k:>8}" for k in ev.keys))
    print("   " + "".join(f"{1000 * rows[1][4][k]['MeanError_V']:>8.1f}" for k in ev.keys) + "   (b)")
    print("   " + "".join(f"{1000 * best_c[4][k]['MeanError_V']:>8.1f}" for k in ev.keys) + f"   {best_c[0]}")
    (OUT / "check.json").write_text(json.dumps(
        [{"variant": r[0], "cost_rms_mV": r[1], "R_c_Ohm": r[2], "dh_V": r[3],
          "per_set": {k: v for k, v in r[4].items()}} for r in rows], indent=2, default=float))
    print(f"\nSaved: {OUT / 'check.json'}")


def phases(d, key):
    """Phase labels per sample: rest / CC / CV taper (charge) or rest / CC / end 5 % (discharge)."""
    I = d.I_raw
    lab = np.full(I.size, "rest", dtype=object)
    if key.startswith("C"):
        imax = I.max()
        lab[I > 0.95 * imax] = "CC"
        lab[(I > 0.05) & (I <= 0.95 * imax)] = "CV taper"
    else:
        imin = I.min()
        q = -d.ah_model()
        lab[I < 0.95 * imin] = "CC"
        lab[(I < 0.95 * imin) & (q > 0.95 * q.max())] = "end 5 %"
        lab[(I < -0.05) & (I >= 0.95 * imin)] = "taper"
    return lab


def run_diagnose(ev):
    print("\nDIAGNOSE MODE - Task 3 model: where is the training error, and is hysteresis real?")
    sims = ev.simulate_all(values_from_z(np.zeros(len(PARAMS))))
    if any(sims[k][0] is None for k in ev.keys):
        print("  STOP: a simulation failed: " + ", ".join(f"{k}: {sims[k][1]}" for k in ev.keys))
        return
    labs = {k: phases(ev.data[k], k) for k in ev.keys}
    print(f"\n  {'set':<4}{'phase':<10}{'share':>7}{'RMSE':>8}{'mean':>8}{'max+':>8}{'max-':>8}"
          f"{'I range [A]':>16}{'V_meas range':>16}{'V_sim range':>16}")
    for k in ev.keys:
        d, V = ev.data[k], sims[k][0]
        for ph in ["rest", "CC", "CV taper", "taper", "end 5 %"]:
            sel = (labs[k] == ph) & np.isfinite(V)
            if sel.sum() < 3:
                continue
            e = (V[sel] - d.V[sel]) * 1000
            print(f"  {k:<4}{ph:<10}{100 * sel.mean():>6.1f}%{np.sqrt(np.mean(e ** 2)):>8.1f}"
                  f"{e.mean():>8.1f}{e.max():>8.1f}{e.min():>8.1f}"
                  f"{d.I_raw[sel].min():>8.2f}..{d.I_raw[sel].max():<6.2f}"
                  f"{d.V[sel].min():>8.3f}..{d.V[sel].max():<6.3f}{V[sel].min():>8.3f}..{V[sel].max():<6.3f}")
    print("  (error = sim - meas [mV]; max+ / max- = largest positive / negative error)")

    cc = {k: labs[k] == "CC" for k in ev.keys}
    print("\n  R_c + hysteresis fitted on the CONSTANT-CURRENT parts only:")
    print(f"  {'variant':<36}{'rms':>7}{'R_c':>8}{'dh':>7}" + "".join(f"{k:>7}" for k in ev.keys))
    out = []
    for label, kap, rc_on, dh_on in [("(a) no correction", 30, False, False),
                                     ("(b) + R_c", 30, True, False),
                                     ("(c) + R_c + hysteresis, kappa 10", 10, True, True),
                                     ("(c) + R_c + hysteresis, kappa 30", 30, True, True),
                                     ("(c) + R_c + hysteresis, kappa 100", 100, True, True)]:
        ev.set_kappa(kap)
        rc, dh, r, per = ev.fit_linear(sims, rc_on, dh_on, masks=cc)
        c = 1000 * np.sqrt(np.sum(r ** 2) / len(ev.keys))
        out.append({"variant": label, "rms_mV": c, "R_c_mOhm": 1000 * rc, "dh_mV": 1000 * dh,
                    "mean_mV": {k: 1000 * per[k]["MeanError_V"] for k in ev.keys}})
        print(f"  {label:<36}{c:>7.1f}{1000 * rc:>8.2f}{1000 * dh:>7.1f}"
              + "".join(f"{1000 * per[k]['RMSE_V']:>7.1f}" for k in ev.keys))
    print("  mean error on CC parts [mV]:   " + "".join(f"{k:>8}" for k in ev.keys))
    for o in [out[1], min(out[2:], key=lambda o: o["rms_mV"])]:
        print(f"  {o['variant']:<31}" + "".join(f"{o['mean_mV'][k]:>8.1f}" for k in ev.keys))
    ev.set_kappa(30)
    (OUT / "diagnose.json").write_text(json.dumps(out, indent=2))

    # plot: each set, measured vs Task 3 model, phases shaded
    fig, ax = plt.subplots(len(ev.keys), 1, figsize=(13, 2.8 * len(ev.keys)), squeeze=False)
    col = {"CC": "tab:blue", "CV taper": "tab:orange", "taper": "tab:orange", "end 5 %": "tab:red"}
    for i, k in enumerate(ev.keys):
        d, V = ev.data[k], sims[k][0]
        a = ax[i, 0]
        a.plot(d.t / 60, d.V, "k", lw=1, label="Measured")
        a.plot(d.t / 60, V, "--", color="tab:orange", lw=1, label="Task 3 model")
        for ph, c in col.items():
            sel = labs[k] == ph
            if sel.any():
                a.fill_between(d.t / 60, d.V.min() - 0.05, d.V.max() + 0.05, where=sel, color=c,
                               alpha=0.08, step="mid", label=ph)
        a.set_ylim(min(d.V.min(), np.nanmin(V)) - 0.05, max(d.V.max(), np.nanmax(V)) + 0.05)
        a.set_ylabel("Voltage [V]")
        a.set_title(DATASETS[k])
        a.legend(fontsize=7, ncol=5)
        a.grid(alpha=0.3)
    ax[-1, 0].set_xlabel("Time [min]")
    fig.tight_layout()
    fig.savefig(OUT / "step4_diagnose.png", dpi=130)
    print(f"\nSaved: {OUT / 'diagnose.json'}, step4_diagnose.png")


def run_optimization(ev, args):
    z0 = np.zeros(len(PARAMS))
    if args.resume and BEST.exists():
        z0 = np.clip(np.array(json.loads(BEST.read_text())["z"]), LO + 1e-9, HI - 1e-9)
        print(f"Resuming from best saved point: {values_from_z(z0)}")
    elif HISTORY.exists():
        HISTORY.unlink()
    print(f"\nOPTIMIZATION: max {args.max_nfev} iterations, {len(ev.keys)} sets in parallel, "
          f"Jacobian step {JAC_STEP} dex, hysteresis term {'ON (kappa %g)' % ev.kappa if ev.use_dh else 'OFF'}")
    print("  samples used in the cost: " + ", ".join(
        f"{k} {100 * ev.masks[k].mean():.0f} %" for k in ev.keys)
        + f"  (charge tests: first {EXCLUDE_AH} Ah = initial relaxation history excluded)")
    print("  bounds: " + ", ".join(f"{p[0]} [{p[4]:.3g}, {p[5]:.3g}]" for p in PARAMS)
          + f", R_c [0, {1000 * RC_MAX:.0f}] mOhm, dh [0, {1000 * DH_MAX:.0f}] mV")
    res = least_squares(ev.residuals, z0, bounds=(LO, HI), method="trf", jac=ev.jacobian,
                        x_scale=1.0, max_nfev=args.max_nfev,
                        xtol=1e-3, ftol=1e-4, gtol=1e-6, verbose=0)
    print(f"\nOptimizer finished: {res.message} (status {res.status}, {ev.n_eval} evaluations)")
    return res


def finalize(ev, res, args, pool):
    z = res.x
    values = values_from_z(z)
    print("\nFINAL CHECK with the default (fine) mesh ...")
    sims_fine = ev.simulate_all(values, mesh={})
    rc, dh, r, per = ev.fit_linear(sims_fine, use_dh=ev.use_dh, masks=ev.masks)
    task3 = {"k_n": 1.0, "k_p": 1.0, "D_n": ALL_PARAMS["D_n"][2], "D_p": 1.0}   # Task 3 model
    sims0 = ev.simulate_all(task3, mesh={})
    _, _, _, per0 = ev.fit_linear(sims0, use_rc=False, use_dh=False, masks=ev.masks)

    def full_metrics(sims, rc_, dh_):
        out = {}
        for k in ev.keys:
            d, V = ev.data[k], sims[k][0]
            ok = np.isfinite(V)
            out[k] = voltage_metrics(V[ok] + rc_ * d.I_raw[ok] + dh_ * ev.eta[k][ok] - d.V[ok],
                                     d.I_raw[ok])
        return out
    per_full = full_metrics(sims_fine, rc, dh)
    per0_full = full_metrics(sims0, 0.0, 0.0)
    _, _, _, per_allw = ev.fit_linear(sims_fine, use_dh=ev.use_dh, masks=ev.masks_all)

    # local uncertainty / correlation from the Jacobian (log10 units)
    J = res.jac
    m, n = J.shape
    s2 = 2 * res.cost / max(m - n, 1)
    unc, corr = None, None
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        unc = np.sqrt(np.diag(cov))
        corr = cov / np.outer(unc, unc)
    except np.linalg.LinAlgError:
        pass

    at_bound = [p[0] for i, p in enumerate(PARAMS) if z[i] <= LO[i] + 0.005 or z[i] >= HI[i] - 0.005]
    print("\nOPTIMIZED PARAMETERS")
    for i, p in enumerate(PARAMS):
        if unc is None:
            u = ""
        elif p[1] == "linear":
            u = f" (+/- {LIN_SCALE * unc[i]:.2f} decades, formal)"
        else:
            u = f" (+/- {unc[i]:.2f} dex, formal)"
        unit = {"factor": " x Prada2013", "absolute": " m2/s", "linear": " decades"}[p[1]]
        print(f"  {p[0]:<5} {values[p[0]]:.4g}{unit}{u}"
              f"{'  AT BOUND' if p[0] in at_bound else ''}")
    print(f"  R_c  {1000 * rc:.2f} mOhm{'  AT BOUND' if rc >= 0.999 * RC_MAX else ''}")
    if ev.use_dh:
        print(f"  dh   {1000 * dh:.1f} mV (kappa {ev.kappa}){'  AT BOUND' if dh >= 0.999 * DH_MAX else ''}")
    else:
        print("  dh   not fitted (hysteresis term off)")
    if corr is not None:
        print("\n  correlation matrix (log10 parameters):")
        print("         " + "".join(f"{p[0]:>7}" for p in PARAMS))
        for i, p in enumerate(PARAMS):
            print(f"  {p[0]:<6}" + "".join(f"{corr[i, j]:>7.2f}" for j in range(n)))

    print("\nTRAINING ERROR, default mesh, training window [mV]   (Task 3 model -> optimized)")
    print(f"  {'set':<4}{'RMSE':>14}{'MAE':>14}{'max':>16}{'mean':>16}")
    for k in ev.keys:
        a, b = per0[k], per[k]
        print(f"  {k:<4}{1000 * a['RMSE_V']:>6.1f} ->{1000 * b['RMSE_V']:>6.1f}"
              f"{1000 * a['MAE_V']:>6.1f} ->{1000 * b['MAE_V']:>6.1f}"
              f"{1000 * a['MaxAbs_V']:>7.1f} ->{1000 * b['MaxAbs_V']:>7.1f}"
              f"{1000 * a['MeanError_V']:>7.1f} ->{1000 * b['MeanError_V']:>7.1f}")
    hppc_split = None
    if "HPPC" in ev.keys:
        d = ev.data["HPPC"]
        tr, ho = hppc_split_masks(d)
        hs = {}
        for lab, sims_, rc_ in [("task3", sims0, 0.0), ("optimized", sims_fine, rc)]:
            V = sims_["HPPC"][0]
            if V is None:
                hs[lab] = None
                continue
            okv = np.isfinite(V)
            e = V + rc_ * d.I_raw - d.V
            hs[lab] = {"train_rmse_mV": float(1000 * np.sqrt(np.mean(e[tr & okv] ** 2))),
                       "heldout_rmse_mV": float(1000 * np.sqrt(np.mean(e[ho & okv] ** 2))),
                       "coverage": float(okv.mean())}
        hppc_split = {"training": "dynamic samples of odd SOC levels", "held_out": "even SOC levels", **hs}
        print("\nHPPC SPLIT (default mesh): RMSE on training levels / HELD-OUT levels [mV]")
        for lab in ["task3", "optimized"]:
            if hs.get(lab):
                print(f"  {lab:<10} train {hs[lab]['train_rmse_mV']:6.1f}   held-out {hs[lab]['heldout_rmse_mV']:6.1f}"
                      f"   (coverage {100 * hs[lab]['coverage']:.1f} %)")
    mean0 = np.nanmean([per0[k]["RMSE_V"] for k in ev.keys])
    mean1 = np.nanmean([per[k]["RMSE_V"] for k in ev.keys])
    print(f"  mean RMSE {1000 * mean0:.1f} -> {1000 * mean1:.1f} mV  (training window)")
    if COST_WINDOW == "cc":
        print("\n  incl. CV phase (all except initial window), optimized: RMSE [mV]")
        for k in ev.keys:
            print(f"  {k:<4}{1000 * per_allw[k]['RMSE_V']:>6.1f}")
    print("\n  FULL curves incl. the excluded initial window: RMSE [mV]")
    for k in ev.keys:
        print(f"  {k:<4}{1000 * per0_full[k]['RMSE_V']:>6.1f} ->{1000 * per_full[k]['RMSE_V']:>6.1f}")

    calib = load_ocv_calibration()
    pv, _ = build_parameter_values(values, calib)
    absolute = {}
    for name, kind, pname, *_ in PARAMS:
        if kind == "linear":
            absolute[pname] = values[name]
            continue
        try:
            v = pv[pname]
            absolute[pname] = (f"function (Prada2013 based, x {values[name]:.4g})" if callable(v) else float(v))
        except Exception as e:  # noqa: BLE001
            absolute[pname] = f"could not read ({type(e).__name__})"
    result = {
        "description": "Optimized parameters for A123 ANR26650M1B: Prada2013 + Step 3c thermodynamics "
                       "+ Step 4 optimization (training: Charge 1C-4C, Discharge 1C)",
        "base_parameter_set": "Prada2013",
        "ocv_calibration_file": "results/step3c/ocv_calibration.json",
        "optimized": {p[0]: values[p[0]] for p in PARAMS},
        "optimized_kind": {p[0]: p[1] for p in PARAMS},
        "dp_shape": ({"a_top": values.get("a_top"), "a_bot": values.get("a_bot", 0.0), "width": DP_WIDTH,
                      "form": "D_p(y) = D_p,0 * 10^-(a_top exp(-y/w) + a_bot exp(-(1-y)/w))"}
                     if "a_top" in values else None),
        "dn_shape": ({"b_bot": values.get("b_bot"), "width": DP_WIDTH,
                      "form": "D_n(x) = D_n,0 * 10^-(b_bot exp(-x/w))"} if "b_bot" in values else None),
        "pybamm_parameter_names": {p[0]: p[2] for p in PARAMS},
        "pybamm_values": absolute,
        "contact_resistance_Ohm": rc, "hysteresis_fitted": ev.use_dh,
        "hysteresis_dh_V": dh, "hysteresis_kappa": ev.kappa,
        "hysteresis_Q_ref_Ah": Q_REF,
        "uncertainty_dex": None if unc is None else {
            p[0]: float(unc[i] * (LIN_SCALE if p[1] == "linear" else 1.0)) for i, p in enumerate(PARAMS)},
        "uncertainty_units": "factor / absolute parameters: log10 decades; shape parameters: decades of diffusivity reduction",
        "uncertainty_note": "formal Jacobian-based (assumes no model error) - lower bound of the true uncertainty",
        "correlation": None if corr is None else corr.tolist(),
        "at_bound": at_bound,
        "training_rmse_mV_task3": {k: 1000 * per0[k]["RMSE_V"] for k in ev.keys},
        "training_rmse_mV_optimized": {k: 1000 * per[k]["RMSE_V"] for k in ev.keys},
        "full_curve_rmse_mV_task3": {k: 1000 * per0_full[k]["RMSE_V"] for k in ev.keys},
        "full_curve_rmse_mV_optimized": {k: 1000 * per_full[k]["RMSE_V"] for k in ev.keys},
        "cost_window": COST_WINDOW,
        "training_window": (f"charge tests from {EXCLUDE_AH} Ah of charge onward"
                            + (", constant-current part only (CV phase excluded)" if COST_WINDOW == "cc" else "")
                            + "; D1 complete"),
        "rmse_mV_all_window_optimized": {k: 1000 * per_allw[k]["RMSE_V"] for k in ev.keys},
        "hppc_split": hppc_split,
        "optimizer": {"method": f"scipy least_squares trf, forward-difference Jacobian with absolute "
                                f"step {JAC_STEP} dex", "message": res.message,
                      "status": int(res.status), "evaluations": ev.n_eval, "mesh": "coarse " + str(COARSE_MESH)},
    }
    (OUT / "optimized_parameters.json").write_text(json.dumps(result, indent=2, default=str))

    # plots: training fits + convergence
    fig, ax = plt.subplots(len(ev.keys), 2, figsize=(14, 3.0 * len(ev.keys)), squeeze=False)
    for i, k in enumerate(ev.keys):
        d = ev.data[k]
        V1 = sims_fine[k][0] + rc * d.I_raw + dh * ev.eta[k]
        V0 = sims0[k][0]
        ax[i, 0].plot(d.t / 60, d.V, "k", lw=1, label="Measured")
        ax[i, 0].plot(d.t / 60, V0, ":", color="tab:gray", lw=1, label="Task 3 model")
        ax[i, 0].plot(d.t / 60, V1, "--", color="tab:orange", lw=1, label="Optimized")
        ax[i, 0].set_title(f"{k}: {DATASETS[k]} (RMSE {1000 * per0[k]['RMSE_V']:.1f} -> "
                           f"{1000 * per[k]['RMSE_V']:.1f} mV, training window)")
        if (~ev.masks[k]).any():
            ax[i, 0].axvspan(0, d.t[~ev.masks[k]][-1] / 60, color="tab:gray", alpha=0.15,
                             label="excluded (initial history)")
        ax[i, 0].set_ylabel("Voltage [V]")
        ax[i, 0].legend(fontsize=7)
        ax[i, 1].plot(d.t / 60, (V0 - d.V) * 1000, ":", color="tab:gray", lw=0.8, label="Task 3")
        ax[i, 1].plot(d.t / 60, (V1 - d.V) * 1000, color="tab:red", lw=0.8, label="Optimized")
        ax[i, 1].axhline(0, color="k", lw=0.5)
        ax[i, 1].set_ylabel("Error [mV]")
        ax[i, 1].legend(fontsize=7)
        for a in ax[i]:
            a.grid(alpha=0.3)
    ax[-1, 0].set_xlabel("Time [min]")
    ax[-1, 1].set_xlabel("Time [min]")
    fig.suptitle("Task 4 - training data: Task 3 model vs optimized", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "step4_training_fit.png", dpi=130)

    hist = [json.loads(line) for line in open(HISTORY)]
    fig, a = plt.subplots(figsize=(8, 4.5))
    a.plot([h["eval"] for h in hist], [h["rms_mV"] for h in hist], "o-", ms=3)
    a.set_xlabel("Evaluation")
    a.set_ylabel("Training cost (rms over sets) [mV]")
    a.set_title("Task 4 - optimization convergence")
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "step4_convergence.png", dpi=130)
    print(f"\nSaved: {OUT / 'optimized_parameters.json'}, step4_training_fit.png, step4_convergence.png, history.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="one evaluation: test the hysteresis term")
    ap.add_argument("--diagnose", action="store_true", help="error by phase + hysteresis test on CC parts")
    ap.add_argument("--kappa", type=float, default=30.0)
    ap.add_argument("--max-nfev", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cost-window", choices=["all", "cc"], default="all",
                    help="'cc': fit charge tests on their constant-current part only (CV phase excluded)")
    ap.add_argument("--tag", default="", help="results go to results/step4<_tag>/")
    ap.add_argument("--dp-shape", action="store_true",
                    help="fit k_n, D_n, D_p, a_top, b_bot (state-dependent LFP / graphite diffusivity; k_p fixed)")
    ap.add_argument("--init", default="", help="initial values, e.g. k_n=0.6,D_n=3e-14,b_bot=2")
    ap.add_argument("--fit", default="", help="parameters to fit, e.g. k_n,D_n,D_p,b_bot (overrides --dp-shape)")
    ap.add_argument("--bounds", default="", help="override bounds, e.g. D_n=3e-14:3e-13,b_bot=0:3")
    ap.add_argument("--hppc-split", action="store_true",
                    help="add HPPC pulse sets of odd SOC levels to the training data; even levels held out")
    ap.add_argument("--hysteresis", action="store_true",
                    help="also fit the hysteresis term (off by default: --diagnose found no support)")
    ap.add_argument("--workers", type=int, default=min(len(TRAINING_KEYS), os.cpu_count() or 1))
    args = ap.parse_args()
    global OUT, HISTORY, BEST, COST_WINDOW, PARAMS, LO, HI
    init = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.init.split(",") if "=" in kv}
    for kv in [b for b in args.bounds.split(",") if "=" in b]:
        n, rng = kv.split("=")
        lo_v, hi_v = (float(x) for x in rng.split(":"))
        kind, pname, ini, _, _ = ALL_PARAMS[n]
        ALL_PARAMS[n] = (kind, pname, ini, lo_v, hi_v)
    for n, v in init.items():                       # keep the start strictly inside the bounds
        kind, pname, ini, lo_v, hi_v = ALL_PARAMS[n]
        if not lo_v < v < hi_v:
            raise SystemExit(f"--init {n}={v} is outside its bounds [{lo_v}, {hi_v}]")
    names = [n.strip() for n in args.fit.split(",") if n.strip()] if args.fit else \
        (SHAPE_SET if args.dp_shape else BASE_SET)
    unknown = [n for n in names if n not in ALL_PARAMS]
    if unknown:
        raise SystemExit(f"unknown parameter(s) {unknown}; choose from {list(ALL_PARAMS)}")
    PARAMS = make_params(names, init)
    LO, HI = bounds_for(PARAMS)
    OUT = RESULTS_DIR / ("step4" + (f"_{args.tag}" if args.tag else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORY, BEST = OUT / "history.jsonl", OUT / "best_so_far.json"
    COST_WINDOW = args.cost_window

    print("=" * 66)
    print("TASK 4 - PARAMETER OPTIMIZATION (training: Charge 1C-4C, Discharge 1C)")
    print(f"Cost window: {COST_WINDOW}; results -> {OUT}")
    print("Fitted parameters: " + ", ".join(f"{p[0]} (start {p[3]:.3g})" for p in PARAMS)
          + ("; k_p fixed at Prada2013" if "k_p" not in names else ""))
    print("=" * 66)
    print(f"Parallel workers: {args.workers}; mesh during optimization: coarse {COARSE_MESH}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        keys = TRAINING_KEYS + (["HPPC"] if args.hppc_split else [])
        ev = Evaluator(keys, COARSE_MESH, args.kappa, pool)
        if args.hppc_split:
            tr, ho = hppc_split_masks(ev.data["HPPC"])
            lv, _ = hppc_levels(ev.data["HPPC"])
            print(f"HPPC split: {lv.max()} SOC levels; training = odd levels ({tr.sum()} samples), "
                  f"held out = even levels ({ho.sum()} samples)")
        ev.use_dh = args.hysteresis
        if args.check:
            run_check(ev)
            return
        if args.diagnose:
            run_diagnose(ev)
            return
        res = run_optimization(ev, args)
        finalize(ev, res, args, pool)


if __name__ == "__main__":
    main()