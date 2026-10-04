"""
ocv_tools.py - Shared OCV / OCP utilities for Step 3 (used by step3b and step3c).

- fit_relaxation   : exponential relaxation fit of a rest, extrapolated to equilibrium
- hppc_rest_points : OCV points from the full HPPC test (start + relaxation-corrected rests)
- end_rest_after_charge : OCV at full charge after a charge test (charge branch)
- pseudo_ocv_1c    : average of the 1C charge and 1C discharge curves vs SOC
- eval_ocp         : evaluate a parameter set's electrode OCP function on numpy arrays
- stoich_window    : electrode stoichiometries at SOC 0 % and 100 % for a parameter set
"""
import inspect

import numpy as np
import pybamm

from model_setup import apply_initial_state, electrode_balance

REST_A = 0.05


def _segments(mask):
    m = np.concatenate(([False], mask, [False])).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def fit_relaxation(t, v, skip_s=120.0, min_points=8):
    """
    Fit V(t) = V_inf + B * exp(-(t - t0) / tau) to a rest tail (first `skip_s`
    seconds skipped to avoid the fast ohmic/charge-transfer transient).
    Returns dict: V_inf, tau_s, ok (fit converged and physically sensible).
    """
    from scipy.optimize import curve_fit

    sel = t >= t[0] + skip_s
    tt, vv = t[sel] - t[0], v[sel]
    out = {"V_inf": float(v[-1]), "tau_s": float("nan"), "ok": False}
    if tt.size < min_points or np.ptp(vv) < 1e-4:
        out["ok"] = np.ptp(vv) < 1e-4 and tt.size >= min_points   # already flat
        return out
    span = tt[-1] - tt[0]
    b0 = vv[0] - vv[-1]
    try:
        p, _ = curve_fit(lambda x, vinf, b, tau: vinf + b * np.exp(-x / tau),
                         tt, vv, p0=[vv[-1], b0, span / 3],
                         bounds=([vv[-1] - 0.1, -1.0, 10.0], [vv[-1] + 0.1, 1.0, 5 * span]),
                         maxfev=20000)
    except Exception:  # noqa: BLE001
        return out
    vinf, _, tau = p
    delta = vinf - v[-1]
    out.update({"V_inf": float(vinf), "tau_s": float(tau)})
    # sensible: time constant not stuck at the bound, extrapolation small
    out["ok"] = bool(tau < 4.5 * span and abs(delta) < 0.030)
    return out


def hppc_rest_points(data, min_rest_s=600.0, slope_window_s=300.0):
    """
    OCV points from the full HPPC test.
      - start point (SOC 1): first sample, rested fully charged cell before the test
      - end-of-rest points (rest >= min_rest_s), each with an exponential
        relaxation fit extrapolated to equilibrium (V_inf)
    SOC from measured charge throughput (backward ZOH): SOC = 1 + q / q_total.
    'V_ocv' = V_inf when the relaxation fit is valid, else the end-of-rest voltage.
    """
    q = data.ah_model()
    q_total = -q[-1]
    rest = np.abs(data.I_raw) < REST_A
    pts = [{"source": "HPPC start", "t_s": 0.0, "soc": 1.0, "Ah_discharged": 0.0,
            "V_end": float(data.V[0]), "V_inf": float(data.V[0]), "V_ocv": float(data.V[0]),
            "relax_correction_mV": 0.0, "tau_s": float("nan"), "rest_s": float("nan"),
            "slope_mV_per_h": float("nan"), "fit_ok": False}]
    for a, b in _segments(rest):
        dur = data.t[b] - data.t[a]
        if dur < min_rest_s:
            continue
        tseg, vseg = data.t[a:b + 1], data.V[a:b + 1]
        v_ref = np.interp(data.t[b] - slope_window_s, tseg, vseg)
        slope = (data.V[b] - v_ref) / slope_window_s * 3600 * 1000
        fit = fit_relaxation(tseg, vseg)
        v_ocv = fit["V_inf"] if fit["ok"] else float(data.V[b])
        pts.append({
            "source": "HPPC rest", "t_s": float(data.t[b]),
            "soc": float(1.0 + q[b] / q_total), "Ah_discharged": float(-q[b]),
            "V_end": float(data.V[b]), "V_inf": fit["V_inf"], "V_ocv": float(v_ocv),
            "relax_correction_mV": float(1000 * (v_ocv - data.V[b])),
            "tau_s": fit["tau_s"], "rest_s": float(dur),
            "slope_mV_per_h": float(slope), "fit_ok": fit["ok"],
        })
    return pts, float(q_total)


def end_rest_after_charge(data, min_rest_s=600.0, rest_a=0.005, pinned_mv=2.0):
    """
    Final rest of a charge test: OCV at full charge on the charge branch.
    A valid rest needs |I| < rest_a (5 mA) for >= min_rest_s AND a relaxing voltage.
    A segment held at the charge limit (voltage range < pinned_mv) is the CV phase
    of a CC-CV charge, not a rest.
    Returns dict with 'valid' and 'reason' (plus values when valid).
    """
    tail = data.t >= data.t[-1] - min_rest_s
    i_tail = np.abs(data.I_raw[tail])
    v_tail = data.V[tail]
    info = {"valid": False, "final_current_A": float(data.I_raw[-1]),
            "tail_current_mean_A": float(i_tail.mean()),
            "tail_V_range_mV": float(1000 * np.ptp(v_tail)), "V_end": float(data.V[-1])}
    if i_tail.max() >= rest_a:
        info["reason"] = f"current not zero at the end (|I| up to {1000 * i_tail.max():.1f} mA)"
        return info
    if 1000 * np.ptp(v_tail) < pinned_mv and data.V[-1] > 3.55:
        info["reason"] = "voltage pinned at the charge limit (CV phase), not a rest"
        return info
    rest = np.abs(data.I_raw) < rest_a
    segs = _segments(rest)
    a, b = segs[-1]
    fit = fit_relaxation(data.t[a:b + 1], data.V[a:b + 1])
    info.update({"valid": True, "reason": "true rest", "rest_s": float(data.t[b] - data.t[a]),
                 "V_inf": fit["V_inf"], "fit_ok": fit["ok"]})
    return info


def pseudo_ocv_1c(charge, discharge, n=200):
    """
    1C pseudo-OCV: average of the CC part of the 1C charge and the 1C discharge,
    each on its own SOC scale (charge: SOC = q_in / Q_charge_total, starting empty;
    discharge: SOC = 1 - q_out / Q_discharge_total, starting full).
    Returns dict of arrays on a common SOC grid (overlap of both curves).
    """
    # charge: constant-current part only (current within 5 % of its maximum)
    qc = charge.ah_model()                       # > 0, charge in
    Ic = charge.I_raw
    cc = Ic > 0.95 * Ic.max()
    soc_c = qc / qc[-1]
    soc_c, v_c = soc_c[cc], charge.V[cc]

    # discharge: loaded part only
    qd = -discharge.ah_model()                   # > 0, charge out
    Id = discharge.I_raw
    ld = Id < 0.95 * Id.min()
    soc_d = 1.0 - qd / qd[-1]
    soc_d, v_d = soc_d[ld], discharge.V[ld]

    o_c = np.argsort(soc_c)
    o_d = np.argsort(soc_d)
    lo = max(soc_c.min(), soc_d.min())
    hi = min(soc_c.max(), soc_d.max())
    grid = np.linspace(lo, hi, n)
    vc = np.interp(grid, soc_c[o_c], v_c[o_c])
    vd = np.interp(grid, soc_d[o_d], v_d[o_d])
    return {"soc": grid, "V_charge": vc, "V_discharge": vd,
            "V_avg": 0.5 * (vc + vd), "half_gap": 0.5 * (vc - vd),
            "Q_charge_Ah": float(qc[-1]), "Q_discharge_Ah": float(qd[-1])}


def eval_ocp(pv, electrode, sto, T=298.15):
    """Evaluate '<Electrode> electrode OCP [V]' of a parameter set on an array of stoichiometries."""
    f = pv[f"{electrode.capitalize()} electrode OCP [V]"]
    sto = np.atleast_1d(np.asarray(sto, dtype=float))
    if not callable(f):                          # tabulated: ("name", (x, y))
        x, y = f[1]
        return np.interp(sto, np.asarray(x).ravel(), np.asarray(y).ravel())
    try:
        n_args = len(inspect.signature(f).parameters)
    except (TypeError, ValueError):
        n_args = 1
    out = np.empty_like(sto)
    for i, s in enumerate(sto):
        args = [pybamm.Scalar(s)] + ([pybamm.Scalar(T)] if n_args >= 2 else [])
        v = f(*args)
        out[i] = float(v.evaluate()) if hasattr(v, "evaluate") else float(v)
    return out


def stoich_window(pv):
    """Electrode stoichiometries at SOC 1 and SOC 0 (parameter set's own voltage limits)."""
    pv1, ok1 = apply_initial_state(pv.copy(), 1.0)
    pv0, ok0 = apply_initial_state(pv.copy(), 0.0)
    if not (ok1 and ok0):
        raise RuntimeError("PyBaMM initial-state API not available")
    b1, b0 = electrode_balance(pv1), electrode_balance(pv0)
    w = {"x100": b1["sto_n_init"], "y100": b1["sto_p_init"],
         "x0": b0["sto_n_init"], "y0": b0["sto_p_init"],
         "Q_n_Ah": b1["Q_n_Ah"], "Q_p_Ah": b1["Q_p_Ah"]}
    w["capacity_Ah"] = (w["x100"] - w["x0"]) * w["Q_n_Ah"]
    return w


def full_cell_ocv(pv, window, soc):
    """Full-cell OCV U_p(y) - U_n(x) along the parameter set's stoichiometry window."""
    soc = np.asarray(soc, dtype=float)
    x = window["x0"] + soc * (window["x100"] - window["x0"])
    y = window["y0"] + soc * (window["y100"] - window["y0"])
    return eval_ocp(pv, "positive", y) - eval_ocp(pv, "negative", x)