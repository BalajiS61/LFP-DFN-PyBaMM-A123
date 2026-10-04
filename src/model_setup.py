"""
model_setup.py - Task 1: DFN model and parameter set (shared by all steps).

Model         : Doyle-Fuller-Newman (pybamm.lithium_ion.DFN), isothermal
                (test temperature stayed 22.8-24.6 degC, so thermal effects ignored)
Parameter set : Prada2013 - LFP/graphite, parametrised on an A123 26650 cell
                (ANR26650M1A, 2.3 Ah), the predecessor of the ANR26650M1B (2.5 Ah)
Solver        : IDAKLU (fast, adaptive, suited to stiff DAE systems like the DFN)

Current input : measured current as a step-shaped Interpolant (backward
                zero-order hold, see data_loader.py). Breakpoints are placed at
                every current step so the adaptive solver never skips a pulse.

Initial state : set FIRST with the parameter set's own voltage limits, so that
                SOC 0/100 % keep their physical meaning (or taken directly from the
                parameter values when initial_soc=None, as for the calibrated model).
                Only AFTER that are the voltage limits widened (diagnostic limits).
"""
import time

import numpy as np
import pybamm

PARAM_SET = "Prada2013"
FARADAY = 96485.33212  # C/mol

# Wide "diagnostic" voltage limits: the simulation should not be stopped by a
# voltage event just because the un-tuned model over/undershoots during a pulse.
# They are applied AFTER the initial state is set, so they do not change the
# definition of SOC.
DIAG_V_MIN, DIAG_V_MAX = 1.0, 4.5


def base_parameter_values():
    """Prada2013 parameter values (fresh copy)."""
    return pybamm.ParameterValues(PARAM_SET)


def build_model(options=None):
    """Isothermal DFN model (options e.g. {"contact resistance": "true"})."""
    return pybamm.lithium_ion.DFN(options=options) if options else pybamm.lithium_ion.DFN()


def _p(pv, key, default=None):
    try:
        return float(pv[key])
    except Exception:  # noqa: BLE001
        if default is None:
            raise
        return default


def electrode_balance(pv):
    """
    Electrode capacities and initial stoichiometries from the parameter values.
      Q = F * c_max * eps_active * thickness * electrode area * n_parallel / 3600
    Returns a dict (capacities in Ah).
    """
    area = _p(pv, "Electrode width [m]") * _p(pv, "Electrode height [m]")
    area *= _p(pv, "Number of electrodes connected in parallel to make a cell", 1.0)
    out = {}
    for e, short in [("negative", "n"), ("positive", "p")]:
        cmax = _p(pv, f"Maximum concentration in {e} electrode [mol.m-3]")
        eps = _p(pv, f"{e.capitalize()} electrode active material volume fraction")
        L = _p(pv, f"{e.capitalize()} electrode thickness [m]")
        Q = FARADAY * cmax * eps * L * area / 3600
        sto0 = _p(pv, f"Initial concentration in {e} electrode [mol.m-3]") / cmax
        out[f"Q_{short}_Ah"] = Q
        out[f"sto_{short}_init"] = sto0
    # charge that can be discharged before the anode is empty / the cathode is full
    out["Ah_until_anode_empty"] = out["sto_n_init"] * out["Q_n_Ah"]
    out["Ah_until_cathode_full"] = (1 - out["sto_p_init"]) * out["Q_p_Ah"]
    out["max_dischargeable_Ah"] = min(out["Ah_until_anode_empty"], out["Ah_until_cathode_full"])
    out["limiting_electrode"] = ("negative (anode empties)"
                                 if out["Ah_until_anode_empty"] < out["Ah_until_cathode_full"]
                                 else "positive (cathode fills)")
    out["nominal_capacity_Ah"] = _p(pv, "Nominal cell capacity [A.h]", float("nan"))
    return out


def apply_initial_state(pv, initial_soc):
    """
    Set initial electrode concentrations for `initial_soc` (number, or 'x.xxx V')
    using the parameter set's OWN voltage limits. Works across PyBaMM versions.
    Returns (pv, applied: bool).
    """
    for name in ("set_initial_state", "set_initial_stoichiometries"):
        f = getattr(pv, name, None)
        if f is None:
            continue
        out = f(initial_soc)
        if isinstance(out, pybamm.ParameterValues):
            pv = out
        return pv, True
    return pv, False


def current_breakpoints(tp, Ip, threshold=0.5):
    """Times where the step current changes by more than `threshold` A."""
    jumps = np.where(np.abs(np.diff(Ip)) > threshold)[0]
    bp = np.concatenate(([tp[0]], tp[jumps], tp[jumps + 1], [tp[-1]]))
    return np.unique(bp)


# Mesh used during optimization (coarser than the PyBaMM default, checked in step4a)
COARSE_MESH = {"x_n": 10, "x_s": 6, "x_p": 10, "r_n": 12, "r_p": 12}


def simulate(data, parameter_values=None, initial_soc=1.0,
             v_limits=(DIAG_V_MIN, DIAG_V_MAX), rtol=1e-6, atol=1e-6, verbose=True,
             var_pts=None, model_options=None):
    """
    Run the DFN with the measured current of `data` (a CycleData object).

    Returns dict with: sim, sol, t_sim, V_sim, initial_soc_used, balance,
                       runtime_s, stopped_early
    """
    pv = (parameter_values or base_parameter_values()).copy()

    # 1) initial state: either already in the parameter values (initial_soc=None,
    #    e.g. set from the calibrated lithium line) or from an SOC / voltage using
    #    the parameter set's own voltage limits
    if initial_soc is None:
        applied = True
        balance = electrode_balance(pv)
        label = "from parameter values"
    else:
        pv, applied = apply_initial_state(pv, initial_soc)
        balance = electrode_balance(pv) if applied else None
        label = f"'{initial_soc}'"
    if verbose and balance:
        print(f"  initial state {label}: anode sto {balance['sto_n_init']:.4f}, "
              f"cathode sto {balance['sto_p_init']:.4f}")
        print(f"  electrode capacity: Q_n {balance['Q_n_Ah']:.3f} Ah, Q_p {balance['Q_p_Ah']:.3f} Ah")
        print(f"  dischargeable from this state: {balance['max_dischargeable_Ah']:.3f} Ah "
              f"(limited by {balance['limiting_electrode']})")

    # 2) measured current + wide diagnostic limits (events only)
    tp, Ip = data.current_profile()
    pv.update({
        "Current function [A]": pybamm.Interpolant(tp, Ip, pybamm.t, name="measured_current"),
        "Lower voltage cut-off [V]": v_limits[0],
        "Upper voltage cut-off [V]": v_limits[1],
    })

    model = build_model(model_options)
    solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
    sim_kwargs = {"parameter_values": pv, "solver": solver}
    if var_pts:
        sim_kwargs["var_pts"] = var_pts
    sim = pybamm.Simulation(model, **sim_kwargs)

    t_eval = current_breakpoints(tp, Ip)
    t_interp = data.t

    t0 = time.perf_counter()
    if applied:
        sol = _solve(sim, t_eval, t_interp, None, verbose)
    else:
        if verbose:
            print("  WARNING: no initial-state API found; SOC is defined with the "
                  "diagnostic voltage limits")
        sol = _solve(sim, t_eval, t_interp, initial_soc, verbose)
    runtime = time.perf_counter() - t0

    t_sim = np.asarray(sol["Time [s]"].entries).ravel()
    V_sim = np.asarray(sol["Voltage [V]"].entries).ravel()
    stopped_early = t_sim[-1] < data.t[-1] - 1e-6
    if verbose:
        print(f"  solved in {runtime:.1f} s, simulated to {t_sim[-1]:.1f} s "
              f"of {data.t[-1]:.1f} s{'  (STOPPED EARLY)' if stopped_early else ''}")
        if stopped_early:
            print(f"  termination: {getattr(sol, 'termination', 'unknown')}")
    return {
        "sim": sim, "sol": sol, "t_sim": t_sim, "V_sim": V_sim,
        "initial_soc_used": initial_soc, "balance": balance, "runtime_s": runtime,
        "stopped_early": bool(stopped_early),
        "termination": str(getattr(sol, "termination", "unknown")),
    }


def _solve(sim, t_eval, t_interp, initial_soc, verbose):
    """Solve, tolerating small API differences between PyBaMM versions."""
    kwargs = {} if initial_soc is None else {"initial_soc": initial_soc}
    try:
        return sim.solve(t_eval, t_interp=t_interp, **kwargs)
    except TypeError:
        if verbose:
            print("  (this PyBaMM version has no t_interp; solving at breakpoints)")
        return sim.solve(t_eval, **kwargs)


def compare_on_data_grid(data, result):
    """Simulated voltage on the measured time grid (overlap only)."""
    t_sim, V_sim = result["t_sim"], result["V_sim"]
    mask = data.t <= t_sim[-1] + 1e-9
    V_on_grid = np.interp(data.t[mask], t_sim, V_sim)
    return mask, V_on_grid


def voltage_metrics(err, I=None, rest_threshold=0.05):
    """Error metrics in volts. err = simulated - measured."""
    m = {
        "RMSE_V": float(np.sqrt(np.mean(err ** 2))),
        "MAE_V": float(np.mean(np.abs(err))),
        "MaxAbs_V": float(np.max(np.abs(err))),
        "MeanError_V": float(np.mean(err)),
        "StdError_V": float(np.std(err)),
    }
    if I is not None:
        rest = np.abs(I) < rest_threshold
        if rest.any():
            m["RMSE_rest_V"] = float(np.sqrt(np.mean(err[rest] ** 2)))
        if (~rest).any():
            m["RMSE_load_V"] = float(np.sqrt(np.mean(err[~rest] ** 2)))
    return m