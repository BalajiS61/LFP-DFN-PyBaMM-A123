"""
final_model.py - The final (optimized) A123 ANR26650M1B model, ready to reuse.

    Prada2013
  + Step 3c thermodynamics (graphite OCP Ecker2015, electrode capacities, stoichiometry window)
  + Step 4 optimized parameters (exchange-current densities, particle diffusivities)
  + Step 4 output terms: contact resistance R_c (and hysteresis, if it was fitted)

Parameters are read from results/step4/optimized_parameters.json (saved by step4_optimize.py).

Usage
    from final_model import load_optimized, optimized_parameter_values, simulate_optimized
    opt = load_optimized()
    pv, calib = optimized_parameter_values(opt)          # pybamm.ParameterValues
    res, mask, V = simulate_optimized(data, "HPPC", opt)  # terminal voltage incl. R_c / hysteresis
"""
import json

import numpy as np

from calibration import set_state_from_full
from data_loader import RESULTS_DIR
from model_setup import compare_on_data_grid, simulate
from step4_optimize import ETA0, build_parameter_values, hysteresis_state, start_from_full

OPT_FILE = RESULTS_DIR / "step4" / "optimized_parameters.json"


def load_optimized(path=None):
    path = path or OPT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run step4_optimize.py first")
    return json.loads(path.read_text())


def optimized_parameter_values(opt=None):
    """pybamm.ParameterValues of the final model (initial state not yet set)."""
    opt = opt or load_optimized()
    return build_parameter_values(opt["optimized"])


def output_correction(data, key, opt):
    """Voltage terms outside the DFN: + R_c * I (charge > 0) + dh * eta(t) if hysteresis was fitted."""
    corr = opt["contact_resistance_Ohm"] * data.I_raw
    if opt.get("hysteresis_fitted"):
        corr = corr + opt["hysteresis_dh_V"] * hysteresis_state(data, ETA0.get(key, 1.0),
                                                                opt["hysteresis_kappa"])
    return corr


def simulate_optimized(data, key, opt=None, mesh=None, ah_from_full=None):
    """
    Simulate one dataset with the final model.
    Returns (pybamm result dict, mask of covered samples, terminal voltage on those samples).
    """
    opt = opt or load_optimized()
    pv, calib = optimized_parameter_values(opt)
    ah0 = start_from_full(data, key) if ah_from_full is None else ah_from_full
    pv = set_state_from_full(pv, calib, ah0)
    res = simulate(data, parameter_values=pv, initial_soc=None, var_pts=mesh, verbose=False)
    mask, V_dfn = compare_on_data_grid(data, res)
    V = V_dfn + output_correction(data, key, opt)[mask]
    return res, mask, V