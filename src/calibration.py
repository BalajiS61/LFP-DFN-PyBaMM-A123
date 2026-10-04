"""
calibration.py - Calibrated parameter sets.
  calibrated_parameter_values : Prada2013 + Step 3c thermodynamics
  task3_parameter_values      : + Step 3e graphite diffusivity (manually tuned model)

Step 3c changes to Prada2013:
  - negative electrode OCP taken from the selected literature set (Ecker2015)
  - negative / positive active material volume fractions (electrode capacities)
  - initial state of each test from the calibrated lithium line:
        x = x100 - Ah_from_full / Q_n,   y = y100 + Ah_from_full / Q_p
"""
import functools
import json

import pybamm

from data_loader import RESULTS_DIR
from model_setup import PARAM_SET, base_parameter_values

OCV_CALIB_FILE = RESULTS_DIR / "step3c" / "ocv_calibration.json"
EPS_N = "Negative electrode active material volume fraction"
EPS_P = "Positive electrode active material volume fraction"
DN = "Negative particle diffusivity [m2.s-1]"

# Step 3e decision: smallest physically realistic graphite diffusivity that completes
# the full HPPC test (x10 Prada2013); larger values worsen the full-test error.
TASK3_DN = 3.0e-14


def load_ocv_calibration(path=OCV_CALIB_FILE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run step3c_ocv_calibration.py first")
    return json.loads(path.read_text())


def calibrated_parameter_values(calib=None, extra=None):
    """Prada2013 + Step 3c thermodynamic calibration (+ optional extra updates)."""
    calib = calib or load_ocv_calibration()
    pv = base_parameter_values()
    src = calib["negative_ocp_source"]
    if src != PARAM_SET:
        pv.update({"Negative electrode OCP [V]":
                   pybamm.ParameterValues(src)["Negative electrode OCP [V]"]})
    pv.update({EPS_N: calib[EPS_N], EPS_P: calib[EPS_P]})
    if extra:
        pv.update(extra)
    return pv


def lithium_line(calib, ah_from_full):
    """Electrode stoichiometries after discharging `ah_from_full` Ah from full."""
    x = calib["x100"] - ah_from_full / calib["Q_n_Ah"]
    y = calib["y100"] + ah_from_full / calib["Q_p_Ah"]
    return x, y


def set_state_from_full(pv, calib, ah_from_full):
    """Write the initial electrode concentrations for a state `ah_from_full` below full."""
    x, y = lithium_line(calib, ah_from_full)
    if not (0 < x < 1 and 0 < y < 1):
        raise ValueError(f"state {ah_from_full:.3f} Ah from full is outside the window (x={x:.3f}, y={y:.3f})")
    pv.update({
        "Initial concentration in negative electrode [mol.m-3]":
            x * float(pv["Maximum concentration in negative electrode [mol.m-3]"]),
        "Initial concentration in positive electrode [mol.m-3]":
            y * float(pv["Maximum concentration in positive electrode [mol.m-3]"]),
    })
    return pv


def scale_parameter(pv, name, k):
    """Multiply a parameter (constant or function) by k, keeping a function's signature."""
    v = pv[name]
    if callable(v):
        @functools.wraps(v)
        def scaled(*args, _f=v, _k=k):
            return _k * _f(*args)
        pv.update({name: scaled})
    else:
        pv.update({name: k * float(v)})
    return pv


def task3_parameter_values(calib=None):
    """Manually tuned model at the end of Task 3: Step 3c thermodynamics + D_n from Step 3e."""
    return calibrated_parameter_values(calib, extra={DN: TASK3_DN})