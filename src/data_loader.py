"""
data_loader.py - Shared data loading for the MEAtec A123 ANR26650M1B case study.

File formats (checked in MATLAB):
  Charge_1C/2C/3C/4C.mat : top-level vectors  Time, Current, Voltage
  Discharge_1C.mat, HPPC.mat : struct 'meas' with fields Time, Current, Voltage, Ah, ...

Sign convention:
  Raw data : charge current > 0, discharge current < 0
  PyBaMM   : discharge current > 0   ->   I_pybamm = -I_raw

Current representation (verified with check_hppc_current.py):
  The cycler logs sparsely during slow CC steps (gaps up to ~267 s), and the
  sample AFTER a gap holds the current that flowed during the gap.
  -> Backward zero-order hold: current I[i] applies on (t[i-1], t[i]].
  On HPPC this matches the logged Ah counter to ~6 mAh, whereas linear
  interpolation loses ~0.31 Ah. All datasets use the same representation.

Run on its own to check the data and make an overview plot:
    python src\\data_loader.py
Outputs -> results/step0_data/
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# ------------------------------------------------------------------ cell facts
NOMINAL_CAPACITY_AH = 2.5          # A123 ANR26650M1B rating
V_MIN_TEST, V_MAX_TEST = 2.5, 3.6  # voltage window seen in the test data

DATASETS = {
    "C1": "Charge_1C.mat",
    "C2": "Charge_2C.mat",
    "C3": "Charge_3C.mat",
    "C4": "Charge_4C.mat",
    "D1": "Discharge_1C.mat",
    "HPPC": "HPPC.mat",
}
TRAINING_KEYS = ["C1", "C2", "C3", "C4", "D1"]


def cumulative_ah_zoh_backward(t, I):
    """Cumulative charge [Ah] with backward zero-order hold (sign as given)."""
    inc = I[1:] * np.diff(t)
    return np.concatenate(([0.0], np.cumsum(inc))) / 3600.0


@dataclass
class CycleData:
    name: str
    t: np.ndarray                     # time [s], starts at 0, strictly increasing
    I_raw: np.ndarray                 # current [A], charge > 0 (as logged)
    V: np.ndarray                     # terminal voltage [V]
    ah_logged: Optional[np.ndarray]   # cycler Ah counter (rebased to 0), if present
    n_removed: int = 0                # samples removed during cleaning

    @property
    def I_pybamm(self):
        """Current at the sample times, PyBaMM convention (discharge > 0)."""
        return -self.I_raw

    def current_profile(self):
        """
        Step-shaped current for a PyBaMM linear Interpolant (discharge > 0).
        Implements backward zero-order hold: I[i] on (t[i-1], t[i]].
        Each interval gets two points: just after t[i-1] and at t[i].
        """
        t, I = self.t, self.I_pybamm
        dt = np.diff(t)
        eps = np.minimum(1e-3, 0.05 * dt)  # tiny ramp, always < dt
        tp = np.empty(1 + 2 * dt.size)
        Ip = np.empty_like(tp)
        tp[0], Ip[0] = t[0], I[1] if t.size > 1 else I[0]
        tp[1::2] = t[:-1] + eps
        tp[2::2] = t[1:]
        Ip[1::2] = I[1:]
        Ip[2::2] = I[1:]
        return tp, Ip

    def ah_model(self):
        """Cumulative charge [Ah] the model will see (charge > 0)."""
        return cumulative_ah_zoh_backward(self.t, self.I_raw)

    def summary(self):
        q = self.ah_model()
        dq = np.diff(q, prepend=0.0)
        s = {
            "samples": int(self.t.size),
            "removed_samples": int(self.n_removed),
            "duration_s": float(self.t[-1]),
            "median_dt_s": float(np.median(np.diff(self.t))),
            "max_dt_s": float(np.max(np.diff(self.t))),
            "V_start": float(self.V[0]),
            "V_end": float(self.V[-1]),
            "V_min": float(self.V.min()),
            "V_max": float(self.V.max()),
            "I_min_A": float(self.I_raw.min()),
            "I_max_A": float(self.I_raw.max()),
            "Ah_charged": float(dq[dq > 0].sum()),
            "Ah_discharged": float(-dq[dq < 0].sum()),
            "Ah_net_end": float(q[-1]),
        }
        if self.ah_logged is not None:
            s["Ah_logged_end"] = float(self.ah_logged[-1])
            s["Ah_max_abs_diff_vs_logged"] = float(np.max(np.abs(q - self.ah_logged)))
        return s


def _get(src, key):
    return src[key] if isinstance(src, dict) else getattr(src, key)


def _has(src, key):
    return key in src if isinstance(src, dict) else hasattr(src, key)


def load_dataset(key):
    """Load one dataset by key ('C1'...'C4', 'D1', 'HPPC') and clean it."""
    path = DATA_DIR / DATASETS[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Put all .mat files in {DATA_DIR}")

    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    src = m["meas"] if "meas" in m else m

    t = np.asarray(_get(src, "Time"), dtype=float).ravel()
    I = np.asarray(_get(src, "Current"), dtype=float).ravel()
    V = np.asarray(_get(src, "Voltage"), dtype=float).ravel()
    ah = np.asarray(_get(src, "Ah"), dtype=float).ravel() if _has(src, "Ah") else None
    n0 = t.size

    ok = np.isfinite(t) & np.isfinite(I) & np.isfinite(V)
    if ah is not None:
        ok &= np.isfinite(ah)
    t, I, V = t[ok], I[ok], V[ok]
    ah = ah[ok] if ah is not None else None

    order = np.argsort(t, kind="stable")
    keep = np.concatenate(([True], np.diff(t[order]) > 0))  # strictly increasing
    idx = order[keep]
    t, I, V = t[idx], I[idx], V[idx]
    if ah is not None:
        ah = ah[idx] - ah[idx][0]

    t = t - t[0]
    return CycleData(key, t, I, V, ah, n_removed=n0 - t.size)


def load_all(keys=None):
    return {k: load_dataset(k) for k in (keys or DATASETS)}


# ------------------------------------------------------------------ self-check
def main():
    out = RESULTS_DIR / "step0_data"
    out.mkdir(parents=True, exist_ok=True)

    data = load_all()
    summaries = {k: d.summary() for k, d in data.items()}

    print(f"{'Set':<5}{'N':>7}{'dur[s]':>9}{'dt_max':>8}{'V0':>7}{'Vmin':>7}"
          f"{'Vmax':>7}{'Imin':>8}{'Imax':>8}{'Ah_in':>7}{'Ah_out':>7}{'Ah_net':>8}"
          f"{'logged':>8}{'diff':>7}{'rm':>5}")
    for k, s in summaries.items():
        logged = f"{s['Ah_logged_end']:>8.3f}" if "Ah_logged_end" in s else f"{'-':>8}"
        diff = f"{s['Ah_max_abs_diff_vs_logged']:>7.3f}" if "Ah_logged_end" in s else f"{'-':>7}"
        print(f"{k:<5}{s['samples']:>7}{s['duration_s']:>9.0f}{s['max_dt_s']:>8.1f}"
              f"{s['V_start']:>7.3f}{s['V_min']:>7.3f}{s['V_max']:>7.3f}"
              f"{s['I_min_A']:>8.2f}{s['I_max_A']:>8.2f}"
              f"{s['Ah_charged']:>7.3f}{s['Ah_discharged']:>7.3f}{s['Ah_net_end']:>8.3f}"
              f"{logged}{diff}{s['removed_samples']:>5}")

    # check the PyBaMM step profile integrates to the same charge
    print("\nStep-profile check (charge the model will receive, discharge > 0):")
    for k, d in data.items():
        tp, Ip = d.current_profile()
        q_profile = np.trapezoid(Ip, tp) / 3600
        print(f"  {k:<5} profile {q_profile:>8.4f} Ah   ZOH-b {-d.ah_model()[-1]:>8.4f} Ah")

    (out / "data_summary.json").write_text(json.dumps(summaries, indent=2))

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for ax, (k, d) in zip(axes.ravel(), data.items()):
        th = d.t / 3600
        ax.plot(th, d.V, color="tab:blue", lw=1)
        ax.set_ylabel("Voltage [V]", color="tab:blue")
        ax.set_xlabel("Time [h]")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(th, d.I_raw, color="tab:orange", lw=0.8)
        ax2.set_ylabel("Current [A] (charge +)", color="tab:orange")
        ax.set_title(f"{k}: {DATASETS[k]}")
    fig.suptitle("A123 ANR26650M1B - measured data overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "data_overview.png", dpi=150)

    print(f"\nSaved: {out / 'data_summary.json'}")
    print(f"Saved: {out / 'data_overview.png'}")


if __name__ == "__main__":
    main()