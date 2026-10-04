"""
check_hppc_current.py - Check that the HPPC current reproduces the cycler's Ah counter.

Compares the logged 'Ah' field with three integrations of 'Current':
  trapz : linear interpolation between samples (what a linear interpolant does)
  ZOH-f : each sample's current held until the next sample (forward hold)
  ZOH-b : each sample's current held back to the previous sample (backward hold)
The method that matches the logged Ah tells us how to feed current to PyBaMM.

Run from the project root:
    python src\\check_hppc_current.py
Outputs -> results/step0_data/
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "HPPC.mat"
OUT = ROOT / "results" / "step0_data"
OUT.mkdir(parents=True, exist_ok=True)


def load_raw():
    m = sio.loadmat(DATA_FILE, squeeze_me=True, struct_as_record=False)
    s = m["meas"]
    t = np.asarray(s.Time, float).ravel()
    I = np.asarray(s.Current, float).ravel()
    V = np.asarray(s.Voltage, float).ravel()
    ah = np.asarray(s.Ah, float).ravel()
    order = np.argsort(t, kind="stable")  # keep original order for equal times
    return t[order], I[order], V[order], ah[order]


def cumulative(t, I, method):
    dt = np.diff(t)
    if method == "trapz":
        inc = 0.5 * (I[1:] + I[:-1]) * dt
    elif method == "zoh_f":
        inc = I[:-1] * dt
    elif method == "zoh_b":
        inc = I[1:] * dt
    else:
        raise ValueError(method)
    return np.concatenate(([0.0], np.cumsum(inc))) / 3600.0


def main():
    t, I, V, ah = load_raw()
    dt = np.diff(t)
    ah_rel = ah - ah[0]

    print("=" * 66)
    print("HPPC CURRENT INTEGRITY CHECK")
    print("=" * 66)
    print(f"Samples (raw)               : {t.size}")
    print(f"Duration                    : {t[-1] - t[0]:.1f} s")
    print(f"Logged Ah: start {ah[0]:.4f}, end {ah[-1]:.4f}, min {ah.min():.4f}, max {ah.max():.4f}")

    # duplicate timestamps
    dup = np.where(dt == 0)[0]
    dup_diff_I = np.sum(np.abs(I[dup + 1] - I[dup]) > 0.05)
    print(f"\nDuplicate timestamps        : {dup.size}")
    print(f"  ...with different current : {dup_diff_I}")

    # sampling structure
    print(f"\nTime step dt [s]: median {np.median(dt):.3f}, "
          f"95th pct {np.percentile(dt, 95):.3f}, max {dt.max():.3f}")
    big = np.argsort(dt)[::-1][:10]
    print("\nTop 10 largest time gaps:")
    print(f"{'t_start[s]':>12}{'gap[s]':>9}{'I_before[A]':>13}{'I_after[A]':>12}")
    for i in sorted(big):
        print(f"{t[i]:>12.1f}{dt[i]:>9.2f}{I[i]:>13.3f}{I[i + 1]:>12.3f}")

    # integration comparison
    print("\nCumulative charge vs logged Ah counter (discharge negative):")
    print(f"{'method':<8}{'end Ah':>10}{'end diff':>11}{'max |diff|':>12}")
    print(f"{'logged':<8}{ah_rel[-1]:>10.4f}{0:>11.4f}{0:>12.4f}")
    curves = {}
    for meth in ["trapz", "zoh_f", "zoh_b"]:
        q = cumulative(t, I, meth)
        curves[meth] = q
        d = q - ah_rel
        print(f"{meth:<8}{q[-1]:>10.4f}{d[-1]:>11.4f}{np.max(np.abs(d)):>12.4f}")

    best = min(curves, key=lambda k: np.max(np.abs(curves[k] - ah_rel)))
    print(f"\nBest match to logged Ah     : {best}")

    # where does the trapz error build up?
    err_tr = curves["trapz"] - ah_rel
    jumps = np.diff(err_tr)
    worst = np.argsort(np.abs(jumps))[::-1][:5]
    print("\nLargest single-step trapz error contributions:")
    print(f"{'t[s]':>10}{'dt[s]':>8}{'I_i[A]':>9}{'I_i+1[A]':>10}{'dErr[mAh]':>11}")
    for i in sorted(worst):
        print(f"{t[i]:>10.1f}{dt[i]:>8.2f}{I[i]:>9.2f}{I[i + 1]:>10.2f}{jumps[i] * 1000:>11.2f}")

    # plots
    th = (t - t[0]) / 3600
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(th, I, lw=0.6)
    ax[0].set_ylabel("Current [A]")
    ax[0].set_title("HPPC current integrity check")
    ax[0].grid(alpha=0.3)
    ax[1].plot(th, ah_rel, "k", lw=1.5, label="Logged Ah")
    for k, q in curves.items():
        ax[1].plot(th, q, "--", lw=1, label=k)
    ax[1].set_ylabel("Cumulative charge [Ah]")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    for k, q in curves.items():
        ax[2].plot(th, (q - ah_rel) * 1000, lw=1, label=k)
    ax[2].set_ylabel("Integrated - logged [mAh]")
    ax[2].set_xlabel("Time [h]")
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "hppc_current_check.png", dpi=150)
    print(f"\nSaved: {OUT / 'hppc_current_check.png'}")


if __name__ == "__main__":
    main()