"""
step4b_dp_shape_probe.py - feasibility check BEFORE a long optimization.

Question: can state-dependent solid diffusivities reproduce the two features a
constant-diffusivity DFN misses?
    - the CV hold at 3.6 V during the charge tests  -> LFP near y -> 0 (a_top)
    - the voltage plunge at the end of the 1C discharge -> graphite near x -> 0 (b_bot)
      (first probe: an LFP term near y -> 1 had no effect - LFP only reaches y ~ 0.75)

D_p(y) = D_p,0 * 10^-(a_top exp(-y/w)),  D_n(x) = D_n,0 * 10^-(b_bot exp(-x/w)),  w = 0.05
Base parameters: Run A result (results/step4/optimized_parameters.json) if present,
otherwise the Task 3 model. Coarse mesh (as in the optimization).

Run from the project root:
    python src\\step4b_dp_shape_probe.py
    python src\\step4b_dp_shape_probe.py --workers 2
Outputs -> results/step4b/
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import DATASETS, RESULTS_DIR, load_dataset
from model_setup import COARSE_MESH
from step4_optimize import _simulate_set

OUT = RESULTS_DIR / "step4b"
# (a_top: LFP slow near y=0, b_bot: graphite slow near x=0)
COMBOS = [(0, 0), (0.1, 0), (0.25, 0), (0.5, 0), (0, 1), (0, 2), (0, 3), (0.25, 2)]
SETS = ["C1", "C4", "D1"]


def phase_errors(d, key, V):
    ok = np.isfinite(V)
    e = (V - d.V) * 1000
    I = d.I_raw
    out = {"covered": float(ok.mean())}
    if key.startswith("C"):
        q = d.ah_model()
        win = ok & (q >= 0.15)
        cc = win & (I > 0.95 * I.max())
        cv = win & ~cc
        out.update({"all": float(np.sqrt(np.mean(e[win] ** 2))),
                    "CC": float(np.sqrt(np.mean(e[cc] ** 2))) if cc.any() else np.nan,
                    "CV": float(np.sqrt(np.mean(e[cv] ** 2))) if cv.any() else np.nan,
                    "CV_mean": float(np.mean(e[cv])) if cv.any() else np.nan})
    else:
        q = -d.ah_model()
        load = ok & (I < -0.05)
        end = load & (q > 0.95 * q.max())
        rest_after = ok & (np.abs(I) < 0.05) & (d.t > d.t[load][-1] if load.any() else False)
        out.update({"all": float(np.sqrt(np.mean(e[ok] ** 2))),
                    "CC": float(np.sqrt(np.mean(e[load & ~end] ** 2))),
                    "end5%": float(np.sqrt(np.mean(e[end] ** 2))) if end.any() else np.nan,
                    "rest_after": float(np.sqrt(np.mean(e[rest_after] ** 2))) if rest_after.any() else np.nan})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    run_a = RESULTS_DIR / "step4" / "optimized_parameters.json"
    if run_a.exists():
        base = dict(json.loads(run_a.read_text())["optimized"])
        src = "Run A (step4)"
    else:
        base = {"k_n": 1.0, "k_p": 1.0, "D_n": 3.0e-14, "D_p": 1.0}
        src = "Task 3 model"
    base["k_p"] = 1.0                                   # fixed in the shape model
    if base.get("D_n", 3e-14) < 3.0e-14:
        base["D_n"] = 3.0e-14                           # Task 3e: needed to complete HPPC
    print("=" * 66)
    print("STEP 4b - FEASIBILITY: state-dependent LFP diffusivity")
    print("=" * 66)
    print(f"Base parameters ({src}, k_p fixed 1, D_n >= 3e-14): "
          + ", ".join(f"{k}={v:.3g}" for k, v in base.items()))

    jobs, labels = [], []
    for at, ab in COMBOS:
        vals = dict(base, a_top=float(at), b_bot=float(ab))
        for k in SETS:
            jobs.append((k, vals, COARSE_MESH))
            labels.append((at, ab, k))
    data = {k: load_dataset(k) for k in SETS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        out = list(pool.map(_simulate_set, jobs))

    res = {}
    for (at, ab, k), (_, V, info) in zip(labels, out):
        if V is None:
            res[(at, ab, k)] = {"failed": info}
            continue
        res[(at, ab, k)] = phase_errors(data[k], k, V)
        res[(at, ab, k)]["V"] = V
        res[(at, ab, k)]["info"] = info

    print(f"\n  {'a_top':>5}{'b_bot':>6} | {'C1 all':>7}{'C1 CC':>7}{'C1 CV':>7}{'CVmean':>7} | "
          f"{'C4 all':>7}{'C4 CC':>7}{'C4 CV':>7}{'CVmean':>7} | {'D1 all':>7}{'D1 CC':>7}{'end5%':>7}{'rest':>7}  [mV]")
    table = []
    for at, ab in COMBOS:
        line = f"  {at:>5}{ab:>6} |"
        rowd = {"a_top": at, "b_bot": ab}
        for k in SETS:
            r = res[(at, ab, k)]
            if "failed" in r:
                line += f"  failed: {r['failed'][:40]}"
                continue
            cols = ["all", "CC", "CV", "CV_mean"] if k.startswith("C") else ["all", "CC", "end5%", "rest_after"]
            line += "".join(f"{r[c]:>7.1f}" for c in cols) + " |"
            rowd[k] = {c: r[c] for c in cols}
            if r["covered"] < 0.999:
                line += f"(stopped {100 * r['covered']:.0f}%)"
        print(line)
        table.append(rowd)
    (OUT / "probe.json").write_text(json.dumps({"base": base, "source": src, "rows": table}, indent=2, default=float))

    fig, ax = plt.subplots(len(SETS), 1, figsize=(12, 3.2 * len(SETS)), squeeze=False)
    for i, k in enumerate(SETS):
        d = data[k]
        a = ax[i, 0]
        a.plot(d.t / 60, d.V, "k", lw=1.2, label="Measured")
        for at, ab in COMBOS:
            r = res[(at, ab, k)]
            if "V" in r:
                a.plot(d.t / 60, r["V"], lw=0.8, label=f"a_top={at}, b_bot={ab}")
        a.set_title(DATASETS[k])
        a.set_ylabel("Voltage [V]")
        a.legend(fontsize=7, ncol=4)
        a.grid(alpha=0.3)
    ax[-1, 0].set_xlabel("Time [min]")
    fig.suptitle("Step 4b - state-dependent LFP diffusivity: feasibility", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "step4b_dp_shape_probe.png", dpi=130)
    print(f"\nSaved: {OUT / 'step4b_dp_shape_probe.png'}, probe.json")


if __name__ == "__main__":
    main()