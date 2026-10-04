"""
step4c_bbot_feasibility.py - which graphite low-x diffusivity reduction (b_bot) still lets
the model simulate the full HPPC test on the validation (default) mesh?

b_bot improves the 1C discharge end (training data) but can empty the graphite surface during
15C HPPC pulses at low SOC, stopping the run. This scan maps that trade-off before optimizing:
for each (D_n, b_bot) it simulates HPPC and the 1C discharge on the default mesh and reports
HPPC coverage / termination and the D1 error.

Other parameters: k_n 1.16, k_p 10.5, D_p 0.37 x Prada2013 (split-fit values).

Run from the project root:
    python src\\step4c_bbot_feasibility.py --workers 3
Outputs -> results/step4c/
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from data_loader import RESULTS_DIR, load_dataset
from model_setup import voltage_metrics
from step4_optimize import _simulate_set

OUT = RESULTS_DIR / "step4c"
BASE = {"k_n": 1.16, "k_p": 10.5, "D_p": 0.37}
COMBOS = [(3.6e-14, 0.0), (3.6e-14, 0.5), (3.6e-14, 1.0), (3.6e-14, 1.5), (3.6e-14, 2.0),
          (6.0e-14, 2.0)]
SETS = ["HPPC", "D1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("STEP 4c - b_bot FEASIBILITY ON THE DEFAULT MESH (HPPC coverage vs D1 error)")
    print("=" * 66)
    print("Base: " + ", ".join(f"{k}={v}" for k, v in BASE.items()))
    data = {k: load_dataset(k) for k in SETS}
    jobs, labels = [], []
    for dn, bb in COMBOS:
        vals = dict(BASE, D_n=dn, b_bot=bb)
        for k in SETS:
            jobs.append((k, vals, {}))          # {} -> PyBaMM default mesh
            labels.append((dn, bb, k))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        out = list(pool.map(_simulate_set, jobs))

    res = {}
    for (dn, bb, k), (_, V, info) in zip(labels, out):
        d = data[k]
        if V is None:
            res[(dn, bb, k)] = {"coverage": 0.0, "termination": str(info), "rmse_mV": float("nan")}
            continue
        ok = np.isfinite(V)
        t_stop = float(d.t[ok][-1]) if ok.any() else 0.0
        m = voltage_metrics(V[ok] - d.V[ok], d.I_raw[ok]) if ok.any() else {"RMSE_V": float("nan")}
        res[(dn, bb, k)] = {"coverage": float(ok.mean()), "termination": str(info),
                            "stop_h": t_stop / 3600, "rmse_mV": 1000 * m["RMSE_V"]}

    print(f"\n  {'D_n':>9}{'b_bot':>7} | {'HPPC cover':>11}{'stop [h]':>10}{'HPPC RMSE':>11} | "
          f"{'D1 cover':>9}{'D1 RMSE':>9}   [mV]  verdict")
    rows = []
    for dn, bb in COMBOS:
        h, d1 = res[(dn, bb, "HPPC")], res[(dn, bb, "D1")]
        ok_h = h["coverage"] > 0.999
        verdict = "feasible" if ok_h and d1["coverage"] > 0.999 else "STOPS"
        print(f"  {dn:>9.2e}{bb:>7.1f} | {100 * h['coverage']:>10.1f}%{h.get('stop_h', 0):>10.2f}"
              f"{h['rmse_mV']:>11.1f} | {100 * d1['coverage']:>8.1f}%{d1['rmse_mV']:>9.1f}   {verdict}")
        rows.append({"D_n": dn, "b_bot": bb, "hppc": h, "d1": d1, "feasible": verdict == "feasible"})
    feas = [r for r in rows if r["feasible"] and r["D_n"] == COMBOS[0][0]]
    if feas:
        best = max(feas, key=lambda r: r["b_bot"])
        print(f"\n  Largest feasible b_bot at D_n {COMBOS[0][0]:.1e}: {best['b_bot']:.1f} "
              f"(D1 RMSE {best['d1']['rmse_mV']:.1f} mV)")
    else:
        print("\n  No feasible b_bot at this D_n - HPPC stops even without the low-x reduction")
    (OUT / "bbot_feasibility.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nSaved: {OUT / 'bbot_feasibility.json'}")


if __name__ == "__main__":
    main()