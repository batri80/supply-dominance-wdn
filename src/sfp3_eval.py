"""Revision experiment R1 (rev_protocol.md): evaluate NEET-2PT under model mismatch.

Usage: python sfp3_eval.py [networks...]   (default: ltown ky4)
Needs rev_r1_<net>.npy (sfp3_sim.py). Writes rev_r1_results.json.
"""
import json
import sys
import numpy as np
from scipy.stats import spearmanr
import config  # noqa: F401

NETS = sys.argv[1:] or ["ltown", "ky4"]
PMIN, INSERV, A0, A1 = 20.0, 22.0, 2e-4, 4e-4
CONDITIONS = ["nominal", "roughness", "demand", "pdd", "cd", "combined", "drift"]
NOISE = (0.01, 0.05, 0.10)


def two_point(p0, pA0, pA1, serv):
    dh0 = np.maximum(p0 - pA0, 1e-9)
    dh1 = np.maximum(p0 - pA1, 1e-9)
    b = np.clip(np.log(dh1 / dh0) / np.log(A1 / A0), 0.5, 3.0)
    c = dh0 / A0 ** b
    M = np.maximum(p0 - PMIN, 1e-6)
    Ai = np.where(serv, (M / c) ** (1.0 / b), np.inf)
    return Ai


def summary(pred, A, nodes_pred=None, first=None):
    pred, A = np.asarray(pred, float), np.asarray(A, float)
    rel = np.abs(pred - A) / A
    out = dict(n=int(len(A)), median_rel_error=float(np.median(rel)), within_50pct=float(np.mean(rel < 0.5)),
               spearman=float(spearmanr(pred, A).correlation), median_log_ratio=float(np.median(np.log(pred / A))))
    if nodes_pred is not None:
        out["first_top1"] = float(np.mean([a == b for a, b in zip(nodes_pred[0], first)]))
        out["first_top1_neartie"] = float(np.mean([b in s for s, b in zip(nodes_pred[1], first)]))
    return out


def predict(p0, pA0, pA1, serv):
    Ai = two_point(p0, pA0, pA1, serv)
    m = Ai.min()
    return m, int(np.argmin(Ai)), set(np.where(Ai <= 1.05 * m)[0].tolist())


results = {}
for net in NETS:
    R = list(np.load(f"rev_r1_{net}.npy", allow_pickle=True))
    nom = {r["k"]: r for r in R if r["cond"] == "nominal"}
    res = {}
    print(f"\n{net}: NEET-2PT median relative error / within 50% / Spearman / median log ratio / first-node top-1")
    for cond in CONDITIONS:
        rows = [r for r in R if r["cond"] == cond and not r["censored"]]
        A = [r["Astar"] for r in rows]
        first = [r["first"] for r in rows]
        res[cond] = dict(censored=int(sum(r["cond"] == cond and r["censored"] for r in R)))
        routes = {"MODEL": [predict(nom[r["k"]]["p0"], nom[r["k"]]["pA0"], nom[r["k"]]["pA1"],
                                    nom[r["k"]]["p0"] >= INSERV) for r in rows]}
        if cond != "drift":
            routes["OBS"] = [predict(r["p0"], r["pA0"], r["pA1"], r["serv_true"]) for r in rows]
        for route, P in routes.items():
            s = summary([p[0] for p in P], A, ([p[1] for p in P], [p[2] for p in P]), first)
            res[cond][route] = s
            print(f"  {cond:9s} {route:5s} {s['median_rel_error']:.3f}  {s['within_50pct']:.2f}  {s['spearman']:.3f}  "
                  f"{s['median_log_ratio']:+.3f}  {s['first_top1']:.2f} (near-ties {s['first_top1_neartie']:.2f})  n={s['n']}")
    # noise on the observed responses (nominal condition)
    rows = [r for r in R if r["cond"] == "nominal" and not r["censored"]]
    A = np.array([r["Astar"] for r in rows])
    res["noise"] = {}
    rng = np.random.default_rng(0)
    for s in NOISE:
        meds, w50 = [], []
        for _ in range(20):
            pred = []
            for r in rows:
                e = lambda x: x + rng.normal(0, s, x.shape)        # noqa: E731
                pred.append(predict(e(r["p0"]), e(r["pA0"]), e(r["pA1"]), r["serv_true"])[0])
            rel = np.abs(np.array(pred) - A) / A
            meds.append(np.median(rel)); w50.append(np.mean(rel < 0.5))
        res["noise"][str(s)] = dict(median_rel_error=float(np.mean(meds)), median_rel_error_sd=float(np.std(meds)),
                                    within_50pct=float(np.mean(w50)))
        print(f"  noise {s:.2f} m   OBS {np.mean(meds):.3f} (sd {np.std(meds):.3f})  within 50% {np.mean(w50):.2f}")
    # wrong leak junction (model route)
    P = [predict(r["p0"], r["pA0w"], r["pA1w"], r["p0"] >= INSERV) for r in rows]
    res["location_error"] = summary([p[0] for p in P], A)
    print(f"  wrong junction (1-3 hops) MODEL {res['location_error']['median_rel_error']:.3f}  "
          f"within 50% {res['location_error']['within_50pct']:.2f}")
    res["robust"] = {c: res[c]["MODEL"]["median_rel_error"] < 0.5 for c in CONDITIONS}
    results[net] = res

decision = {c: all(results[n]["robust"][c] for n in results) for c in CONDITIONS}
results["decision"] = {c: ("robust on all networks" if v else "not robust") for c, v in decision.items()}
print("\nPre-specified decision (MODEL route, median relative error < 0.5 on every network):")
for c, v in results["decision"].items():
    print(f"  {c:9s} {v}")
json.dump(results, open("rev_r1_results.json", "w"), indent=1)
