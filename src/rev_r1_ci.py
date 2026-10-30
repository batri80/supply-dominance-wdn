"""Bootstrap confidence intervals for the failure-size results (second review, comment 3).

R1: for each network and condition, the median relative error of NEET-2PT (model route) with a
95% percentile interval from 2,000 bootstrap resamples of the leak junctions. The same resampled
junctions are used for every condition (the conditions are paired), and the paired difference
(condition minus nominal) is reported with its interval. Censored scenarios are excluded within
each condition; the number of uncensored scenarios is reported.
X2: the same intervals for NEET-LIN, NEET-2PT and NEET-REG on the held-out test sets.

Usage: python rev_r1_ci.py
Needs rev_r1_<net>.npy and sfp2_<net>_s<seed>.npy. Writes rev_r1_ci.json. Runs in seconds.
"""
import json
import os
import numpy as np
import config  # noqa: F401
from core import simulate
from networks import load

PMIN, INSERV, A0, A1 = 20.0, 22.0, 2e-4, 4e-4
CONDITIONS = ["nominal", "roughness", "demand", "pdd", "cd", "combined", "drift"]
SEEDS = {"ltown": (3026, 4026), "ky4": (3027, 4027)}
B = 2000
rng = np.random.default_rng(2027)


def two_point(p0, pA0, pA1, serv):
    dh0 = np.maximum(p0 - pA0, 1e-9)
    dh1 = np.maximum(p0 - pA1, 1e-9)
    b = np.clip(np.log(dh1 / dh0) / np.log(A1 / A0), 0.5, 3.0)
    c = dh0 / A0 ** b
    M = np.maximum(p0 - PMIN, 1e-6)
    return float(np.where(serv, (M / c) ** (1.0 / b), np.inf).min())


def ci_median(rel, idx_sets):
    meds = [np.median(rel[s]) for s in idx_sets if len(s)]
    return [float(np.median(rel)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


out = {"R1": {}, "X2": {}}
for net in ("ltown", "ky4"):
    f = f"rev_r1_{net}.npy"
    if not os.path.exists(f):
        print(f"{f} missing, skipped"); continue
    R = list(np.load(f, allow_pickle=True))
    nom = {r["k"]: r for r in R if r["cond"] == "nominal"}
    K = sorted(nom)
    rel = {}
    for cond in CONDITIONS:
        e = np.full(len(K), np.nan)
        for r in R:
            if r["cond"] != cond or r["censored"]:
                continue
            m = nom[r["k"]]
            pred = two_point(m["p0"], m["pA0"], m["pA1"], m["p0"] >= INSERV)
            e[K.index(r["k"])] = abs(pred - r["Astar"]) / r["Astar"]
        rel[cond] = e
    boots = [rng.integers(0, len(K), len(K)) for _ in range(B)]       # resample leak junctions, shared by all conditions
    res = {}
    print(f"\n{net}: NEET-2PT model route, median relative error [95% CI], n uncensored; difference from nominal [95% CI]")
    for cond in CONDITIONS:
        e = rel[cond]
        ok = ~np.isnan(e)
        meds = [np.nanmedian(e[b]) for b in boots]
        diff = [np.nanmedian(e[b]) - np.nanmedian(rel["nominal"][b]) for b in boots]
        res[cond] = dict(n_uncensored=int(ok.sum()), median=float(np.nanmedian(e)),
                         ci=[float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
                         diff_from_nominal=float(np.nanmedian(e) - np.nanmedian(rel["nominal"])),
                         diff_ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))])
        r = res[cond]
        print(f"  {cond:9s} {r['median']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  n={r['n_uncensored']:3d}   "
              f"{r['diff_from_nominal']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]")
    out["R1"][net] = res

# X2 held-out test sets
for net, (cs, ts) in SEEDS.items():
    fc, ft = f"sfp2_{net}_s{cs}.npy", f"sfp2_{net}_s{ts}.npy"
    if not (os.path.exists(fc) and os.path.exists(ft)):
        continue
    wn = load(net)
    J = wn.junction_name_list
    idx = {n: k for k, n in enumerate(J)}
    elev = np.array([wn.get_node(n).elevation for n in J])
    h0, _ = simulate(wn)
    serv = (h0[J].values - elev) >= INSERV

    def feats(s):
        p0, pA0 = s["p0"], s["pA0"]
        dh0 = np.maximum(p0 - pA0, 1e-9)
        m = pA0 - PMIN
        r = 1 + np.maximum(m, 0) / dh0
        jd = idx[s["j"]]
        pj = pA0[jd] - dh0[jd] * (r - 1)
        lin = float(np.where(pj > 0, A0 * r * np.sqrt(pA0[jd] / np.maximum(pj, 1e-9)), np.inf)[serv].min())
        d = (dh0 / np.maximum(m, 1e-6))[serv]
        return s["Astar"], lin, two_point(s["p0"], s["pA0"], s["pA1"], serv), float(d.std() / d.mean()), float(d.mean())

    cal = np.array([feats(s) for s in np.load(fc, allow_pickle=True) if not s["censored"]])
    tst = np.array([feats(s) for s in np.load(ft, allow_pickle=True) if not s["censored"]])

    def design(T):
        return np.column_stack([np.ones(len(T)), np.minimum(np.log(T[:, 2]), 0), np.minimum(np.log(T[:, 1]), 0), T[:, 3], T[:, 4]])

    beta, *_ = np.linalg.lstsq(design(cal), np.log(cal[:, 0]), rcond=None)
    preds = {"NEET_LIN": tst[:, 1], "NEET_2PT": tst[:, 2], "NEET_REG": np.exp(design(tst) @ beta)}
    boots = [rng.integers(0, len(tst), len(tst)) for _ in range(B)]
    out["X2"][net] = {}
    print(f"\n{net} X2 test (n={len(tst)}): median relative error [95% CI]")
    for k, p in preds.items():
        rel = np.abs(p - tst[:, 0]) / tst[:, 0]
        c = ci_median(rel, boots)
        out["X2"][net][k] = dict(median=c[0], ci=c[1:], n=int(len(tst)))
        print(f"  {k:9s} {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]")

json.dump(out, open("rev_r1_ci.json", "w"), indent=1)
print("\nsaved rev_r1_ci.json")
