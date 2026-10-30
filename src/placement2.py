"""Revision experiment R2 (rev_protocol.md): placement evaluated on an independent library.

Layouts are selected exactly as in placement.py (on the 1 cm^2 nominal library) and
evaluated on leaks of 5 cm^2 at demand multiplier 1.2, simulated separately.
Usage: python placement2.py synthetic|ky4|ltown [workers]
Needs lib_<net>_0.0001.npz. Writes lib_eval_<net>.npz and rev_r2_placement_<net>.json.
"""
import json
import os
import sys
import numpy as np
from multiprocessing import Pool
import config
import placement as P                      # reads sys.argv[1]; builds shadows, libraries, layouts
from core import simulate
from networks import load

NET = P.NET
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else config.WORKERS
AREA, MULT = 5e-4, 1.2
SIGMAS = (0.01, 0.05)
N_NOISE = 10
_wn = _h0 = None


def _init():
    global _wn, _h0
    _wn = load(NET)
    _h0, _ = simulate(_wn, demand_mult=MULT)


def _one(j):
    h, _ = simulate(_wn, {j: AREA}, demand_mult=MULT)
    return (_h0 - h)[P.J].values.astype(np.float32)


def eval_library():
    f = f"lib_eval_{NET}.npz"
    if os.path.exists(f):
        return np.load(f)["R"].astype(float)
    print(f"{NET}: simulating the evaluation library ({len(P.cand)} leaks, {AREA*1e4:.0f} cm2, demand x{MULT})", flush=True)
    with Pool(WORKERS, initializer=_init) as pool:
        R = np.array(pool.map(_one, P.cand, chunksize=4))
    np.savez_compressed(f, R=R, cand=np.array(P.cand))
    return R.astype(float)


def evaluate(sens, dictionary, obsR, sigma, seed=123):
    sens = np.array(sens)
    D = dictionary[:, sens]
    D = D - D.mean(1, keepdims=True)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    rn = np.random.default_rng(seed)               # identical noise draws for every layout
    err, t1, t5, ties = [], [], [], []
    nC = len(obsR)
    for _ in range(N_NOISE):
        obs = obsR[:, sens] + rn.normal(0, sigma, (nC, len(sens)))
        obs = obs - obs.mean(1, keepdims=True)
        S = obs @ Dn.T
        order = np.argsort(-S, axis=1, kind="stable")
        est = order[:, 0]
        top = S[np.arange(nC), est]
        ties.append(np.mean(np.sum(S >= top[:, None] - 1e-12, axis=1) > 1))
        t1.append(np.mean(est == np.arange(nC)))
        t5.append(np.mean([i in order[i, :5] for i in range(nC)]))
        err.append(np.mean([P.hops_from(P.cand[i]).get(P.cand[est[i]], 0) for i in range(nC)]))
    return dict(hop_error=float(np.mean(err)), top1=float(np.mean(t1)), top5=float(np.mean(t5)),
                tied_top_share=float(np.mean(ties)))


if __name__ == "__main__":
    Rev = eval_library()
    dictionaries = {"library": P.R, "linear": P.Lsig}
    out = {"network": NET, "eval_area_m2": AREA, "eval_demand_mult": MULT, "budgets": {}}
    rng = np.random.default_rng(7)
    for frac in P.FRACS:
        m = max(3, int(round(frac * P.nJ)))
        util = m == len(P.UTILITY.get(NET, []))
        key = f"{m} sensors (utility budget)" if util else f"{int(round(frac*100))}%"
        layouts = {"pagerank": P.pagerank(m), "dom_struct": P.dom_struct(m), "fps_all": P.fps_all(m),
                   "dom_linear": P.greedy(P.Lsig, m, range(P.nJ), max_per_shadow=1),
                   "greedy_lib": P.greedy(P.R, m, range(P.nJ))}
        if util:
            layouts["utility_layout"] = [P.jidx[n] for n in P.UTILITY[NET]]
        rand = [rng.choice(P.nJ, m, replace=False) for _ in range(10)]
        row = {}
        print(f"budget {key} (m={m})", flush=True)
        for dname, Dm in dictionaries.items():
            for s in SIGMAS:
                if dname == "linear" and s != SIGMAS[0]:
                    continue
                tag = f"{dname}_sigma{s}"
                row[tag] = {name: evaluate(L, Dm, Rev, s) for name, L in layouts.items()}
                rr = [evaluate(L, Dm, Rev, s) for L in rand]
                row[tag]["random"] = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
                row[tag]["random"]["hop_error_sd"] = float(np.std([r["hop_error"] for r in rr], ddof=1))
                line = "  ".join(f"{n} {v['hop_error']:.2f}" for n, v in row[tag].items())
                print(f"   {tag:20s} {line}", flush=True)
        out["budgets"][key] = dict(m=m, **row)
    ok = all(v["library_sigma0.01"]["fps_all"]["hop_error"] < min(v["library_sigma0.01"]["dom_struct"]["hop_error"],
                                                                   v["library_sigma0.01"]["pagerank"]["hop_error"])
             for v in out["budgets"].values())
    out["even_spreading_beats_shadow_rule_and_pagerank_at_every_budget"] = bool(ok)
    print(f"\nEven spreading better than the shadow rule and PageRank at every budget: {ok}", flush=True)
    json.dump(out, open(f"rev_r2_placement_{NET}.json", "w"), indent=1)
