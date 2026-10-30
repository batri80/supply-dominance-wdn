"""Simulated leak scenarios for training and testing the hybrid localiser on L-Town.

Each scenario: one leak (70%) or two concurrent leaks (30%) at random candidate
junctions, orifice area uniform on [1, 10] cm^2, demand multiplier uniform on
[0.6, 1.4]. Saved: head drop at every junction relative to the leak-free state
at the same demand, leak nodes, areas and multiplier.

Usage: python gen_train.py ltown N [workers] [seed]
"""
import sys
import numpy as np
import config
from multiprocessing import Pool
from core import simulate
from networks import load

NET, N = sys.argv[1], int(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else config.WORKERS
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
_wn = _J = None
_base = {}


def _init():
    global _wn, _J
    _wn = load(NET)
    _J = _wn.junction_name_list


def _one(sc):
    nodes, areas, mult = sc
    key = round(mult, 3)
    if key not in _base:
        _base[key] = simulate(_wn, demand_mult=mult)[0][_J].values
    h, _ = simulate(_wn, dict(zip(nodes, areas)), demand_mult=mult)
    return (_base[key] - h[_J].values).astype(np.float32)


if __name__ == "__main__":
    wn = load(NET)
    J = wn.junction_name_list
    h0, _ = simulate(wn)
    elev = np.array([wn.get_node(n).elevation for n in J])
    cand = [n for n, p in zip(J, h0[J].values - elev) if p > 15]
    rng = np.random.default_rng(SEED)
    mults = np.round(rng.choice(np.linspace(0.6, 1.4, 9), N), 3)   # 9 levels so baselines are reused
    scen = []
    for k in range(N):
        n_leak = 2 if rng.random() < 0.3 else 1
        nodes = list(rng.choice(cand, n_leak, replace=False))
        areas = list(rng.uniform(1e-4, 1e-3, n_leak))
        scen.append((nodes, areas, float(mults[k])))
    out = []
    with Pool(WORKERS, initializer=_init) as pool:
        for k, r in enumerate(pool.imap(_one, scen, chunksize=8)):
            out.append(r)
            if (k + 1) % 250 == 0:
                print(f"  {k+1}/{N}", flush=True)
    np.savez(f"train_{NET}_s{SEED}.npz", J=np.array(J), cand=np.array(cand), R=np.array(out),
             nodes=np.array([s[0] + [""] * (2 - len(s[0])) for s in scen]),
             areas=np.array([s[1] + [0.0] * (2 - len(s[1])) for s in scen]),
             mult=np.array([s[2] for s in scen]))
    print("saved", flush=True)
