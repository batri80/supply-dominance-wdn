"""Experiment X2 (ext_protocol.md): leak-growth scenarios with two early states.

As sfp_sim.py, with a chosen seed, and also recording the state at A1 = 4 cm^2.
Usage: python sfp2_sim.py ltown|ky4 SEED [n_scen=300] [workers]
Writes sfp2_<net>_s<seed>.npy; checkpoints every 25 scenarios and resumes if restarted.
"""
import os
import sys
import numpy as np
import config
from multiprocessing import Pool
from core import simulate
from networks import load

NET, SEED = sys.argv[1], int(sys.argv[2])
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else config.WORKERS
PMIN, INSERV, A0, A1, AMAX = 20.0, 22.0, 2e-4, 4e-4, 2e-2
_wn = _J = _elev = _serv = None


def _init():
    global _wn, _J, _elev, _serv
    _wn = load(NET)
    _J = _wn.junction_name_list
    _elev = np.array([_wn.get_node(n).elevation for n in _J])
    h0, _ = simulate(_wn)
    _serv = (h0[_J].values - _elev) >= INSERV


def pressures(j, area):
    h, _ = simulate(_wn, {j: area})
    return h[_J].values - _elev


def _one(j):
    p0, pA0, pA1 = pressures(j, 1e-9), pressures(j, A0), pressures(j, A1)
    rec = dict(j=j, p0=p0, pA0=pA0, pA1=pA1)
    if (pressures(j, AMAX)[_serv] >= PMIN).all():
        return dict(rec, censored=True)
    lo, hi = (0.0 if (pA0[_serv] < PMIN).any() else A0), AMAX
    for _ in range(9):
        mid = (lo + hi) / 2
        if (pressures(j, mid)[_serv] < PMIN).any():
            hi = mid
        else:
            lo = mid
    ph = pressures(j, hi)
    fail = np.where(_serv & (ph < PMIN))[0]
    return dict(rec, censored=False, Astar=hi, first=int(fail[np.argmin(ph[fail])]))


if __name__ == "__main__":
    _init()
    rng = np.random.default_rng(SEED)
    cand = [n for n, s in zip(_J, _serv) if s]
    leaks = list(rng.choice(cand, NS, replace=False))
    part = f"sfp2_{NET}_s{SEED}_partial.npy"
    out = list(np.load(part, allow_pickle=True)) if os.path.exists(part) else []
    print(f"{NET} seed {SEED}: {NS} scenarios, resuming at {len(out)}, {WORKERS} workers", flush=True)
    with Pool(WORKERS, initializer=_init) as pool:
        for k, r in enumerate(pool.imap(_one, leaks[len(out):], chunksize=2), start=len(out)):
            out.append(r)
            if (k + 1) % 25 == 0:
                print(f"  {k+1}/{NS}", flush=True)
                np.save(part, np.array(out, dtype=object), allow_pickle=True)
    np.save(f"sfp2_{NET}_s{SEED}.npy", np.array(out, dtype=object), allow_pickle=True)
    print("saved; censored", sum(r["censored"] for r in out), flush=True)
