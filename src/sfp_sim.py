"""Service-failure prediction scenarios (see sfp_protocol.md).

For each scenario: leak node j; simulate the early state at A0 = 2 cm^2; then bisect
for the critical area A* where the first in-service node drops below 20 m.
Usage: python sfp_sim.py ltown|ky4 [n_scen] [workers]
"""
import sys
import numpy as np
import config
from multiprocessing import Pool
from core import simulate
from networks import load

NET = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else config.WORKERS
PMIN, INSERV, A0, AMAX = 20.0, 22.0, 2e-4, 2e-2
SEED = {"ltown": 2026, "ky4": 2027}[NET]
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
    p0 = pressures(j, 1e-9)
    pA0 = pressures(j, A0)
    pmax = pressures(j, AMAX)
    if (pmax[_serv] >= PMIN).all():
        return dict(j=j, censored=True, p0=p0, pA0=pA0)
    lo, hi = A0, AMAX
    if (pA0[_serv] < PMIN).any():
        lo = 0.0
    for _ in range(9):
        mid = (lo + hi) / 2
        if (pressures(j, mid)[_serv] < PMIN).any():
            hi = mid
        else:
            lo = mid
    ph = pressures(j, hi)
    fail = np.where(_serv & (ph < PMIN))[0]
    first = int(fail[np.argmin(ph[fail])])
    return dict(j=j, censored=False, Astar=hi, first=first, p0=p0, pA0=pA0)


if __name__ == "__main__":
    _init()
    rng = np.random.default_rng(SEED)
    cand = [n for n, s in zip(_J, _serv) if s]
    leaks = list(rng.choice(cand, NS, replace=False))
    import os
    part = f"sfp_{NET}_partial.npy"
    out = list(np.load(part, allow_pickle=True)) if os.path.exists(part) else []
    todo = leaks[len(out):]                         # same seed, same order: resume where it stopped
    print(f"resuming at {len(out)}/{NS}", flush=True)
    with Pool(WORKERS, initializer=_init) as pool:
        for k, r in enumerate(pool.imap(_one, todo, chunksize=2), start=len(out)):
            out.append(r)
            if (k + 1) % 25 == 0:
                print(f"  {k+1}/{NS}", flush=True)
                np.save(part, np.array(out, dtype=object), allow_pickle=True)
    np.save(f"sfp_{NET}.npy", np.array(out, dtype=object), allow_pickle=True)
    print("saved; censored", sum(r["censored"] for r in out), flush=True)
