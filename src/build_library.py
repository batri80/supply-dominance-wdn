"""Simulate a single leak at every eligible junction and save the head-drop library.

Usage: python build_library.py ky4|synthetic [area_m2] [workers]
Output: lib_<net>_<area>.npz with candidates, junction names, and R[k] = head drop
for a leak at candidates[k], over all junctions.
"""
import sys
import time
import numpy as np
import config
from multiprocessing import Pool
from core import simulate
from networks import synthetic, ky4, load

NET = sys.argv[1]
AREA = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else config.WORKERS

_wn = _h0 = _J = None


def _init():
    global _wn, _h0, _J
    _wn = load(NET)
    _J = _wn.junction_name_list
    _h0, _ = simulate(_wn)


def _one(j):
    h, _ = simulate(_wn, {j: AREA})
    return (_h0 - h)[_J].values.astype(np.float32)


if __name__ == "__main__":
    wn = load(NET)
    J = wn.junction_name_list
    h0, _ = simulate(wn)
    elev = np.array([wn.get_node(n).elevation for n in J])
    cand = [n for n, p in zip(J, h0[J].values - elev) if p > 15]
    print(f"{NET}: {len(cand)} candidate leak sites, area {AREA:g} m2, {WORKERS} workers", flush=True)
    t = time.time()
    out = []
    with Pool(WORKERS, initializer=_init) as pool:
        for k, r in enumerate(pool.imap(_one, cand, chunksize=4)):
            out.append(r)
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(cand)} done, {time.time()-t:.0f}s", flush=True)
    np.savez(f"lib_{NET}_{AREA:g}.npz", cand=np.array(cand), J=np.array(J), R=np.array(out))
    print("saved", flush=True)
