"""Revision experiment R1 (rev_protocol.md): robustness of NEET-2PT to model mismatch.

For each leak junction, a "true" network is built for every condition (perturbed
roughness, demands, pressure-dependent-demand parameters, discharge coefficient, or a
later demand level), and the leak area at first service failure A* is found on it.
The nominal model's responses at A0 and A1 (the deployment route) are stored once, in
the nominal condition, together with the responses at a wrong junction 1-3 hops away.

Usage: python sfp3_sim.py ltown|ky4 [n_scen=100] [workers]
Writes rev_r1_<net>.npy; checkpoints every 25 tasks and resumes if restarted.
"""
import copy
import os
import sys
import numpy as np
import networkx as nx
import config
import wntr
from multiprocessing import Pool
from networks import load

NET = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else config.WORKERS
PMIN, INSERV, A0, A1, AMAX = 20.0, 22.0, 2e-4, 4e-4, 2e-2
SEED = {"ltown": 6026, "ky4": 6027}[NET]
CONDITIONS = ["nominal", "roughness", "demand", "pdd", "cd", "combined", "drift"]
OUTFILE = f"rev_r1_{NET}.npy"
_wn = _J = _elev = None


def _init():
    global _wn, _J, _elev
    _wn = load(NET)
    _J = _wn.junction_name_list
    _elev = np.array([_wn.get_node(n).elevation for n in _J])


def params(cond, k):
    """Perturbation of the true network for scenario k (drawn from seed 7000 + k)."""
    r = np.random.default_rng(7000 + k)
    p = dict(rough=None, dem=None, mult=1.0, req=20.0, minp=0.0, cd=0.75, fail_mult=None)
    draws = dict(rough=r.normal(0, 0.15, 100000), mult=r.uniform(0.8, 1.2), dem=r.normal(0, 0.2, 100000),
                 req=r.uniform(15, 30), minp=r.uniform(0, 5), cd=r.uniform(0.6, 0.9), drift=r.uniform(1.1, 1.3))
    if cond in ("roughness", "combined"):
        p["rough"] = np.exp(draws["rough"])
    if cond in ("demand", "combined"):
        p["dem"] = np.exp(draws["dem"]); p["mult"] = draws["mult"]
    if cond in ("pdd", "combined"):
        p["req"], p["minp"] = draws["req"], draws["minp"]
    if cond in ("cd", "combined"):
        p["cd"] = draws["cd"]
    if cond == "drift":
        p["fail_mult"] = draws["drift"]
    return p


def true_network(p):
    w = copy.deepcopy(_wn)
    if p["rough"] is not None:
        for k, (_, pipe) in enumerate(w.pipes()):
            pipe.roughness = pipe.roughness * p["rough"][k]
    if p["dem"] is not None:
        for k, n in enumerate(_J):
            for d in w.get_node(n).demand_timeseries_list:
                d.base_value = d.base_value * p["dem"][k]
    w.options.time.duration = 0
    w.options.time.hydraulic_timestep = 3600
    w.options.hydraulic.demand_model = "PDD"
    w.options.hydraulic.required_pressure = p["req"]
    w.options.hydraulic.minimum_pressure = p["minp"]
    for name in list(w.control_name_list):
        w.remove_control(name)
    return w


def pressures(w, j, area, cd, mult):
    x = copy.deepcopy(w)
    x.options.hydraulic.demand_multiplier = mult
    x.get_node(j).add_leak(x, area=area, discharge_coeff=cd, start_time=0)
    h = wntr.sim.WNTRSimulator(x).run_sim().node["head"].iloc[0]
    return (h[_J].values - _elev).astype(np.float32)


def critical(w, j, cd, mult):
    """A* and first failing junction on network w at demand multiplier mult."""
    p0 = pressures(w, j, 1e-9, cd, mult)
    serv = p0 >= INSERV
    if (pressures(w, j, AMAX, cd, mult)[serv] >= PMIN).all():
        return dict(censored=True, serv_true=serv)
    lo, hi = (0.0 if (pressures(w, j, A0, cd, mult)[serv] < PMIN).any() else A0), AMAX
    for _ in range(9):
        mid = (lo + hi) / 2
        if (pressures(w, j, mid, cd, mult)[serv] < PMIN).any():
            hi = mid
        else:
            lo = mid
    ph = pressures(w, j, hi, cd, mult)
    fail = np.where(serv & (ph < PMIN))[0]
    return dict(censored=False, Astar=hi, first=int(fail[np.argmin(ph[fail])]), serv_true=serv)


def _one(task):
    cond, k, j, jw = task
    p = params(cond, k)
    w = true_network(p)
    rec = dict(cond=cond, k=k, j=j, params={a: (float(np.mean(v)) if isinstance(v, np.ndarray) else v)
                                            for a, v in p.items() if v is not None})
    if cond != "drift":                       # responses observed on the true network
        rec.update(p0=pressures(w, j, 1e-9, p["cd"], p["mult"]), pA0=pressures(w, j, A0, p["cd"], p["mult"]),
                   pA1=pressures(w, j, A1, p["cd"], p["mult"]))
    if cond == "nominal":                     # model responses at a wrong junction
        rec.update(jw=jw, pA0w=pressures(w, jw, A0, 0.75, 1.0), pA1w=pressures(w, jw, A1, 0.75, 1.0))
    rec.update(critical(w, j, p["cd"], p["fail_mult"] or p["mult"]))
    return rec


if __name__ == "__main__":
    _init()
    from core import simulate
    h0, _ = simulate(_wn)
    serv0 = (h0[_J].values - _elev) >= INSERV
    cand = [n for n, s in zip(_J, serv0) if s]
    rng = np.random.default_rng(SEED)
    leaks = list(rng.choice(cand, NS, replace=False))
    Gu = nx.Graph([(l.start_node_name, l.end_node_name) for _, l in _wn.links()])
    cset = set(cand)
    wrong = []
    for j in leaks:
        d = nx.single_source_shortest_path_length(Gu, j, cutoff=3)
        near = sorted(n for n, h in d.items() if 1 <= h <= 3 and n in cset)
        wrong.append(near[rng.integers(len(near))] if near else j)
    tasks = [(c, k, j, wrong[k]) for c in CONDITIONS for k, j in enumerate(leaks)]
    part = OUTFILE.replace(".npy", "_partial.npy")
    out = list(np.load(part, allow_pickle=True)) if os.path.exists(part) else []
    print(f"{NET}: {NS} leak junctions x {len(CONDITIONS)} conditions = {len(tasks)} tasks, "
          f"resuming at {len(out)}, {WORKERS} workers", flush=True)
    with Pool(WORKERS, initializer=_init) as pool:
        for i, r in enumerate(pool.imap(_one, tasks[len(out):], chunksize=1), start=len(out)):
            out.append(r)
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)}  ({r['cond']})", flush=True)
                np.save(part, np.array(out, dtype=object), allow_pickle=True)
    np.save(OUTFILE, np.array(out, dtype=object), allow_pickle=True)
    for c in CONDITIONS:
        print(f"  {c:9s} censored {sum(r['censored'] for r in out if r['cond'] == c)}/{NS}", flush=True)
    print("saved", flush=True)
