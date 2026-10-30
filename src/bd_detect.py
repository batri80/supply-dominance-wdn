"""BattLeDIM step 2: causal detection + physics-prior localisation.

Detection (causal, daily): baseline = mean residual over [max(reset, t-60), t-7),
recent = mean over [t-7, t). Alarm when the drop statistic s = ||max(baseline-recent,0)||_2
exceeds THETA and the baseline holds at least 14 days. On alarm the baseline restarts.
At the start of an evaluation year, the offset of the first week against the leak-free
reference is decomposed by matching pursuit, so leaks already running are reported.

Localisation: the drop vector is compared (centred cosine) with the linearised
influence signatures of the nominal model at the 33 sensors. Two estimators:
  argmax  - best-scoring node
  medoid  - medoid (by pipe length) of all candidates within DELTA of the best score:
            the set the sensors cannot separate, reported by its centre so that the
            reported pipe is as close as possible to wherever in the set the leak is.
The reported pipe is the pipe incident to the chosen node with the higher-scoring far end.

Configuration is tuned on 2018 only (python bd_detect.py tune) and then applied once
to 2019 (python bd_detect.py test).
"""
import json
import os
import config
import sys
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import wntr

wn = wntr.network.WaterNetworkModel(config.LTOWN_INP)
J = wn.junction_name_list
idx = {n: k for k, n in enumerate(J)}
Gi = np.load("ltown_Ginv.npy")
SERIES = sys.argv[2] if len(sys.argv) > 2 else "daily"
R = pd.read_pickle("bd_resid.pkl")[SERIES]
S = list(R.columns)
sig = Gi[[idx[s] for s in S], :]                  # sensors x candidate nodes
sigc = sig - sig.mean(0, keepdims=True)
sign = sigc / (np.linalg.norm(sigc, axis=0, keepdims=True) + 1e-12)
G = nx.Graph()
for name, l in wn.links():
    G.add_edge(l.start_node_name, l.end_node_name, w=getattr(l, "length", 0.0), link=name)
_dist = {}


def dist_from(n):
    if n not in _dist:
        _dist[n] = nx.single_source_dijkstra_path_length(G, n, weight="w")
    return _dist[n]


def scores(drop):
    o = drop - drop.mean()
    return (o / (np.linalg.norm(o) + 1e-12)) @ sign


def choose_node(sc, method, delta):
    best = int(np.argmax(sc))
    if method == "argmax":
        return J[best]
    cand = [J[k] for k in np.where(sc >= sc[best] - delta)[0]]
    if len(cand) > 150:                           # keep the medoid search bounded
        cand = [J[k] for k in np.argsort(-sc)[:150]]
    return min(cand, key=lambda a: sum(dist_from(a).get(b, 1e9) for b in cand))


def node_to_pipe(node, sc):
    best, bv = None, -np.inf
    for nb in G.neighbors(node):
        link = G[node][nb]["link"]
        if wn.get_link(link).link_type != "Pipe":
            continue
        v = sc[idx[nb]] if nb in idx else -np.inf
        if v > bv:
            best, bv = link, v
    return best


def pursuit(drop, k_max, min_gain):
    """Greedy decomposition of a drop vector into up to k_max single-leak signatures."""
    out, resid = [], drop.copy()
    base = np.linalg.norm(resid)
    for _ in range(k_max):
        sc = scores(resid)
        j = int(np.argmax(sc))
        a = sig[:, j]
        amp = max(resid @ a / (a @ a), 0)
        new = resid - amp * a
        if np.linalg.norm(resid) - np.linalg.norm(new) < min_gain * base:
            break
        out.append(sc.copy())
        # keep the candidate's own score vector for localisation, remove its share
        resid = new
    return out


def run(year, cfg):
    theta, delta, method = cfg["theta"], cfg["delta"], cfg["method"]
    r = R.loc[str(year)]
    days = r.index
    dets = []
    # leaks already present at the start of the year (vs the leak-free reference)
    if year == 2019:
        first = r.iloc[:7].mean().values
        for sc in pursuit(-first, cfg["k0"], cfg["min_gain"]):
            node = choose_node(sc, method, delta)
            dets.append((node_to_pipe(node, sc), days[7] - pd.Timedelta(minutes=5)))
        reset = 0
    else:
        reset = 0
    t = 21
    while t < len(days):
        lo = max(reset, t - 60)
        if t - 7 - lo >= 14:
            base = r.iloc[lo:t - 7].mean().values
            rec = r.iloc[t - 7:t].mean().values
            drop = base - rec
            s = np.linalg.norm(np.maximum(drop, 0))
            rise = np.linalg.norm(np.maximum(-drop, 0))
            if cfg.get("reset_on_repair") and rise > theta and rise > s:
                reset = t - 7                     # a repair: restart the baseline, no alarm
                t += 1
                continue
            if s > theta:
                sc = scores(drop)
                node = choose_node(sc, method, delta)
                stamp = days[t] - pd.Timedelta(minutes=5)   # end of the last day used
                dets.append((node_to_pipe(node, sc), stamp))
                reset = t - 7
                t += 7
                continue
        t += 1
    return dets


LEAKS18 = [("p257", "2018-01-08 13:30", "2018-12-31 23:55"), ("p461", "2018-01-23 04:25", "2018-04-02 11:40"),
           ("p232", "2018-01-31 02:35", "2018-02-10 09:20"), ("p427", "2018-02-13 08:25", "2018-12-31 23:55"),
           ("p673", "2018-03-05 15:45", "2018-03-23 10:25"), ("p810", "2018-07-28 03:05", "2018-12-31 23:55"),
           ("p628", "2018-05-02 14:55", "2018-05-29 21:20"), ("p538", "2018-05-18 08:35", "2018-06-02 06:05"),
           ("p866", "2018-06-01 09:05", "2018-06-12 03:00"), ("p31", "2018-06-28 10:35", "2018-08-12 17:30"),
           ("p654", "2018-07-05 03:40", "2018-12-31 23:55"), ("p183", "2018-08-07 02:35", "2018-09-01 17:10"),
           ("p158", "2018-10-06 02:35", "2018-10-23 13:35"), ("p369", "2018-10-26 02:05", "2018-11-08 20:25")]


def pipe_dist(p, q):
    if p == q:
        return 0.0
    lp, lq = wn.get_link(p), wn.get_link(q)
    b = min(dist_from(a).get(c, np.inf) for a in (lp.start_node_name, lp.end_node_name)
            for c in (lq.start_node_name, lq.end_node_name))
    return b + getattr(lp, "length", 0) / 2 + getattr(lq, "length", 0) / 2


def proxy_score_2018(dets):
    """TP/FP with the official space-time rule (300 m, inside the leak window), no water volumes."""
    done, tp, fp, dl = set(), 0, 0, []
    for link, t in sorted(dets, key=lambda d: d[1]):
        hits = [k for k, (lk, a, b) in enumerate(LEAKS18)
                if pd.Timestamp(a) <= t <= pd.Timestamp(b) and pipe_dist(lk, link) <= 300]
        new = [k for k in hits if k not in done]
        if hits and new:
            k = min(new, key=lambda k: pipe_dist(LEAKS18[k][0], link))
            done.add(k); tp += 1; dl.append(pipe_dist(LEAKS18[k][0], link))
        elif not hits:
            fp += 1
    return dict(TP=tp, FP=fp, FN=len(LEAKS18) - tp, utility=tp - 0.5 * fp)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tune":
        grid = []
        for theta, delta, method in itertools.product((0.05, 0.08, 0.12, 0.16, 0.2, 0.3),
                                                      (0.002, 0.005, 0.01), ("argmax", "medoid")):
            if method == "argmax" and delta != 0.005:
                continue
            for rr in (False, True):
                cfg = dict(theta=theta, delta=delta, method=method, k0=4, min_gain=0.05,
                           reset_on_repair=rr, series=SERIES)
                ps = proxy_score_2018(run(2018, cfg))
                grid.append(dict(cfg=cfg, **ps))
                print(cfg, ps, flush=True)
        json.dump(grid, open(f"bd_tune_2018_{SERIES}.json", "w"), indent=1)
    else:
        from battledim_score import Scorer
        cfgs = json.load(open(os.path.join(config.SRC, "bd_selected.json")))
        sc = Scorer()
        out = {}
        for name, cfg in cfgs.items():
            dets = run(2019, cfg)
            res = sc.score([(l, t.strftime("%Y-%m-%d %H:%M")) for l, t in dets])
            out[name] = dict(cfg=cfg, **{k: v for k, v in res.items() if k != "rows"}, rows=res["rows"])
            print(name, cfg, {k: res[k] for k in ("total_score", "TP", "FP", "FN")}, flush=True)
        json.dump(out, open("bd_results_2019.json", "w"), indent=1, default=str)
