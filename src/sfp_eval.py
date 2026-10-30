"""Evaluate service-failure predictors exactly as in sfp_protocol.md.
Usage: python sfp_eval.py ltown|ky4
"""
import json
import sys
import numpy as np
import networkx as nx
from scipy.stats import spearmanr, rankdata, pearsonr
from core import simulate, influence_matrix
from networks import load

NET = sys.argv[1]
PMIN, INSERV, A0 = 20.0, 22.0, 2e-4
wn = load(NET)
J = wn.junction_name_list
idx = {n: k for k, n in enumerate(J)}
elev = np.array([wn.get_node(n).elevation for n in J])
h0, q0 = simulate(wn)
serv = (h0[J].values - elev) >= INSERV
Gi, _ = influence_matrix(wn, h0, q0, J)
G = nx.Graph()
for _, l in wn.links():
    G.add_edge(l.start_node_name, l.end_node_name, w=getattr(l, "length", 0.0) or 0.0)
_d = {}


def dist(a, b):
    if a not in _d:
        _d[a] = nx.single_source_dijkstra_path_length(G, a, weight="w")
    return _d[a].get(b, np.inf)


S = list(np.load(f"sfp_{NET}.npy", allow_pickle=True))
done = [s for s in S if not s["censored"]]
sv = np.where(serv)[0]


def area_from_ratio(r, pj0, dhj0):
    pj = pj0 - dhj0 * (r - 1)
    return np.where(pj > 0, A0 * r * np.sqrt(pj0 / np.maximum(pj, 1e-9)), np.inf)


node_hits = {k: [] for k in ("MARGIN", "NEAREST", "DROP", "NEET", "NEET_LIN")}
node_top5 = {k: [] for k in node_hits}
node_dist = {k: [] for k in node_hits}
size_pred = {k: [] for k in ("AT_LEAK", "MARGIN_MIN", "NEET")}
astar, psi, total = [], [], []
for s in done:
    j = idx[s["j"]]
    p0, pA0 = s["p0"], s["pA0"]
    dh = np.maximum(p0 - pA0, 1e-9)
    m = pA0 - PMIN
    dh_lin = np.maximum(Gi[:, j] / Gi[j, j] * dh[j], 1e-9)
    scores = {
        "MARGIN": -m,
        "NEAREST": -np.array([dist(s["j"], n) for n in J]),
        "DROP": dh,
        "NEET": dh / np.maximum(m, 1e-6),
        "NEET_LIN": dh_lin / np.maximum(m, 1e-6),
    }
    true = s["first"]
    for k, sc in scores.items():
        order = sv[np.argsort(-sc[sv])]
        node_hits[k].append(float(order[0] == true))
        node_top5[k].append(float(true in order[:5]))
        node_dist[k].append(dist(J[order[0]], J[true]))
    r = 1 + np.maximum(m, 0) / dh
    A_all = area_from_ratio(r, pA0[j], dh[j])
    size_pred["AT_LEAK"].append(float(A_all[j]))
    size_pred["MARGIN_MIN"].append(float(A_all[sv[np.argmin(m[sv])]]))
    size_pred["NEET"].append(float(A_all[sv].min()))
    astar.append(s["Astar"])
    d = (dh / np.maximum(m, 1e-6))[sv]
    total.append(d.mean()); psi.append(d.std() / d.mean())

astar = np.array(astar)
res = dict(network=NET, scenarios=len(S), censored=len(S) - len(done), node={}, size={})
for k in node_hits:
    res["node"][k] = dict(top1=float(np.mean(node_hits[k])), top5=float(np.mean(node_top5[k])),
                          median_dist_m=float(np.median(node_dist[k])))
for k, v in size_pred.items():
    v = np.array(v)
    rel = np.abs(v - astar) / astar
    res["size"][k] = dict(median_rel_error=float(np.median(rel)), spearman=float(spearmanr(v, astar).correlation))
# partial Spearman of A* with PSI_N controlling for TOTAL
ra, rp, rt = rankdata(astar), rankdata(psi), rankdata(total)
resid = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)
pr, pp = pearsonr(resid(ra, rt), resid(rp, rt))
res["disequilibrium"] = dict(partial_spearman_psi=float(pr), p=float(pp),
                             spearman_total=float(spearmanr(total, astar).correlation),
                             spearman_psi=float(spearmanr(psi, astar).correlation))
json.dump(res, open(f"sfp_results_{NET}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
