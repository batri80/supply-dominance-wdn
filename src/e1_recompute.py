"""Recompute Test 1 from the saved leak responses with the corrected linearisation
(zero-flow pipes stiff, closed pumps open-circuit, power pumps by their own law)."""
import json
import sys
import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from core import simulate, influence_matrix
from networks import synthetic, ky4

net = sys.argv[1]
wn = synthetic() if net == "synthetic" else ky4()
J = wn.junction_name_list
h0, q0 = simulate(wn)
Gi, idx = influence_matrix(wn, h0, q0, J)
Gu = nx.Graph([(l.start_node_name, l.end_node_name) for _, l in wn.links()])
d = np.load(f"resp_{net}.npz")
L, R = list(d["leaks"]), d["R"]
rho_hop, rho_lin, r2 = [], [], []
for k, j in enumerate(L):
    dh, jj = R[k], idx[j]
    o = np.array([i for i in range(len(J)) if i != jj])
    hop = nx.single_source_shortest_path_length(Gu, j)
    hops = np.array([hop.get(J[i], np.nan) for i in o])
    ok = np.isfinite(hops)
    rho_hop.append(spearmanr(hops[ok], dh[o][ok]).correlation)
    rho_lin.append(spearmanr(Gi[o, jj], dh[o]).correlation)
    pred = Gi[:, jj] / Gi[jj, jj] * dh[jj]
    r2.append(1 - np.sum((dh - pred) ** 2) / np.sum((dh - dh.mean()) ** 2))
r2 = np.array(r2)
res = json.load(open(f"results_{net}.json"))
res["E1"] = dict(spearman_hop_median=float(np.nanmedian(rho_hop)),
                 spearman_linear_influence_median=float(np.nanmedian(rho_lin)),
                 frac_leaks_linear_beats_hop=float(np.mean(np.abs(rho_lin) > np.abs(rho_hop))),
                 linear_profile_R2_median=float(np.median(r2)),
                 linear_profile_R2_q10=float(np.percentile(r2, 10)),
                 frac_R2_below_0p9=float(np.mean(r2 < 0.9)))
json.dump(res, open(f"results_{net}.json", "w"), indent=2)
print(net, res["E1"])
