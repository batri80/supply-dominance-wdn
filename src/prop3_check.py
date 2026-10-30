"""Numerical check of the isolability limit (Proposition 3) on the KY4 leak library.
For the ten largest shadows, with sensors placed at every shadow node: the spread of
the residual across those sensors (norm after removing the mean, over the raw norm)
for leaks outside the shadow (theory: exactly 0) and inside it (theory: > 0)."""
import json
import numpy as np
import networkx as nx
from networks import ky4

wn = ky4(); d = np.load("lib_ky4_0.0001.npz")
J, C, R = list(d["J"]), list(d["cand"]), d["R"].astype(float)
jidx = {n: k for k, n in enumerate(J)}
G = nx.Graph([(l.start_node_name, l.end_node_name) for _, l in wn.links()])
F = set(wn.reservoir_name_list) | set(wn.tank_name_list)


def sh(j):
    H = G.copy(); H.remove_node(j); r = set()
    for f in F:
        if f in H:
            r |= nx.node_connected_component(H, f)
    return [n for n in J if n != j and n not in r]


rows = []
for size, k, S in sorted(((len(s), j, s) for j in J for s in [sh(j)] if len(s) >= 4), reverse=True)[:10]:
    si = [jidx[n] for n in S]
    spread = lambda X: np.linalg.norm(X - X.mean(1, keepdims=True), axis=1) / (np.linalg.norm(X, axis=1) + 1e-12)
    out = spread(R[np.ix_([i for i, c in enumerate(C) if c not in S and c != k], si)])
    ins = spread(R[np.ix_([i for i, c in enumerate(C) if c in S], si)])
    rows.append(dict(entry=k, size=size, outside_max=float(out.max()), outside_median=float(np.median(out)),
                     inside_median=float(np.median(ins)), inside_min=float(ins.min())))
json.dump(rows, open("prop3_check_ky4.json", "w"), indent=1)
print(json.dumps(rows, indent=1))
