"""Sensor layouts for experiment X1 (see ext_protocol.md).

L-Town: the utility's 33 pressure sensors.
KY4: 48 sensors (5%) by farthest-point spreading in hop distance, from the network alone.
Writes ext_sensors.json.
"""
import json
import networkx as nx
import config  # noqa: F401
from networks import ky4

LTOWN = ["n1", "n4", "n31", "n54", "n105", "n114", "n163", "n188", "n215", "n229", "n288", "n296", "n332", "n342",
         "n410", "n415", "n429", "n458", "n469", "n495", "n506", "n516", "n519", "n549", "n613", "n636", "n644",
         "n679", "n722", "n726", "n740", "n752", "n769"]


def farthest_point(wn, m):
    J = wn.junction_name_list
    G = nx.Graph([(l.start_node_name, l.end_node_name) for _, l in wn.links()])
    hop = {}

    def d(n):
        if n not in hop:
            hop[n] = nx.single_source_shortest_path_length(G, n)
        return hop[n]

    chosen = [max(J, key=lambda n: d(J[0]).get(n, 0))]          # start: the junction farthest from the first
    best = {n: d(chosen[0]).get(n, 10 ** 9) for n in J}
    while len(chosen) < m:
        nxt = max(J, key=lambda n: (best[n], n))                 # ties broken by name for reproducibility
        chosen.append(nxt)
        dn = d(nxt)
        for n in J:
            best[n] = min(best[n], dn.get(n, 10 ** 9))
    return chosen


layouts = {"ltown": LTOWN, "ky4": farthest_point(ky4(), 48)}
json.dump(layouts, open("ext_sensors.json", "w"), indent=1)
for k, v in layouts.items():
    print(f"{k}: {len(v)} sensors, first five {v[:5]}")
