"""Explore 2018 (training year): are leak onsets visible, and does the linear
influence of the nominal model point at the right place?"""
import numpy as np
import config  # noqa: F401
import pandas as pd
import networkx as nx
import wntr
from core import simulate, influence_matrix

wn = wntr.network.WaterNetworkModel(config.LTOWN_INP)
J = wn.junction_name_list
h0, q0 = simulate(wn)
Gi, idx = influence_matrix(wn, h0, q0, J)
np.save("ltown_Ginv.npy", Gi)
r = pd.read_pickle("bd_resid.pkl")["daily"]
S = list(r.columns)
sidx = [idx[s] for s in S]
G = nx.Graph()
for _, l in wn.links():
    G.add_edge(l.start_node_name, l.end_node_name, w=getattr(l, "length", 0.0))

leaks18 = [("p461", "2018-01-23"), ("p232", "2018-01-31"), ("p673", "2018-03-05"), ("p628", "2018-05-02"),
           ("p538", "2018-05-18"), ("p866", "2018-06-01"), ("p31", "2018-06-28"), ("p654", "2018-07-05"),
           ("p810", "2018-07-28"), ("p183", "2018-08-07"), ("p158", "2018-10-06"), ("p369", "2018-10-26")]
sig = Gi[np.ix_(sidx, range(len(J)))]            # sensors x candidates
sigc = sig - sig.mean(0, keepdims=True)
sign = sigc / (np.linalg.norm(sigc, axis=0, keepdims=True) + 1e-12)
for link, day in leaks18:
    d = pd.Timestamp(day)
    before = r.loc[d - pd.Timedelta(days=14): d - pd.Timedelta(days=1)].mean()
    after = r.loc[d + pd.Timedelta(days=1): d + pd.Timedelta(days=7)].mean()
    dr = (after - before).values                   # negative near a leak
    obs = -dr - (-dr).mean()
    score = (obs / (np.linalg.norm(obs) + 1e-12)) @ sign
    best = J[int(np.argmax(score))]
    l = wn.get_link(link)
    dist = min(nx.dijkstra_path_length(G, best, n, weight="w") for n in (l.start_node_name, l.end_node_name))
    rank_true = int((score > max(score[idx[l.start_node_name]], score[idx[l.end_node_name]])).sum())
    print(f"{link:5s} {day}  max|dr| {np.abs(dr).max():.3f} m  min dr {dr.min():+.3f}  best node {best:5s}  dist {dist:6.0f} m  rank of true {rank_true}")
