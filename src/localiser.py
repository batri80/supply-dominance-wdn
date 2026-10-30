"""Hybrid localiser on simulated L-Town scenarios, at the utility's 33 sensors.

Methods compared on the same held-out scenarios (seed-1 set, never used in training):
  nearest_sensor  - the sensor with the largest drop (naive baseline)
  graph_interp    - harmonic interpolation of sensor drops over the pipe graph
                    (the graph-based state interpolation family, e.g. Romero-Ben et al.)
  linear          - angle method against the linearised influence matrix (physics prior)
  gnn_plain       - graph network on sensor drops only (no physics prior)
  hybrid          - graph network given the linear scores and dominance flags as node
                    features, trained to correct the prior
  hybrid_nodom    - ablation: hybrid without dominance flags

The graph network: 4 rounds of conductance-weighted message passing with separate
weights for self and neighbour terms, node-level softmax output. Trained on the
seed-0 scenarios (single and concurrent leaks, 1-10 cm^2, demand 0.6-1.4x).
Metrics: top-1 / top-5 node accuracy, pipe-length distance from estimate to the
nearest true leak node, and share within 300 m (the BattLeDIM radius); reported
for single vs concurrent leaks and small (<= 4 cm^2) vs large leaks.
"""
import json
import numpy as np
import networkx as nx
import contextlib
import io
with contextlib.redirect_stderr(io.StringIO()):   # silence the NumPy 1.x/2.x notice from torch 2.2 on Intel Macs
    import torch
    import torch.nn as nn
from core import simulate, influence_matrix
from networks import ltown

torch.manual_seed(0)


def to_t(a):
    """NumPy -> torch without the NumPy bridge (torch 2.2 on Intel Macs cannot use it with NumPy 2)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(a.shape)


def to_np(t):
    """torch -> NumPy without the NumPy bridge."""
    return np.array(t.detach().tolist(), dtype=np.float32)

torch.set_num_threads(2)
SIGMA = 0.01
SENS = ["n1", "n4", "n31", "n54", "n105", "n114", "n163", "n188", "n215", "n229", "n288", "n296", "n332", "n342",
        "n410", "n415", "n429", "n458", "n469", "n495", "n506", "n516", "n519", "n549", "n613", "n636", "n644",
        "n679", "n722", "n726", "n740", "n752", "n769"]

wn = ltown()
J = wn.junction_name_list
idx = {n: k for k, n in enumerate(J)}
nJ = len(J)
s_idx = np.array([idx[s] for s in SENS])
h0, q0 = simulate(wn)
Gi, _ = influence_matrix(wn, h0, q0, J)
sig = Gi[s_idx, :]
sigc = sig - sig.mean(0, keepdims=True)
sign = sigc / (np.linalg.norm(sigc, axis=0, keepdims=True) + 1e-12)

# graph, conductances, distances
Gw = nx.Graph()
edges, cond = [], []
for name, l in wn.links():
    a, b = l.start_node_name, l.end_node_name
    Gw.add_edge(a, b, w=getattr(l, "length", 0.0))
    if a in idx and b in idx:
        dq = abs(q0[name]); dh = abs(h0[a] - h0[b])
        g = 0.1 if dh < 1e-9 else min(dq / (1.852 * dh), 0.1)
        edges += [(idx[a], idx[b]), (idx[b], idx[a])]
        cond += [float(g), float(g)]
E = torch.tensor(edges, dtype=torch.long).T
w = torch.tensor(cond, dtype=torch.float32)
w = w / (torch.zeros(nJ).index_add_(0, E[0], w)[E[0]] + 1e-9)        # row-normalised
A = torch.sparse_coo_tensor(E, w, (nJ, nJ)).coalesce()

# dominance flags: in a shadow, and shadow-entry node
fixed = set(wn.reservoir_name_list) | set(wn.tank_name_list)
in_sh = np.zeros(nJ); entry = np.zeros(nJ)
for j in J:
    H = Gw.copy(); H.remove_node(j)
    reach = set()
    for f in fixed:
        if f in H:
            reach |= nx.node_connected_component(H, f)
    sh = [n for n in J if n != j and n not in reach]
    if sh:
        entry[idx[j]] = 1
        for n in sh:
            in_sh[idx[n]] = 1
dist_cache = {}


def dist(a, b):
    if a not in dist_cache:
        dist_cache[a] = nx.single_source_dijkstra_path_length(Gw, a, weight="w")
    return dist_cache[a].get(b, np.inf)


# harmonic interpolation operator for graph_interp: solve L x = 0 off sensors
L = nx.laplacian_matrix(Gw.subgraph(J), nodelist=J).astype(float).toarray()
free = np.setdiff1d(np.arange(nJ), s_idx)
Lff, Lfs = L[np.ix_(free, free)], L[np.ix_(free, s_idx)]
interp = -np.linalg.solve(Lff + 1e-9 * np.eye(len(free)), Lfs)      # free x sensors


def load(seed):
    d = np.load(f"train_ltown_s{seed}.npz")
    rng = np.random.default_rng(100 + seed)
    obs = d["R"][:, s_idx] + rng.normal(0, SIGMA, (len(d["R"]), len(SENS)))
    return obs.astype(np.float32), d["nodes"], d["areas"]


def linear_scores(obs):
    o = obs - obs.mean(1, keepdims=True)
    o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    return o @ sign                                                   # scenarios x nJ


def node_features(obs, use_prior=True, use_dom=True):
    n = len(obs)
    X = np.zeros((n, nJ, 5), dtype=np.float32)
    X[:, s_idx, 0] = obs / (np.abs(obs).max(1, keepdims=True) + 1e-9)  # scaled drops at sensors
    X[:, s_idx, 1] = 1.0                                               # sensor mask
    if use_prior:
        X[:, :, 2] = linear_scores(obs)
    if use_dom:
        X[:, :, 3] = in_sh; X[:, :, 4] = entry
    return to_t(X)


class GNN(nn.Module):
    def __init__(self, fin=5, h=64, layers=4):
        super().__init__()
        self.inp = nn.Linear(fin, h)
        self.self_w = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])
        self.nb_w = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(layers)])
        self.out = nn.Linear(h + fin, 1)

    def forward(self, X):                       # X: batch x nJ x fin
        h = torch.relu(self.inp(X))
        for s, nb in zip(self.self_w, self.nb_w):
            m = torch.stack([torch.sparse.mm(A, hb) for hb in h])       # conductance-weighted mean
            h = torch.relu(s(h) + nb(m)) + h
        return self.out(torch.cat([h, X], -1)).squeeze(-1)             # logits per node


def targets(nodes):
    T = np.zeros((len(nodes), nJ), dtype=np.float32)
    for i, ns in enumerate(nodes):
        for n in ns:
            if n:
                T[i, idx[n]] = 1.0
    return to_t(T / T.sum(1, keepdims=True))


def train(Xtr, Ttr, epochs=30):
    m = GNN(fin=Xtr.shape[-1])
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0
        for b in range(0, n, 32):
            ib = perm[b:b + 32]
            logp = torch.log_softmax(m(Xtr[ib]), -1)
            loss = -(Ttr[ib] * logp).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(ib)
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}: loss {tot/n:.3f}", flush=True)
    return m


def evaluate(score, nodes, areas):
    """score: scenarios x nJ (higher = more likely)."""
    res = []
    for i in range(len(score)):
        true = [n for n in nodes[i] if n]
        order = np.argsort(-score[i])
        est = J[order[0]]
        d = min(dist(est, t) for t in true)
        res.append(dict(top1=float(est in true), top5=float(any(J[k] in true for k in order[:5])),
                        dist=d, within300=float(d <= 300), concurrent=len(true) > 1,
                        large=float(max(areas[i])) > 4e-4))
    return res


def summarise(res):
    out = {}
    groups = {"all": res, "single": [r for r in res if not r["concurrent"]], "concurrent": [r for r in res if r["concurrent"]],
              "small": [r for r in res if not r["large"]], "large": [r for r in res if r["large"]]}
    for g, rs in groups.items():
        if rs:
            out[g] = dict(n=len(rs), top1=np.mean([r["top1"] for r in rs]), top5=np.mean([r["top5"] for r in rs]),
                          median_dist_m=float(np.median([r["dist"] for r in rs])),
                          within300=np.mean([r["within300"] for r in rs]))
    return out


if __name__ == "__main__":
    obs_tr, nodes_tr, areas_tr = load(0)
    obs_te, nodes_te, areas_te = load(1)
    Ttr = targets(nodes_tr)
    results = {}
    # non-learned methods
    results["nearest_sensor"] = summarise(evaluate(np.eye(nJ)[s_idx][np.argmax(obs_te, 1)] , nodes_te, areas_te))
    full = np.zeros((len(obs_te), nJ)); full[:, s_idx] = obs_te; full[:, free] = obs_te @ interp.T
    results["graph_interp"] = summarise(evaluate(full, nodes_te, areas_te))
    results["linear"] = summarise(evaluate(linear_scores(obs_te), nodes_te, areas_te))
    for name, prior, dom in (("gnn_plain", False, False), ("hybrid", True, True), ("hybrid_nodom", True, False)):
        print("training", name, flush=True)
        m = train(node_features(obs_tr, prior, dom), Ttr)
        with torch.no_grad():
            sc = to_np(torch.cat([m(node_features(obs_te[b:b + 64], prior, dom)) for b in range(0, len(obs_te), 64)]))
        results[name] = summarise(evaluate(sc, nodes_te, areas_te))
    for k, v in results.items():
        a = v["all"]
        print(f"{k:15s} top1 {a['top1']:.3f} top5 {a['top5']:.3f} median {a['median_dist_m']:.0f} m  <=300m {a['within300']:.3f}"
              f" | single {v['single']['within300']:.3f} concurrent {v['concurrent']['within300']:.3f}"
              f" | small {v['small']['within300']:.3f} large {v['large']['within300']:.3f}", flush=True)
    json.dump(results, open("localiser_ltown.json", "w"), indent=1, default=float)
