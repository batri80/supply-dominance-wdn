"""Experiment X1 (ext_protocol.md): localiser across seeds, with baselines, on any network.

Usage: python localiser2.py ltown|ky4 [n_seeds=5] [epochs=30]
Needs: train_<net>_s0.npz, train_<net>_s1.npz (gen_train.py) and ext_sensors.json (ext_sensors.py).
Writes ext_localiser_<net>.json, updated after every seed so an interrupted run resumes.
"""
import contextlib
import io
import json
import os
import sys
import numpy as np
import networkx as nx
import config
from core import simulate, influence_matrix, grounded_laplacian
from networks import load as load_net

with contextlib.redirect_stderr(io.StringIO()):   # silence the NumPy 1.x/2.x notice from torch 2.2 on Intel Macs
    import torch
    import torch.nn as nn

NET = sys.argv[1]
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
N_TRAIN = int(os.environ.get("PSEP_EXT_NTRAIN", 0)) or None      # for quick tests only
N_TEST = int(os.environ.get("PSEP_EXT_NTEST", 0)) or None
SIGMA, RADIUS, LAMBDA_PHYS = 0.01, 300.0, 0.1
torch.set_num_threads(max(1, min(8, config.WORKERS)))
OUTFILE = f"ext_localiser_{NET}.json"


def to_t(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(a.shape)


def to_np(t):
    return np.array(t.detach().tolist(), dtype=np.float32)


# ---------------------------------------------------------------- network quantities
wn = load_net(NET)
J = wn.junction_name_list
idx = {n: k for k, n in enumerate(J)}
nJ = len(J)
SENS = json.load(open("ext_sensors.json"))[NET]
s_idx = np.array([idx[s] for s in SENS])
h0, q0 = simulate(wn)
Gi, _ = influence_matrix(wn, h0, q0, J)
Lg, _ = grounded_laplacian(wn, h0, q0, J)
sigc = Gi[s_idx, :] - Gi[s_idx, :].mean(0, keepdims=True)
sign = sigc / (np.linalg.norm(sigc, axis=0, keepdims=True) + 1e-12)

Gw = nx.Graph()
edges, cond = [], []
for name, l in wn.links():
    a, b = l.start_node_name, l.end_node_name
    Gw.add_edge(a, b, w=getattr(l, "length", 0.0) or 0.0)
    if a in idx and b in idx:
        dq = abs(q0[name]); dh = abs(h0[a] - h0[b])
        g = 0.1 if dh < 1e-9 else min(dq / (1.852 * dh), 0.1)
        edges += [(idx[a], idx[b]), (idx[b], idx[a])]
        cond += [float(g), float(g)]
E = torch.tensor(edges, dtype=torch.long).T
w = torch.tensor(cond, dtype=torch.float32)
w = w / (torch.zeros(nJ).index_add_(0, E[0], w)[E[0]] + 1e-9)
A_msg = torch.sparse_coo_tensor(E, w, (nJ, nJ)).coalesce()           # conductance-weighted mean
Ls = Lg / np.mean(np.diag(Lg))                                       # scaled grounded Laplacian
nz = np.nonzero(Ls)
L_phys = torch.sparse_coo_tensor(torch.tensor(np.vstack(nz).tolist(), dtype=torch.long),
                                 torch.tensor(Ls[nz].tolist(), dtype=torch.float32), (nJ, nJ)).coalesce()

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

Lc = nx.laplacian_matrix(Gw.subgraph(J), nodelist=J).astype(float).toarray()
free = np.setdiff1d(np.arange(nJ), s_idx)
interp = -np.linalg.solve(Lc[np.ix_(free, free)] + 1e-9 * np.eye(len(free)), Lc[np.ix_(free, s_idx)])
_dist = {}


def dist(a, b):
    if a not in _dist:
        _dist[a] = nx.single_source_dijkstra_path_length(Gw, a, weight="w")
    return _dist[a].get(b, np.inf)


# ---------------------------------------------------------------- data
def load_set(seed, n_max):
    d = np.load(f"train_{NET}_s{seed}.npz")
    R = d["R"][:n_max] if n_max else d["R"]
    rng = np.random.default_rng(100 + seed)
    obs = R[:, s_idx] + rng.normal(0, SIGMA, (len(R), len(SENS)))
    return obs.astype(np.float32), R.astype(np.float32), d["nodes"][:len(R)], d["areas"][:len(R)]


def field(obs):
    F = np.zeros((len(obs), nJ), dtype=np.float32)
    F[:, s_idx] = obs
    F[:, free] = obs @ interp.T
    return F


def scale(obs):
    return np.abs(obs).max(1, keepdims=True) + 1e-9


def linear_scores(obs):
    o = obs - obs.mean(1, keepdims=True)
    return (o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)) @ sign


def features(obs, prior, dom):
    X = [field(obs) / scale(obs), np.broadcast_to(np.isin(np.arange(nJ), s_idx), (len(obs), nJ))]
    if prior:
        X.append(linear_scores(obs))
    if dom:
        X += [np.broadcast_to(in_sh, (len(obs), nJ)), np.broadcast_to(entry, (len(obs), nJ))]
    return to_t(np.stack(X, -1))


def targets(nodes):
    T = np.zeros((len(nodes), nJ), dtype=np.float32)
    for i, ns in enumerate(nodes):
        for n in ns:
            if n:
                T[i, idx[n]] = 1.0
    return to_t(T / T.sum(1, keepdims=True))


# ---------------------------------------------------------------- model
class GNN(nn.Module):
    def __init__(self, fin, h=64, layers=4):
        super().__init__()
        self.inp = nn.Linear(fin, h)
        self.self_w = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])
        self.nb_w = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(layers)])
        self.out = nn.Linear(h + fin, 1)

    def forward(self, X):
        B, n, _ = X.shape
        h = torch.relu(self.inp(X))
        for s, nb in zip(self.self_w, self.nb_w):
            m = torch.sparse.mm(A_msg, h.permute(1, 0, 2).reshape(n, -1)).reshape(n, B, -1).permute(1, 0, 2)
            h = torch.relu(s(h) + nb(m)) + h
        return self.out(torch.cat([h, X], -1)).squeeze(-1)


def implied_outflow(f):                     # s = L_g f for a batch of fields
    return torch.sparse.mm(L_phys, f.T).T


METHODS = {   # name: (use prior, use dominance flags, objective)
    "gnn_plain": (False, False, "class"),
    "reconstructor": (False, False, "field"),
    "physics_loss": (False, False, "physics"),
    "hybrid_nodom": (True, False, "class"),
    "hybrid": (True, True, "class"),
}


def train(name, seed, Xtr, Ttr, Ftr):
    prior, dom, obj = METHODS[name]
    torch.manual_seed(seed)
    m = GNN(fin=Xtr.shape[-1])
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    n = len(Xtr)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, 32):
            ib = perm[b:b + 32]
            out = m(Xtr[ib])
            if obj == "class":
                loss = -(Ttr[ib] * torch.log_softmax(out, -1)).sum(-1).mean()
            else:
                loss = ((out - Ftr[ib]) ** 2).mean()
                if obj == "physics":
                    loss = loss + LAMBDA_PHYS * implied_outflow(out).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(ib)
        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            print(f"      {name} seed {seed} epoch {ep+1}: loss {tot/n:.4f}", flush=True)
    return m


def scores_of(m, X, outflow=False):
    with torch.no_grad():
        out = torch.cat([m(X[b:b + 64]) for b in range(0, len(X), 64)])
        if outflow:
            out = implied_outflow(out)
    return to_np(out)


def evaluate(score, nodes):
    rows = []
    for i in range(len(score)):
        true = [n for n in nodes[i] if n]
        order = np.argsort(-score[i])
        est = J[order[0]]
        d = min(dist(est, t) for t in true)
        rows.append((float(est in true), float(any(J[k] in true for k in order[:5])), d, float(d <= RADIUS), len(true) > 1))
    rows = np.array(rows, dtype=object)

    def summ(mask):
        r = rows[mask]
        return dict(n=int(len(r)), within300=float(np.mean(r[:, 3].astype(float))), top1=float(np.mean(r[:, 0].astype(float))),
                    top5=float(np.mean(r[:, 1].astype(float))), median_dist_m=float(np.median(r[:, 2].astype(float))))
    conc = rows[:, 4].astype(bool)
    return dict(all=summ(np.ones(len(rows), bool)), single=summ(~conc), concurrent=summ(conc))


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [float(v.mean()), float("nan"), float("nan")]
    from scipy.stats import t
    h = t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


if __name__ == "__main__":
    obs_tr, R_tr, nodes_tr, _ = load_set(0, N_TRAIN)
    obs_te, _, nodes_te, _ = load_set(1, N_TEST)
    print(f"{NET}: {len(SENS)} sensors, {len(obs_tr)} training and {len(obs_te)} test scenarios, "
          f"{N_SEEDS} seeds, {EPOCHS} epochs", flush=True)
    res = json.load(open(OUTFILE)) if os.path.exists(OUTFILE) else {}
    res.setdefault("deterministic", {})
    res.setdefault("seeds", {})
    if not res["deterministic"]:
        nearest = np.zeros((len(obs_te), nJ)); nearest[np.arange(len(obs_te)), s_idx[np.argmax(obs_te, 1)]] = 1
        res["deterministic"] = {"nearest_sensor": evaluate(nearest, nodes_te),
                                "graph_interp": evaluate(field(obs_te), nodes_te),
                                "linear": evaluate(linear_scores(obs_te), nodes_te)}
        json.dump(res, open(OUTFILE, "w"), indent=1)
    for k, v in res["deterministic"].items():
        print(f"  {k:15s} within 300 m {v['all']['within300']:.3f}", flush=True)
    Ttr, Ftr = targets(nodes_tr), to_t(R_tr / scale(obs_tr))
    feats = {}
    for seed in range(N_SEEDS):
        if str(seed) in res["seeds"] and all(k in res["seeds"][str(seed)] for k in METHODS):
            print(f"  seed {seed}: already done", flush=True)
            continue
        res["seeds"].setdefault(str(seed), {})
        for name, (prior, dom, _) in METHODS.items():
            if name in res["seeds"][str(seed)]:
                continue
            key = (prior, dom)
            if key not in feats:
                feats[key] = (features(obs_tr, prior, dom), features(obs_te, prior, dom))
            Xtr, Xte = feats[key]
            m = train(name, seed, Xtr, Ttr, Ftr)
            r = evaluate(scores_of(m, Xte), nodes_te)
            res["seeds"][str(seed)][name] = r
            print(f"  seed {seed} {name:14s} within 300 m {r['all']['within300']:.3f}", flush=True)
            if name == "physics_loss":                      # secondary scoring of the same model
                r2 = evaluate(scores_of(m, Xte, outflow=True), nodes_te)
                res["seeds"][str(seed)]["physics_loss_outflow"] = r2
                print(f"  seed {seed} {'physics_loss_outflow':14s} within 300 m {r2['all']['within300']:.3f}", flush=True)
            json.dump(res, open(OUTFILE, "w"), indent=1)
    # summary
    summ = {}
    REPORT = list(METHODS)[:3] + ["physics_loss_outflow"] + list(METHODS)[3:]
    for name in REPORT:
        vals = [res["seeds"][s][name] for s in res["seeds"] if name in res["seeds"][s]]
        summ[name] = {grp: {met: ci([v[grp][met] for v in vals]) for met in ("within300", "top1", "top5", "median_dist_m")}
                      for grp in ("all", "single", "concurrent")}
    lin = res["deterministic"]["linear"]["all"]["within300"]
    seeds = [s for s in res["seeds"] if "hybrid" in res["seeds"][s]]
    summ["diff_hybrid_minus_linear"] = ci([res["seeds"][s]["hybrid"]["all"]["within300"] - lin for s in seeds])
    summ["diff_hybrid_minus_nodom"] = ci([res["seeds"][s]["hybrid"]["all"]["within300"]
                                          - res["seeds"][s]["hybrid_nodom"]["all"]["within300"]
                                          for s in seeds if "hybrid_nodom" in res["seeds"][s]])
    res["summary"] = summ
    json.dump(res, open(OUTFILE, "w"), indent=1)
    print(f"\n{NET}: share within 300 m, mean [95% interval] over {len(seeds)} seeds")
    for k, v in res["deterministic"].items():
        print(f"  {k:15s} {v['all']['within300']:.3f}")
    for name in REPORT:
        m_, lo, hi = summ[name]["all"]["within300"]
        print(f"  {name:15s} {m_:.3f} [{lo:.3f}, {hi:.3f}]")
    for k in ("diff_hybrid_minus_linear", "diff_hybrid_minus_nodom"):
        m_, lo, hi = summ[k]
        print(f"  {k:26s} {m_:+.3f} [{lo:+.3f}, {hi:+.3f}]")
