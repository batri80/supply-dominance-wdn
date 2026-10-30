"""Proposition 3 test: does dominance-aware sensor placement localise leaks better?

Placement methods (budget m sensors):
  random        - uniform random junctions (30 draws)
  pagerank      - top-m PageRank centrality (as in van Gemert et al., 2026)
  greedy_lib    - greedy isolability on the SIMULATED leak library (needs one
                  simulation per candidate; the expensive classical route)
  dom_struct    - structure only: one sensor deep inside each of the largest
                  maximal shadows, the rest spread over the non-shadow core by
                  farthest-point sampling in hop distance. No hydraulics at all.
  dom_linear    - greedy isolability on the LINEARISED influence matrix G (one
                  baseline simulation), with at most one sensor per maximal shadow.

Evaluation uses the simulated library: observation = simulated head drop at the
sensors + N(0, sigma^2); estimate = angle method (centred cosine) against the
simulated library. Metrics: mean hop error, top-1, top-5, and the number of
indistinguishable groups (pairs with centred cosine above 0.999 merged).

Usage: python placement.py synthetic|ky4
"""
import json
import sys
import numpy as np
import networkx as nx
from core import simulate, influence_matrix
from networks import synthetic, ky4, load

NET = sys.argv[1]
SIGMA = 0.01
RNG = np.random.default_rng(7)
FRACS = (0.05, 0.10, 0.20)
N_NOISE = 10           # noise draws per leak

wn = load(NET)
lib = np.load(f"lib_{NET}_0.0001.npz")
J, cand, R = list(lib["J"]), list(lib["cand"]), lib["R"].astype(float)
jidx = {n: k for k, n in enumerate(J)}
cidx = np.array([jidx[c] for c in cand])
nJ, nC = len(J), len(cand)

Gu = nx.Graph([(l.start_node_name, l.end_node_name) for _, l in wn.links()])
fixed = set(wn.reservoir_name_list) | set(wn.tank_name_list)
h0, q0 = simulate(wn)
Ginv, _ = influence_matrix(wn, h0, q0, J)
# linear signatures in metres: dh = qL * G e_j, with qL from the orifice law at baseline pressure
elev = np.array([wn.get_node(n).elevation for n in J])
p0 = h0[J].values - elev
qL = 0.75 * 1e-4 * np.sqrt(2 * 9.81 * np.maximum(p0[cidx], 0))
Lsig = (Ginv[:, cidx] * qL).T                            # nC x nJ

# ---------- shadows
def shadow(j):
    H = Gu.copy(); H.remove_node(j)
    reach = set()
    for f in fixed:
        if f in H:
            reach |= nx.node_connected_component(H, f)
    return frozenset(n for n in J if n != j and n not in reach)

sh = {j: shadow(j) for j in J}
nonempty = {j: s for j, s in sh.items() if s}
# maximal shadows: those not contained in another node's shadow
maximal = [j for j, s in nonempty.items() if not any(j in s2 for k, s2 in nonempty.items() if k != j)]
maximal.sort(key=lambda j: -len(sh[j]))
in_shadow = set().union(*[sh[j] for j in maximal]) if maximal else set()
owner = {n: j for j in maximal for n in sh[j]}
print(f"{NET}: {nJ} junctions, {nC} candidates, {len(maximal)} maximal shadows covering {len(in_shadow)} junctions", flush=True)

hop = {}
def hops_from(n):
    if n not in hop:
        hop[n] = nx.single_source_shortest_path_length(Gu, n)
    return hop[n]


# ---------- greedy isolability on a signature matrix (rows = candidates)
def greedy(sig, m, allowed, max_per_shadow=None, n_pairs=4000):
    rp = np.random.default_rng(11)                # same pairs for every method
    pa = rp.integers(0, nC, n_pairs); pb = rp.integers(0, nC, n_pairs)
    keep = pa != pb; pa, pb = pa[keep], pb[keep]
    A, B = sig[pa], sig[pb]                       # pairs x nJ
    sa = np.zeros(len(pa)); sb = np.zeros(len(pa)); sab = np.zeros(len(pa)); saa = np.zeros(len(pa)); sbb = np.zeros(len(pa))
    chosen, count = [], {}
    allowed = list(allowed)
    for step in range(m):
        k = step + 1
        best, bestv = None, -1
        for s in allowed:
            if s in chosen:
                continue
            if max_per_shadow is not None and J[s] in owner and count.get(owner[J[s]], 0) >= max_per_shadow:
                continue
            a, b = A[:, s], B[:, s]
            Sa, Sb, Sab, Saa, Sbb = sa + a, sb + b, sab + a * b, saa + a * a, sbb + b * b
            if k < 2:
                v = np.var(sig[:, s])             # first sensor: most varied response
            else:
                cov = Sab - Sa * Sb / k
                va = np.maximum(Saa - Sa ** 2 / k, 1e-18); vb = np.maximum(Sbb - Sb ** 2 / k, 1e-18)
                cos = np.clip(cov / np.sqrt(va * vb), -1, 1)
                # noise-aware separation: angular distance times the weaker signal's
                # norm in units of sensor noise, capped (beyond ~3 sigma a pair is separated)
                snr = np.sqrt(np.minimum(va, vb)) / SIGMA
                v = np.mean(np.minimum(np.sqrt(1 - cos ** 2) * snr, 3.0))
            if v > bestv:
                best, bestv = s, v
        chosen.append(best)
        if J[best] in owner:
            count[owner[J[best]]] = count.get(owner[J[best]], 0) + 1
        a, b = A[:, best], B[:, best]
        sa += a; sb += b; sab += a * b; saa += a * a; sbb += b * b
    return chosen


def dom_struct(m):
    chosen = []
    # one sensor at the deepest node of each of the largest maximal shadows (up to half the budget)
    for j in maximal[: max(1, m // 2)]:
        if len(sh[j]) < 3:
            break
        d = hops_from(j)
        chosen.append(jidx[max(sh[j], key=lambda n: d.get(n, 0))])
    core = [k for k, n in enumerate(J) if n not in in_shadow]
    if not chosen:
        chosen.append(core[0])
    while len(chosen) < m:                        # farthest-point sampling over the core
        dist = np.full(len(core), np.inf)
        for c in chosen:
            d = hops_from(J[c])
            dist = np.minimum(dist, [d.get(J[k], 1e9) for k in core])
        nxt = core[int(np.argmax(dist))]
        if nxt in chosen:
            break
        chosen.append(nxt)
    return chosen[:m]


def fps_all(m):
    """Ablation of dom_struct: farthest-point sampling over all junctions, no shadow rule."""
    allk = list(range(nJ))
    chosen = [int(np.argmax([len(hops_from(J[0]))]))]
    chosen = [max(allk, key=lambda k: hops_from(J[0]).get(J[k], 0))]
    while len(chosen) < m:
        dist = np.full(nJ, np.inf)
        for c in chosen:
            d = hops_from(J[c])
            dist = np.minimum(dist, [d.get(J[k], 1e9) for k in allk])
        chosen.append(int(np.argmax(dist)))
    return chosen


def pagerank(m):
    pr = nx.pagerank(Gu)
    order = sorted((n for n in J), key=lambda n: -pr[n])
    return [jidx[n] for n in order[:m]]


# ---------- evaluation
def evaluate(sens):
    sens = np.array(sens)
    D = R[:, sens]
    D = D - D.mean(1, keepdims=True)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    err, t1, t5 = [], [], []
    rn = np.random.default_rng(123)               # identical noise draws for every method
    for _ in range(N_NOISE):
        obs = R[:, sens] + rn.normal(0, SIGMA, (nC, len(sens)))
        obs = obs - obs.mean(1, keepdims=True)
        S = obs @ Dn.T                            # nC x nC scores
        order = np.argsort(-S, axis=1)
        est = order[:, 0]
        t1.append(np.mean(est == np.arange(nC)))
        t5.append(np.mean([(np.arange(nC)[i] in order[i, :5]) for i in range(nC)]))
        err.append(np.mean([hops_from(cand[i]).get(cand[est[i]], 0) for i in range(nC)]))
    C = Dn @ Dn.T
    G = nx.Graph(); G.add_nodes_from(range(nC))
    ii, jj = np.where(np.triu(C, 1) > 0.999)
    G.add_edges_from(zip(ii, jj))
    groups = nx.number_connected_components(G)
    return dict(hop_error=float(np.mean(err)), top1=float(np.mean(t1)), top5=float(np.mean(t5)),
                distinguishable_groups=int(groups), sensors_in_shadows=int(sum(J[s] in in_shadow for s in sens)))


UTILITY = {"ltown": ["n1", "n4", "n31", "n54", "n105", "n114", "n163", "n188", "n215", "n229", "n288", "n296", "n332",
                     "n342", "n410", "n415", "n429", "n458", "n469", "n495", "n506", "n516", "n519", "n549", "n613",
                     "n636", "n644", "n679", "n722", "n726", "n740", "n752", "n769"]}
if NET in UTILITY:
    FRACS = (len(UTILITY[NET]) / nJ,) + FRACS

if __name__ == "__main__":
    results = {"network": NET, "n_junctions": nJ, "n_candidates": nC, "n_maximal_shadows": len(maximal),
               "junctions_in_shadows": len(in_shadow), "sigma_m": SIGMA, "budgets": {}}
    for frac in FRACS:
        m = max(3, int(round(frac * nJ)))
        key = f"{int(round(frac*100))}%" if m != len(UTILITY.get(NET, [])) else f"{m} sensors (utility budget)"
        row = {}
        rr = [evaluate(RNG.choice(nJ, m, replace=False)) for _ in range(30 if nJ < 200 else 10)]
        row["random"] = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
        if NET in UTILITY and m == len(UTILITY[NET]):
            row["utility_layout"] = evaluate([jidx[n] for n in UTILITY[NET]])
        row["pagerank"] = evaluate(pagerank(m))
        row["dom_struct"] = evaluate(dom_struct(m))
        row["fps_all"] = evaluate(fps_all(m))                           # ablation: no shadow rule
        row["dom_linear"] = evaluate(greedy(Lsig, m, range(nJ), max_per_shadow=1))
        row["linear_nocap"] = evaluate(greedy(Lsig, m, range(nJ)))        # ablation: no shadow rule
        row["greedy_lib"] = evaluate(greedy(R, m, range(nJ)))
        results["budgets"][key] = dict(m=m, **row)
        print(f"budget {frac:.0%} (m={m})", flush=True)
        for k, v in row.items():
            print(f"   {k:11s} hop_err {v['hop_error']:.2f}  top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  groups {v['distinguishable_groups']}  in_shadow {v['sensors_in_shadows']}", flush=True)
    json.dump(results, open(f"placement_{NET}.json", "w"), indent=2)
