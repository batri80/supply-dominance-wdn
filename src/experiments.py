"""Four checks of the fault-propagation claims.

E1 decay:      does a leak's influence fall off with distance, and is hydraulic
               distance (linearised influence) a better predictor than hop count?
E2 masking:    are leaks on two branches entering a merge harder to tell apart
               (more similar sensor signatures) than leaks on two branches leaving a split?
E3 shadow:     do nodes that are fed only through the leak node ("shadow" nodes,
               e.g. a dead-end branch below it) escape the decay?
E4 nonlinear:  does superposition fail for concurrent leaks, and does the
               influence pattern change with the operating point (demand level)?

Usage: python experiments.py synthetic|ky4 [n_leaks]
"""
import json
import sys
import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from core import simulate, junction_types, influence_matrix
from networks import synthetic, ky4

AREA = 1e-4          # orifice area (m^2); about 2-3 L/s at typical pressures
RNG = np.random.default_rng(42)


def undirected(wn):
    G = nx.Graph()
    for _, l in wn.links():
        G.add_edge(l.start_node_name, l.end_node_name)
    return G


def shadow_sets(wn, Gu, nodes):
    """shadow[j] = junctions disconnected from every fixed-head node if j is removed."""
    fixed = set(wn.reservoir_name_list) | set(wn.tank_name_list)
    out = {}
    for j in nodes:
        H = Gu.copy()
        H.remove_node(j)
        reach = set()
        for f in fixed:
            if f in H:
                reach |= nx.node_connected_component(H, f)
        out[j] = set(n for n in wn.junction_name_list if n != j and n not in reach)
    return out


def leak_response(wn, h0, j, area=AREA, mult=1.0, h_base=None):
    h, _ = simulate(wn, {j: area}, demand_mult=mult)
    base = h0 if h_base is None else h_base
    return (base - h)


def main(netname, n_leaks=None):
    wn = synthetic() if netname == "synthetic" else ky4()
    J = wn.junction_name_list
    h0, q0 = simulate(wn)
    types, Gd = junction_types(wn, q0)
    Gu = undirected(wn)
    Ginv, idx = influence_matrix(wn, h0, q0, J)

    elev = np.array([wn.get_node(n).elevation for n in J])
    press = h0[J].values - elev
    cand = [n for n, p in zip(J, press) if p > 15]          # avoid low-pressure nodes
    cset = set(cand)
    pairs = {"merge_parents": set(), "split_children": set()}
    for n, t in types.items():
        if t == "merge":
            ps = sorted(u for u, _ in Gd.in_edges(n) if u in cset)
            pairs["merge_parents"] |= {(a, b) for a in ps for b in ps if a < b}
        if t == "split":
            cs = sorted(v for _, v in Gd.out_edges(n) if v in cset)
            pairs["split_children"] |= {(a, b) for a in cs for b in cs if a < b}
    hop2 = set()
    for n, t in types.items():
        if t == "pass":
            nb = sorted(m for m in Gu.neighbors(n) if m in cset)
            if len(nb) == 2:
                hop2.add(tuple(nb))
    pairs_all = {k: sorted(v) for k, v in pairs.items()}
    pairs_all["hop2_via_pass"] = sorted(hop2)
    if n_leaks is None:
        leaks = cand
    else:  # sample pairs first so that both members of each pair are simulated
        chosen = set()
        for k, v in pairs_all.items():
            pick = RNG.choice(len(v), size=min(25, len(v)), replace=False) if v else []
            pairs_all[k] = [v[i] for i in pick]
            for a, b in pairs_all[k]:
                chosen |= {a, b}
        rest = [c for c in cand if c not in chosen]
        extra = max(0, n_leaks - len(chosen))
        leaks = sorted(chosen) + list(RNG.choice(rest, size=min(extra, len(rest)), replace=False))
    print(f"{netname}: {len(J)} junctions, testing {len(leaks)} leak locations", flush=True)

    # ---- single-leak responses (shared by E1-E3)
    resp = {}
    for k, j in enumerate(leaks):
        resp[j] = leak_response(wn, h0, j)[J].values
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(leaks)} leaks simulated", flush=True)

    # ---- E1 decay
    hop = dict(nx.all_pairs_shortest_path_length(Gu)) if len(J) < 200 else None
    rho_hop, rho_lin, r2_lin = [], [], []
    for j in leaks:
        dh = resp[j]
        jj = idx[j]
        others = np.array([k for k in range(len(J)) if k != jj])
        rel = dh[others] / dh[jj]
        hops = np.array([(hop[j] if hop else nx.single_source_shortest_path_length(Gu, j)).get(J[k], np.nan) for k in others])
        lin = Ginv[others, jj] / Ginv[jj, jj]
        ok = np.isfinite(hops)
        rho_hop.append(spearmanr(hops[ok], rel[ok]).correlation)
        rho_lin.append(spearmanr(lin, rel).correlation)
        # linear prediction of the full profile, scaled by the observed self-response
        pred = Ginv[:, jj] / Ginv[jj, jj] * dh[jj]
        r2_lin.append(1 - np.sum((dh - pred) ** 2) / np.sum((dh - dh.mean()) ** 2))
    E1 = dict(spearman_hop_median=float(np.nanmedian(rho_hop)),
              spearman_linear_influence_median=float(np.nanmedian(rho_lin)),
              frac_leaks_linear_beats_hop=float(np.mean(np.abs(rho_lin) > np.abs(rho_hop))),
              linear_profile_R2_median=float(np.nanmedian(r2_lin)),
              linear_profile_R2_q10=float(np.nanpercentile(r2_lin, 10)))
    print("E1", E1, flush=True)

    # ---- E2 masking: signature similarity at a sparse sensor set
    pairs = {k: [p for p in pairs_all[k] if p[0] in resp and p[1] in resp] for k in ("merge_parents", "split_children", "hop2_via_pass")}
    # Signatures are centred across sensors (the network-wide common drop carries
    # no location information). Localisation: angle method against the noiseless
    # library under sensor noise SIGMA; we record how often a leak is assigned to its
    # sibling (the other parent of the same merge, or other child of the same split).
    SIGMA = 0.01  # m, pressure-sensor noise
    E2 = {"n_pairs": {k: len(v) for k, v in pairs.items()}, "sensor_noise_m": SIGMA}
    sims = {k: [] for k in pairs}
    confuse = {k: [] for k in pairs}
    n_sens = max(5, len(J) // 10)
    L = list(resp)
    for draw in range(30):
        sens = RNG.choice(len(J), size=n_sens, replace=False)
        lib = np.array([resp[c][sens] for c in L])
        lib = lib - lib.mean(1, keepdims=True)
        libn = lib / (np.linalg.norm(lib, axis=1, keepdims=True) + 1e-12)
        for k, ps in pairs.items():
            for a, b in ps:
                ia, ib = L.index(a), L.index(b)
                sims[k].append(float(libn[ia] @ libn[ib]))
                for true, sib in ((ia, ib), (ib, ia)):
                    obs = resp[L[true]][sens] + RNG.normal(0, SIGMA, n_sens)
                    obs = obs - obs.mean()
                    score = libn @ (obs / (np.linalg.norm(obs) + 1e-12))
                    confuse[k].append(float(np.argmax(score) == sib))
    for k in sims:
        v = np.array(sims[k])
        E2[k] = dict(centred_cos_median=float(np.median(v)) if len(v) else None,
                     sibling_confusion_rate=float(np.mean(confuse[k])) if confuse[k] else None)
    # E2b: localisation error (hops from true leak to the angle-method estimate),
    # grouped by the hydraulic role of the true leak node.
    role = {}
    for j in L:
        t = types.get(j, "pass")
        role[j] = t if t in ("merge", "split") else "other"
    err = {"merge": [], "split": [], "other": []}
    top1 = {"merge": [], "split": [], "other": []}
    dist_cache = {j: nx.single_source_shortest_path_length(Gu, j) for j in L}
    for draw in range(20):
        sens = RNG.choice(len(J), size=n_sens, replace=False)
        lib = np.array([resp[c][sens] for c in L])
        lib = lib - lib.mean(1, keepdims=True)
        libn = lib / (np.linalg.norm(lib, axis=1, keepdims=True) + 1e-12)
        for t, j in enumerate(L):
            obs = resp[j][sens] + RNG.normal(0, SIGMA, n_sens)
            obs = obs - obs.mean()
            est = L[int(np.argmax(libn @ obs))]
            err[role[j]].append(dist_cache[j][est])
            top1[role[j]].append(float(est == j))
    E2["localisation_by_role"] = {r: dict(n=len(err[r]) // 20,
                                          hop_error_mean=float(np.mean(err[r])) if err[r] else None,
                                          top1=float(np.mean(top1[r])) if top1[r] else None)
                                  for r in err}
    print("E2", E2, flush=True)

    # ---- E3 shadow nodes vs non-shadow nodes at the same hop distance
    sh = shadow_sets(wn, Gu, leaks)
    sh_vals, non_vals = [], []
    for j in leaks:
        if not sh[j]:
            continue
        d = nx.single_source_shortest_path_length(Gu, j)
        dh = resp[j]
        selfv = dh[idx[j]]
        for n in sh[j]:
            dist = d[n]
            peers = [m for m in J if m not in sh[j] and m != j and d.get(m) == dist]
            if not peers:
                continue
            sh_vals.append(dh[idx[n]] / selfv)
            non_vals.append(np.mean([dh[idx[m]] / selfv for m in peers]))
    E3 = dict(n_shadow_nodes=len(sh_vals),
              shadow_rel_influence_median=float(np.median(sh_vals)) if sh_vals else None,
              nonshadow_same_hop_median=float(np.median(non_vals)) if non_vals else None)
    print("E3", E3, flush=True)

    # ---- E4a superposition failure vs leak size, by pair type
    E4a = {}
    pool = {"split_children": pairs["split_children"], "merge_parents": pairs["merge_parents"],
            "hop2_via_pass": [p for p in pairs_all["hop2_via_pass"] if p[0] in resp and p[1] in resp],
            "random": [tuple(RNG.choice(leaks, 2, replace=False)) for _ in range(15)]}
    npair = 15 if netname == "synthetic" else 8
    for area in (1e-4, 4e-4, 1e-3):
        row = {}
        for k, ps in pool.items():
            ps = ps[:npair]
            errs = []
            for a, b in ps:
                da = leak_response(wn, h0, a, area)[J].values
                db = leak_response(wn, h0, b, area)[J].values
                h2, _ = simulate(wn, {a: area, b: area})
                dab = (h0 - h2)[J].values
                errs.append(np.linalg.norm(dab - (da + db)) / np.linalg.norm(da + db))
            row[k] = float(np.median(errs)) if errs else None
        E4a[f"area_{area:g}"] = row
        print("E4a", area, row, flush=True)

    # ---- E4b operating point: same leak at low / nominal / peak demand
    sub = leaks[:20]
    E4b = {}
    prof = {}
    for mult in (0.5, 1.0, 1.5):
        hb, _ = simulate(wn, demand_mult=mult)
        prof[mult] = {j: leak_response(wn, h0, j, mult=mult, h_base=hb)[J].values for j in sub}
    for mult in (0.5, 1.5):
        cors, selfratio = [], []
        for j in sub:
            a, b = prof[1.0][j], prof[mult][j]
            cors.append(np.corrcoef(a, b)[0, 1])
            na, nb = a / a[idx[j]], b / b[idx[j]]
            selfratio.append(np.mean(nb) / np.mean(na))   # >1 = influence spreads further
        E4b[f"mult_{mult}"] = dict(profile_corr_median=float(np.median(cors)),
                                   spread_ratio_vs_nominal_median=float(np.median(selfratio)))
    print("E4b", E4b, flush=True)

    out = dict(network=netname, n_junctions=len(J), n_leaks=len(leaks),
               junction_types={t: list(types.values()).count(t) for t in set(types.values())},
               E1=E1, E2=E2, E3=E3, E4a=E4a, E4b=E4b)
    json.dump(out, open(f"results_{netname}.json", "w"), indent=2)
    np.savez(f"resp_{netname}.npz", leaks=np.array(leaks), J=np.array(J),
             R=np.array([resp[j] for j in leaks]))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
