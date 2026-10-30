"""Revision checks R5 (rev_protocol.md). Deterministic; no decision rules.

Usage: python rev_checks.py [cap] [walk] [prop3] [prop4] [battledim]   (default: all)
  cap        conductance-cap sensitivity of the linearised influence (0.01, 0.1, 1, 10)
  walk       worked example of Proposition 2 on the synthetic network
  prop3      isolability-limit contrast on KY4 (sensors inside vs partly outside a shadow)
  prop4      signed interaction of concurrent 10 cm^2 leaks (synthetic and KY4)
  battledim  per-leak detection table with delays and false-alarm times
Writes rev_r5_<check>.json for each check.
"""
import json
import os
import sys
import numpy as np
import networkx as nx
import wntr
from multiprocessing import Pool
from scipy.stats import spearmanr
import config
from core import simulate, influence_matrix, grounded_laplacian, junction_types
from networks import load

WANT = sys.argv[1:] or ["cap", "walk", "prop3", "prop4", "battledim"]
CAPS = (0.01, 0.1, 1.0, 10.0)


def r2(y, x):
    """R^2 of y against the no-intercept fit y = a x."""
    a = (x @ y) / max(x @ x, 1e-300)
    return 1 - np.sum((y - a * x) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)


def shadows(wn, J):
    Gu = nx.Graph([(l.start_node_name, l.end_node_name) for _, l in wn.links()])
    fixed = set(wn.reservoir_name_list) | set(wn.tank_name_list)
    sh = {}
    for j in J:
        H = Gu.copy(); H.remove_node(j)
        reach = set()
        for f in fixed:
            if f in H:
                reach |= nx.node_connected_component(H, f)
        sh[j] = {n for n in J if n != j and n not in reach}
    return sh, Gu


# --------------------------------------------------------------------------- cap
def check_cap():
    out = {}
    for net in ("synthetic", "ky4", "ltown"):
        f = f"lib_{net}_0.0001.npz"
        if not os.path.exists(f):
            print(f"  cap: {f} missing, {net} skipped"); continue
        wn = load(net)
        lib = np.load(f)
        J, cand, R = list(lib["J"]), list(lib["cand"]), lib["R"].astype(float)
        jidx = {n: k for k, n in enumerate(J)}
        h0, q0 = simulate(wn)
        rows = np.random.default_rng(0).choice(len(cand), min(200, len(cand)), replace=False)
        out[net] = {}
        for cap in CAPS:
            G, _ = influence_matrix(wn, h0, q0, J, gmax=cap)
            sp, rr = [], []
            for k in rows:
                j = jidx[cand[k]]
                u, y = G[:, j] / G[j, j], R[k] / max(R[k, j], 1e-12)
                sp.append(spearmanr(u, y).correlation); rr.append(r2(y, u))
            out[net][str(cap)] = dict(spearman_median=float(np.nanmedian(sp)), profile_r2_median=float(np.median(rr)),
                                      share_r2_below_0_9=float(np.mean(np.array(rr) < 0.9)))
        # localisation on the X1 test set, if present
        if os.path.exists(f"train_{net}_s1.npz") and os.path.exists("ext_sensors.json"):
            S = json.load(open("ext_sensors.json"))[net]
            s_idx = np.array([jidx[s] for s in S])
            d = np.load(f"train_{net}_s1.npz")
            obs = d["R"][:, s_idx] + np.random.default_rng(101).normal(0, 0.01, (len(d["R"]), len(S)))
            o = obs - obs.mean(1, keepdims=True)
            o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
            Gw = nx.Graph()
            for _, l in wn.links():
                Gw.add_edge(l.start_node_name, l.end_node_name, w=getattr(l, "length", 0.0) or 0.0)
            dd = {}
            for cap in CAPS:
                G, _ = influence_matrix(wn, h0, q0, J, gmax=cap)
                sig = G[s_idx, :] - G[s_idx, :].mean(0, keepdims=True)
                sig /= np.linalg.norm(sig, axis=0, keepdims=True) + 1e-12
                est = np.argmax(o @ sig, 1)
                hit = []
                for i, e in enumerate(est):
                    if J[e] not in dd:
                        dd[J[e]] = nx.single_source_dijkstra_path_length(Gw, J[e], weight="w")
                    hit.append(min(dd[J[e]].get(t, np.inf) for t in d["nodes"][i] if t) <= 300)
                out[net][str(cap)]["localisation_within_300m"] = float(np.mean(hit))
        print(f"  cap {net}: " + "; ".join(
            f"{c}: Spearman {v['spearman_median']:.3f}, R2 {v['profile_r2_median']:.3f}"
            + (f", within 300 m {v['localisation_within_300m']:.3f}" if "localisation_within_300m" in v else "")
            for c, v in out[net].items()), flush=True)
    return out


# --------------------------------------------------------------------------- walk
def check_walk():
    wn = load("synthetic")
    J = wn.junction_name_list
    idx = {n: k for k, n in enumerate(J)}
    h0, q0 = simulate(wn)
    L, _ = grounded_laplacian(wn, h0, q0, J)
    G = np.linalg.inv(L)
    diag = np.diag(L)
    P = np.where(np.eye(len(J)) > 0, 0.0, -L / diag[:, None])          # junction-to-junction moves
    p_abs = 1 - P.sum(1)                                                 # absorption at a fixed-head node
    rng = np.random.default_rng(1)
    cum = np.cumsum(np.column_stack([P, p_abs]), 1)
    out = {}
    probe = ["T1", "A1", "M", "G0_2", "G2_3", "G3_0", "G5_5", "D0", "D7"]
    for leak in ("G3_0", "G2_3", "A1", "D3"):
        j = idx[leak]
        sims = {}
        for area in (1e-5, 1e-4, 1e-3):
            h, _ = simulate(wn, {leak: area})
            dh = (h0 - h)[J].values
            sims[area] = dh / dh[j]
        rows = {}
        for n in probe:
            i = idx[n]
            pos = np.full(20000, i); alive = np.ones(20000, bool); hit = np.zeros(20000, bool)
            for _ in range(20000):
                if not alive.any():
                    break
                a = np.where(alive)[0]
                nxt = (rng.random(len(a))[:, None] > cum[pos[a]]).sum(1)
                absorbed = nxt >= len(J)
                pos[a[~absorbed]] = nxt[~absorbed]
                alive[a[absorbed]] = False
                reached = alive & (pos == j)
                hit |= reached; alive &= ~reached
            if i == j:
                mc = 1.0
            else:
                mc = float(hit.mean())
            rows[n] = dict(G_ratio=float(G[i, j] / G[j, j]), monte_carlo=mc,
                           **{f"sim_{a*1e4:g}cm2": float(sims[a][i]) for a in sims})
        out[leak] = rows
        print(f"  walk leak {leak}: " + "; ".join(
            f"{n} G {r['G_ratio']:.3f} MC {r['monte_carlo']:.3f} sim1 {r['sim_1cm2']:.3f} sim10 {r['sim_10cm2']:.3f}"
            for n, r in rows.items()), flush=True)
    return out


# --------------------------------------------------------------------------- prop3
def check_prop3():
    net = "ky4"
    wn = load(net)
    lib = np.load(f"lib_{net}_0.0001.npz")
    J, cand, R = list(lib["J"]), list(lib["cand"]), lib["R"].astype(float)
    jidx = {n: k for k, n in enumerate(J)}
    sh, Gu = shadows(wn, J)
    k = max(sh, key=lambda n: len(sh[n]))
    inside = sorted(sh[k] | {k})
    sens_in = [jidx[n] for n in inside]
    outside = [n for n in J if n not in sh[k] and n != k]
    rng = np.random.default_rng(3)
    half = len(sens_in) // 2
    sens_mix = sens_in[:len(sens_in) - half] + [jidx[n] for n in rng.choice(outside, half, replace=False)]
    out_rows = [i for i, c in enumerate(cand) if c not in sh[k]]
    hop = {}

    def hops(a):
        if a not in hop:
            hop[a] = nx.single_source_shortest_path_length(Gu, a)
        return hop[a]

    res = {"shadow_entry": k, "shadow_size": len(sh[k]), "n_sensors": len(sens_in), "n_outside_leaks": len(out_rows)}
    for name, sens in (("all_inside", sens_in), ("half_moved_outside", sens_mix)):
        S = np.array(sens)
        Rs = R[out_rows][:, S]
        spread = (Rs.max(1) - Rs.min(1)) / np.maximum(np.abs(Rs).mean(1), 1e-12)
        D = Rs - Rs.mean(1, keepdims=True)
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        uniform = np.linalg.norm(D, axis=1) <= 1e-6 * np.linalg.norm(Rs, axis=1)   # residual proportional to 1
        C = Dn @ Dn.T
        C[np.ix_(uniform, uniform)] = 1.0                                        # all such leaks look alike
        g = nx.Graph(); g.add_nodes_from(range(len(out_rows)))
        ii, jj = np.where(np.triu(C, 1) > 0.999); g.add_edges_from(zip(ii, jj))
        obs = Rs + np.random.default_rng(5).normal(0, 0.01, Rs.shape)
        obs = obs - obs.mean(1, keepdims=True)
        est = np.argmax(obs @ Dn.T, 1)
        t1 = float(np.mean(est == np.arange(len(out_rows))))
        herr = float(np.mean([hops(cand[out_rows[i]]).get(cand[out_rows[e]], 0) for i, e in enumerate(est)]))
        res[name] = dict(max_relative_spread=float(spread.max()), median_relative_spread=float(np.median(spread)),
                         distinguishable_groups=int(nx.number_connected_components(g)), top1=t1, hop_error=herr)
        print(f"  prop3 {name}: spread max {spread.max():.2e}, groups {res[name]['distinguishable_groups']}/"
              f"{len(out_rows)}, top-1 {t1:.3f}, hop error {herr:.2f}", flush=True)
    return res


# --------------------------------------------------------------------------- prop4
_wn = None


def _init(net):
    global _wn
    _wn = load(net)


def _sim(task):
    mode, leaks = task
    if mode == "orifice_pdd":
        return simulate(_wn, leaks)
    # fixed leak outflows under demand-driven hydraulics: the setting of Proposition 4
    import copy as _copy
    w = _copy.deepcopy(_wn)
    w.options.time.duration = 0
    w.options.hydraulic.demand_model = "DD"
    for name in list(w.control_name_list):
        w.remove_control(name)
    for node, q in leaks.items():
        w.get_node(node).demand_timeseries_list[0].base_value += q
    res = wntr.sim.WNTRSimulator(w).run_sim()
    return res.node["head"].iloc[0], res.link["flowrate"].iloc[0]


def check_prop4():
    out = {}
    for net in ("synthetic", "ky4"):
        wn = load(net)
        J = wn.junction_name_list
        h0, q0 = simulate(wn)
        types, Gd = junction_types(wn, q0)
        jset = set(J)
        rng = np.random.default_rng(4)
        splits = []
        for n, t in types.items():
            if t == "split":
                ch = [v for v in Gd.successors(n) if v in jset]
                if len(ch) >= 2:
                    splits.append(tuple(rng.choice(ch, 2, replace=False)))
        rng.shuffle(splits)
        pairs = {"common_split": splits[:30],
                 "random": [tuple(rng.choice(J, 2, replace=False)) for _ in range(30)]}
        elev = {n: wn.get_node(n).elevation for n in J}
        qL = {n: 0.75 * 1e-3 * np.sqrt(2 * 9.81 * max(h0[n] - elev[n], 0.0)) for n in J}   # 10 cm^2 at baseline pressure
        out[net] = {}
        for mode in ("fixed_outflow_dd", "orifice_pdd"):
            tasks = []
            for kind, ps in pairs.items():
                for a, b in ps:
                    if mode == "orifice_pdd":
                        tasks += [(mode, {a: 1e-3}), (mode, {b: 1e-3}), (mode, {a: 1e-3, b: 1e-3})]
                    else:
                        tasks += [(mode, {a: qL[a]}), (mode, {b: qL[b]}), (mode, {a: qL[a], b: qL[b]})]
            base = _base(wn, mode)
            with Pool(config.WORKERS, initializer=_init, initargs=(net,)) as pool:
                sims = pool.map(_sim, tasks, chunksize=1)
            out[net][mode] = _prop4_summary(wn, J, base, sims, pairs, f"{net} {mode}")
    return out


def _base(wn, mode):
    global _wn
    _wn = wn
    return _sim((mode, {}))


def _prop4_summary(wn, J, base, sims, pairs, label):
    """Signed interaction at nodes, and the second-order cross term n(n-1) r |q0|^(n-2) fa fb on each pipe,
    split by whether both leaks add flow along the base direction (supply), both remove it (return,
    e.g. less inflow to a lower reservoir or a filling tank), or act in opposite directions."""
    from scipy.stats import spearmanr as _sp
    h0, q0 = base
    n_exp = 1.852
    pipes = [name for name, _ in wn.pipes()]
    ends = {name: (wn.get_link(name).start_node_name, wn.get_link(name).end_node_name) for name in pipes}
    Q0 = np.array([q0[p] for p in pipes])
    sgn = np.where(Q0 >= 0, 1.0, -1.0)
    dH0 = np.array([abs(h0[ends[p][0]] - h0[ends[p][1]]) for p in pipes])
    ok = np.abs(Q0) > 1e-6
    curv = np.where(ok, n_exp * (n_exp - 1) * dH0 / np.maximum(Q0 ** 2, 1e-300), 0.0)   # phi'' in base orientation
    res = {}
    t = 0
    for kind, ps in pairs.items():
        node_pos, node_rel, sup, ret, opp = [], [], [], [], []
        for _ in ps:
            (ha, qa), (hb, qb), (hab, _) = sims[t], sims[t + 1], sims[t + 2]
            t += 3
            da, db, dab = (h0 - ha)[J].values, (h0 - hb)[J].values, (h0 - hab)[J].values
            delta = dab - da - db
            node_pos.append(np.mean(delta > 1e-6))
            node_rel.append(np.sum(delta) / max(np.sum(da + db), 1e-12))
            fa = (np.array([qa[p] for p in pipes]) - Q0) * sgn
            fb = (np.array([qb[p] for p in pipes]) - Q0) * sgn
            cross = curv * fa * fb
            tot = np.sum(np.abs(cross)) + 1e-300
            sup.append(np.sum(cross[(fa > 0) & (fb > 0)]) / tot)
            ret.append(np.sum(cross[(fa < 0) & (fb < 0)]) / tot)
            opp.append(np.sum(np.abs(cross[fa * fb < 0])) / tot)
        node_rel, sup, ret = np.array(node_rel), np.array(sup), np.array(ret)
        rho = _sp(sup - ret, node_rel).correlation if len(ps) > 2 else float("nan")
        res[kind] = dict(n_pairs=len(ps), share_nodes_extra_drop=float(np.mean(node_pos)),
                         signed_interaction_rel_median=float(np.median(node_rel)),
                         share_pairs_positive=float(np.mean(node_rel > 0)),
                         cross_term_share_supply=float(np.mean(sup)), cross_term_share_return=float(np.mean(ret)),
                         cross_term_share_opposed=float(np.mean(opp)),
                         spearman_supply_minus_return_vs_interaction=float(rho))
        r = res[kind]
        print(f"  prop4 {label:26s} {kind:12s}: signed interaction {r['signed_interaction_rel_median']:+.3f} "
              f"(positive in {r['share_pairs_positive']:.0%} of pairs); cross term on supply {r['cross_term_share_supply']:.2f}, "
              f"return {r['cross_term_share_return']:.2f}, opposed {r['cross_term_share_opposed']:.2f}; "
              f"Spearman(supply-return, interaction) {rho:+.2f}", flush=True)
    return res


# --------------------------------------------------------------------------- battledim
def check_battledim():
    from datetime import timedelta
    from battledim_score import Scorer, T0
    sc = Scorer()
    res = json.load(open("bd_results_2019.json"))
    out = {"what_was_fitted": {
        "hydraulic_model": "nominal L-Town model, unchanged (no calibration of roughness, demands or leaks)",
        "influence_matrix": "linearised from one steady-state simulation of the nominal model",
        "residual_regression": "per-sensor regression on inflows, their squares, tank level and hour-of-day "
                               "harmonics, fitted on SCADA data before the first 2018 leak (to 2018-01-08 12:00)",
        "detector_settings": "threshold and candidate-set tolerance chosen on 2018 data, scored against the 2018 "
                             "leak list supplied by the organisers; four variants fixed before 2019 was evaluated",
        "2019_labels": "used only for scoring"}}

    def when(step):
        return (T0 + timedelta(minutes=5 * step)).strftime("%Y-%m-%d %H:%M")

    for cfg in ("same_argmax", "strict_argmax"):
        r = res[cfg]
        first = {}
        for row in r["rows"]:
            if row["leak"] and row["leak"] not in first:
                first[row["leak"]] = row
        leaks = []
        for lk in sc.leaks:
            if lk["end"] < 0:
                continue
            d = first.get(lk["link"])
            leaks.append(dict(link=lk["link"], type=lk["type"], start=when(lk["start"]), end=when(lk["end"]),
                              detected=d is not None, alarm=when(d["step"]) if d else None,
                              delay_days=round((d["step"] - max(lk["start"], 0)) / 288, 1) if d else None,
                              distance_m=round(d["distance"], 1) if d else None))
        fps = [dict(link=row["link"], time=when(row["step"])) for row in r["rows"] if row["leak"] is None and row["score"] < 0]
        out[cfg] = dict(score=r["total_score"], TP=r["TP"], FP=r["FP"], leaks=leaks, false_alarms=fps)
        print(f"  battledim {cfg}: score {r['total_score']:,.2f}, TP {r['TP']}, FP {r['FP']}")
        for x in leaks:
            print(f"    {x['link']:6s} {x['type']:12s} start {x['start']}  "
                  + (f"alarm {x['alarm']}  delay {x['delay_days']:6.1f} d  {x['distance_m']:6.1f} m" if x["detected"] else "not detected"))
        for x in fps:
            print(f"    false alarm {x['time']} at {x['link']}")
    return out


if __name__ == "__main__":
    for name in WANT:
        print(f"{name}:", flush=True)
        r = {"cap": check_cap, "walk": check_walk, "prop3": check_prop3, "prop4": check_prop4,
             "battledim": check_battledim}[name]()
        json.dump(r, open(f"rev_r5_{name}.json", "w"), indent=1)
