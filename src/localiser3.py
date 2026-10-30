"""Revision experiment R3 (rev_protocol.md): scenario-level uncertainty for X1.

Retrains hybrid and hybrid_nodom exactly as in localiser2.py (same data, seeds and
settings), keeps the per-scenario results of every method, and reports t-intervals over
seeds, paired scenario bootstraps per seed, McNemar tests, and a hierarchical bootstrap
over seeds and scenarios together.

Usage: python localiser3.py ltown|ky4 [n_seeds=5] [epochs=30]
Needs the same inputs as localiser2.py. Writes rev_r3_<net>.json (resumes per seed).
"""
import json
import os
import numpy as np
from scipy.stats import binomtest
import localiser2 as L                      # reads sys.argv; builds network quantities and data loaders

NET, N_SEEDS = L.NET, L.N_SEEDS
OUTFILE = f"rev_r3_{NET}.json"
NAMES = ("hybrid_nodom", "hybrid")
B = 2000


def per_scenario(score, nodes):
    w, t1, d = [], [], []
    for i in range(len(score)):
        true = [n for n in nodes[i] if n]
        est = L.J[int(np.argmax(score[i]))]
        dist = min(L.dist(est, t) for t in true)
        w.append(int(dist <= L.RADIUS)); t1.append(int(est in true)); d.append(float(dist))
    return dict(within300=w, top1=t1, dist_m=d)


def tint(v):
    return L.ci(v)


def boot_paired(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    s = np.array([np.mean(a[i] - b[i]) for i in (rng.integers(0, n, n) for _ in range(B))])
    return [float(np.mean(a - b)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def boot_hier(H, base, rng):
    """H: seeds x scenarios (0/1); base: scenarios (0/1) or seeds x scenarios."""
    H = np.asarray(H, float)
    base = np.asarray(base, float)
    if base.ndim == 1:
        base = np.broadcast_to(base, H.shape)
    S, n = H.shape
    vals = []
    for _ in range(B):
        si = rng.integers(0, S, S)
        ci = rng.integers(0, n, n)
        vals.append(np.mean(H[np.ix_(si, ci)] - base[np.ix_(si, ci)]))
    return [float(np.mean(H - base)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mcnemar(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n01, n10 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(hybrid_only=n01, prior_only=n10, p=float(p))


if __name__ == "__main__":
    obs_tr, R_tr, nodes_tr, _ = L.load_set(0, L.N_TRAIN)
    obs_te, _, nodes_te, _ = L.load_set(1, L.N_TEST)
    conc = np.array([sum(1 for n in ns if n) > 1 for ns in nodes_te])
    print(f"{NET}: {len(obs_tr)} training and {len(obs_te)} test scenarios ({conc.sum()} concurrent), "
          f"{N_SEEDS} seeds", flush=True)
    res = json.load(open(OUTFILE)) if os.path.exists(OUTFILE) else {"seeds": {}}
    if "deterministic" not in res:
        nearest = np.zeros((len(obs_te), L.nJ)); nearest[np.arange(len(obs_te)), L.s_idx[np.argmax(obs_te, 1)]] = 1
        res["deterministic"] = {"nearest_sensor": per_scenario(nearest, nodes_te),
                                "graph_interp": per_scenario(L.field(obs_te), nodes_te),
                                "linear": per_scenario(L.linear_scores(obs_te), nodes_te)}
        res["concurrent"] = conc.astype(int).tolist()
        json.dump(res, open(OUTFILE, "w"))
    Ttr, Ftr = L.targets(nodes_tr), L.to_t(R_tr / L.scale(obs_tr))
    feats = {}
    for seed in range(N_SEEDS):
        sd = res["seeds"].setdefault(str(seed), {})
        for name in NAMES:
            if name in sd:
                continue
            prior, dom, _ = L.METHODS[name]
            if (prior, dom) not in feats:
                feats[(prior, dom)] = (L.features(obs_tr, prior, dom), L.features(obs_te, prior, dom))
            Xtr, Xte = feats[(prior, dom)]
            m = L.train(name, seed, Xtr, Ttr, Ftr)
            sd[name] = per_scenario(L.scores_of(m, Xte), nodes_te)
            print(f"  seed {seed} {name:13s} within 300 m {np.mean(sd[name]['within300']):.3f}", flush=True)
            json.dump(res, open(OUTFILE, "w"))

    rng = np.random.default_rng(2026)
    seeds = sorted(res["seeds"], key=int)
    lin = np.array(res["deterministic"]["linear"]["within300"])
    H = np.array([res["seeds"][s]["hybrid"]["within300"] for s in seeds])
    N = np.array([res["seeds"][s]["hybrid_nodom"]["within300"] for s in seeds])
    summ = {"per_seed": {}, "deterministic": {}}
    for k, v in res["deterministic"].items():
        w = np.array(v["within300"])
        summ["deterministic"][k] = dict(all=float(w.mean()), single=float(w[~conc].mean()), concurrent=float(w[conc].mean()),
                                        top1=float(np.mean(v["top1"])), median_dist_m=float(np.median(v["dist_m"])))
    for i, s in enumerate(seeds):
        row = {}
        for name in NAMES:
            v = res["seeds"][s][name]; w = np.array(v["within300"])
            row[name] = dict(all=float(w.mean()), single=float(w[~conc].mean()), concurrent=float(w[conc].mean()),
                             top1=float(np.mean(v["top1"])), median_dist_m=float(np.median(v["dist_m"])))
        row["hybrid_minus_prior_bootstrap"] = boot_paired(H[i], lin, rng)
        row["hybrid_minus_nodom_bootstrap"] = boot_paired(H[i], N[i], rng)
        row["mcnemar_hybrid_vs_prior"] = mcnemar(H[i], lin)
        summ["per_seed"][s] = row
    summ["hybrid_minus_prior"] = dict(t_over_seeds=tint(H.mean(1) - lin.mean()), hierarchical=boot_hier(H, lin, rng))
    summ["hybrid_minus_nodom"] = dict(t_over_seeds=tint(H.mean(1) - N.mean(1)), hierarchical=boot_hier(H, N, rng))
    summ["nodom_minus_prior"] = dict(t_over_seeds=tint(N.mean(1) - lin.mean()), hierarchical=boot_hier(N, lin, rng))
    for grp, mask in (("single", ~conc), ("concurrent", conc)):
        summ[f"hybrid_minus_prior_{grp}"] = boot_hier(H[:, mask], lin[mask], rng)
    lo = summ["hybrid_minus_prior"]["hierarchical"][1]
    lo2 = summ["hybrid_minus_nodom"]["hierarchical"][1]
    summ["decision"] = {"hybrid_improves_on_prior": bool(lo > 0), "dominance_flags_help": bool(lo2 > 0)}
    res["summary"] = summ
    json.dump(res, open(OUTFILE, "w"))

    print(f"\n{NET}: per-seed share within 300 m (prior {lin.mean():.3f})")
    for s in seeds:
        r = summ["per_seed"][s]
        m_, a, b = r["hybrid_minus_prior_bootstrap"]
        print(f"  seed {s}: hybrid {r['hybrid']['all']:.3f}  nodom {r['hybrid_nodom']['all']:.3f}  "
              f"hybrid-prior {m_:+.3f} [{a:+.3f}, {b:+.3f}]  McNemar p {r['mcnemar_hybrid_vs_prior']['p']:.3g}")
    for k in ("hybrid_minus_prior", "hybrid_minus_nodom", "nodom_minus_prior"):
        t, h = summ[k]["t_over_seeds"], summ[k]["hierarchical"]
        print(f"  {k:20s} t over seeds {t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]   "
              f"hierarchical {h[0]:+.3f} [{h[1]:+.3f}, {h[2]:+.3f}]")
    for grp in ("single", "concurrent"):
        h = summ[f"hybrid_minus_prior_{grp}"]
        print(f"  hybrid-prior, {grp:10s} hierarchical {h[0]:+.3f} [{h[1]:+.3f}, {h[2]:+.3f}]")
    print("  decision:", summ["decision"], flush=True)
