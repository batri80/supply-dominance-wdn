"""Revision analysis R4 (rev_protocol.md): generalisation of NEET-REG and checks of NEET-2PT.

Uses the X2 calibration and test sets only (no new simulation).
Usage: python sfp2_gen.py
Needs sfp2_<net>_s<seed>.npy for both networks. Writes rev_r4_failure_checks.json.
"""
import json
import numpy as np
from scipy.stats import spearmanr
import config  # noqa: F401
from core import simulate
from networks import load

PMIN, INSERV, A0, A1, AMAX = 20.0, 22.0, 2e-4, 4e-4, 2e-2
SEEDS = {"ltown": (3026, 4026), "ky4": (3027, 4027)}
NETS = list(SEEDS)
FEATS = ["intercept", "log_NEET_2PT", "log_NEET_LIN", "PSI_N", "TOTAL"]
rng = np.random.default_rng(2026)


def in_service(net):
    wn = load(net)
    J = wn.junction_name_list
    elev = np.array([wn.get_node(n).elevation for n in J])
    h0, _ = simulate(wn)
    return (h0[J].values - elev) >= INSERV, {n: k for k, n in enumerate(J)}


def two_point(s, serv, bounds=(0.5, 3.0), min_drop=0.0):
    p0, pA0, pA1 = s["p0"], s["pA0"], s["pA1"]
    dh0 = np.maximum(p0 - pA0, 1e-9)
    dh1 = np.maximum(p0 - pA1, 1e-9)
    braw = np.log(dh1 / dh0) / np.log(A1 / A0)
    b = np.clip(braw, *bounds)
    c = dh0 / A0 ** b
    M = np.maximum(p0 - PMIN, 1e-6)
    ok = serv & (dh0 >= min_drop)
    Ai = np.where(ok, (M / c) ** (1.0 / b), np.inf)
    clipped = (braw < bounds[0]) | (braw > bounds[1])
    return Ai, clipped


def linear(s, serv, idx):
    p0, pA0 = s["p0"], s["pA0"]
    dh0 = np.maximum(p0 - pA0, 1e-9)
    m = pA0 - PMIN
    r = 1 + np.maximum(m, 0) / dh0
    jd = idx[s["j"]]
    pj = pA0[jd] - dh0[jd] * (r - 1)
    A = np.where(pj > 0, A0 * r * np.sqrt(pA0[jd] / np.maximum(pj, 1e-9)), np.inf)
    d = (dh0 / np.maximum(m, 1e-6))[serv]
    return float(A[serv].min()), float(d.std() / d.mean()), float(d.mean())


def table(S, serv, idx):
    rows = []
    for s in S:
        if s["censored"]:
            continue
        Ai, clipped = two_point(s, serv)
        lin, psi, tot = linear(s, serv, idx)
        a = int(np.argmin(Ai))
        rows.append(dict(A=s["Astar"], two=float(Ai.min()), lin=lin, psi=psi, tot=tot, first=s["first"], arg=a,
                         near=set(np.where(Ai <= 1.05 * Ai.min())[0].tolist()), top5=list(np.argsort(Ai)[:5]),
                         argmin_clipped=bool(clipped[a]), frac_clipped=float(clipped[serv].mean())))
    return rows


def design(T):
    cap = 0.0                                        # log(1 m^2): cap infinite estimates
    two = np.array([r["two"] for r in T]); lin = np.array([r["lin"] for r in T])
    return np.column_stack([np.ones(len(T)), np.minimum(np.log(two), cap), np.minimum(np.log(lin), cap),
                            [r["psi"] for r in T], [r["tot"] for r in T]])


def fit(T):
    X, y = design(T), np.log([r["A"] for r in T])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = (res @ res / dof) * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


def score(pred, A):
    pred, A = np.asarray(pred, float), np.asarray(A, float)
    rel = np.abs(pred - A) / A
    return dict(n=int(len(A)), median_rel_error=float(np.median(rel)), within_50pct=float(np.mean(rel < 0.5)),
                spearman=float(spearmanr(pred, A).correlation), median_log_ratio=float(np.median(np.log(pred / A))))


data, raw, serv_of, idx_of = {}, {}, {}, {}
for net in NETS:
    serv, idx = in_service(net)
    serv_of[net], idx_of[net] = serv, idx
    c, t = SEEDS[net]
    raw[net] = {k: list(np.load(f"sfp2_{net}_s{s}.npy", allow_pickle=True)) for k, s in (("cal", c), ("test", t))}
    data[net] = {k: table(v, serv, idx) for k, v in raw[net].items()}

out = {}
for net in NETS:
    cal, test = data[net]["cal"], data[net]["test"]
    A = np.array([r["A"] for r in test])
    o = {"n_calibration": len(cal), "n_test": len(test),
         "censored": {k: int(sum(s["censored"] for s in v)) for k, v in raw[net].items()}}
    beta, se = fit(cal)
    o["coefficients"] = {f: dict(estimate=float(b), se=float(e)) for f, b, e in zip(FEATS, beta, se)}
    o["NEET_REG"] = score(np.exp(design(test) @ beta), A)
    o["NEET_2PT"] = score([r["two"] for r in test], A)
    # learning curve
    lc = {}
    for n in sorted({x for x in (10, 20, 40, 80, 160) if x < len(cal)} | {len(cal)}):
        meds = []
        for _ in range(200 if n < len(cal) else 1):
            sub = [cal[i] for i in rng.choice(len(cal), n, replace=False)]
            b_, _ = fit(sub)
            meds.append(score(np.exp(design(test) @ b_), A)["median_rel_error"])
        lc[str(n)] = dict(median=float(np.median(meds)), q25=float(np.percentile(meds, 25)), q75=float(np.percentile(meds, 75)))
    o["learning_curve_median_rel_error"] = lc
    # exponent bounds and small drops
    o["exponent_bounds"] = {}
    for name, bnd in (("[0.5,3]", (0.5, 3.0)), ("[0.25,5]", (0.25, 5.0)), ("none", (1e-3, 1e3))):
        pred = [two_point(s, serv_of[net], bnd)[0].min() for s in raw[net]["test"] if not s["censored"]]
        o["exponent_bounds"][name] = score(pred, A)
    pred = [two_point(s, serv_of[net], min_drop=1e-3)[0].min() for s in raw[net]["test"] if not s["censored"]]
    o["exclude_drops_below_1mm"] = score(pred, A)
    o["share_scenarios_estimate_from_clipped_fit"] = float(np.mean([r["argmin_clipped"] for r in test]))
    o["mean_share_junction_fits_clipped"] = float(np.mean([r["frac_clipped"] for r in test]))
    # first failing junction
    o["first_failing"] = dict(top1=float(np.mean([r["arg"] == r["first"] for r in test])),
                              top1_with_near_ties=float(np.mean([r["first"] in r["near"] for r in test])),
                              top5=float(np.mean([r["first"] in r["top5"] for r in test])),
                              mean_near_tie_set_size=float(np.mean([len(r["near"]) for r in test])))
    # error by tercile of A*
    q = np.percentile(A, [100 / 3, 200 / 3])
    two = np.array([r["two"] for r in test])
    o["NEET_2PT_by_Astar_tercile"] = {}
    for name, m in (("low", A <= q[0]), ("mid", (A > q[0]) & (A <= q[1])), ("high", A > q[1])):
        o["NEET_2PT_by_Astar_tercile"][name] = dict(range_cm2=[float(A[m].min() * 1e4), float(A[m].max() * 1e4)],
                                                     **score(two[m], A[m]))
    # censored scenarios: does NEET-2PT wrongly predict failure below 200 cm^2?
    cens = [s for s in raw[net]["test"] + raw[net]["cal"] if s["censored"]]
    o["censored_predicted_below_200cm2"] = (float(np.mean([two_point(s, serv_of[net])[0].min() < AMAX for s in cens]))
                                            if cens else None)
    out[net] = o

# transfer between networks and a pooled fit
for a, b in (("ltown", "ky4"), ("ky4", "ltown")):
    beta, _ = fit(data[a]["cal"])
    A = np.array([r["A"] for r in data[b]["test"]])
    out[b][f"NEET_REG_fitted_on_{a}"] = score(np.exp(design(data[b]["test"]) @ beta), A)
beta, _ = fit(data["ltown"]["cal"] + data["ky4"]["cal"])
for n in NETS:
    A = np.array([r["A"] for r in data[n]["test"]])
    out[n]["NEET_REG_pooled_fit"] = score(np.exp(design(data[n]["test"]) @ beta), A)

for n in NETS:
    o = out[n]
    print(f"\n{n}: calibration {o['n_calibration']}, test {o['n_test']}, censored {o['censored']}")
    print("  coefficients: " + ", ".join(f"{k} {v['estimate']:+.3f} ({v['se']:.3f})" for k, v in o["coefficients"].items()))
    print(f"  NEET-2PT {o['NEET_2PT']['median_rel_error']:.3f}   NEET-REG {o['NEET_REG']['median_rel_error']:.3f}   "
          f"transfer {o[f'NEET_REG_fitted_on_{[x for x in NETS if x != n][0]}']['median_rel_error']:.3f}   "
          f"pooled {o['NEET_REG_pooled_fit']['median_rel_error']:.3f}")
    print("  learning curve (median rel. error): " + ", ".join(f"n={k}: {v['median']:.3f}" for k, v in o["learning_curve_median_rel_error"].items()))
    print("  exponent bounds: " + ", ".join(f"{k} {v['median_rel_error']:.3f}" for k, v in o["exponent_bounds"].items())
          + f";  drops < 1 mm excluded {o['exclude_drops_below_1mm']['median_rel_error']:.3f}")
    print(f"  estimate from a clipped fit in {o['share_scenarios_estimate_from_clipped_fit']:.1%} of scenarios; "
          f"{o['mean_share_junction_fits_clipped']:.1%} of junction fits clipped on average")
    f = o["first_failing"]
    print(f"  first failing junction: top-1 {f['top1']:.2f}, with near-ties {f['top1_with_near_ties']:.2f}, top-5 {f['top5']:.2f}")
    print("  NEET-2PT by A* tercile: " + ", ".join(f"{k} ({v['range_cm2'][0]:.0f}-{v['range_cm2'][1]:.0f} cm2) {v['median_rel_error']:.3f}"
                                                   for k, v in o["NEET_2PT_by_Astar_tercile"].items()))
    print(f"  censored scenarios predicted to fail below 200 cm2: {o['censored_predicted_below_200cm2']}")
json.dump(out, open("rev_r4_failure_checks.json", "w"), indent=1)
