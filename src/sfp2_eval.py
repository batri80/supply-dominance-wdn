"""Experiment X2 (ext_protocol.md): evaluate nonlinear NEET estimates of the failure size A*.

Usage: python sfp2_eval.py [networks...]   (default: ltown ky4)
Needs sfp2_<net>_s<calibration seed>.npy and sfp2_<net>_s<test seed>.npy.
Writes ext_failure_size.json.
"""
import json
import sys
import numpy as np
from scipy.stats import spearmanr
import config  # noqa: F401
from core import simulate
from networks import load

PMIN, INSERV, A0, A1 = 20.0, 22.0, 2e-4, 4e-4
SEEDS = {"ltown": (3026, 4026), "ky4": (3027, 4027)}
NETS = sys.argv[1:] or ["ltown", "ky4"]


def in_service(net):
    wn = load(net)
    J = wn.junction_name_list
    elev = np.array([wn.get_node(n).elevation for n in J])
    h0, _ = simulate(wn)
    return (h0[J].values - elev) >= INSERV, {n: k for k, n in enumerate(J)}


def predictors(s, serv, idx):
    p0, pA0, pA1 = s["p0"], s["pA0"], s["pA1"]
    jdx = idx[s["j"]]                                    # the leak node
    dh0 = np.maximum(p0 - pA0, 1e-9)
    dh1 = np.maximum(p0 - pA1, 1e-9)
    # NEET_LIN (as in sfp_eval.py): linear extrapolation from A0 with the orifice-law correction
    m = pA0 - PMIN
    r = 1 + np.maximum(m, 0) / dh0
    pj = pA0[jdx] - dh0[jdx] * (r - 1)
    A_lin = np.where(pj > 0, A0 * r * np.sqrt(pA0[jdx] / np.maximum(pj, 1e-9)), np.inf)
    neet_lin = float(A_lin[serv].min())
    # NEET_2PT: per-node power law through the drops at A0 and A1
    b = np.clip(np.log(dh1 / dh0) / np.log(A1 / A0), 0.5, 3.0)
    c = dh0 / A0 ** b
    m0 = np.maximum(p0 - PMIN, 1e-6)
    A_i = (m0 / c) ** (1.0 / b)
    neet_2pt = float(A_i[serv].min())
    d = (dh0 / np.maximum(m, 1e-6))[serv]
    return neet_lin, neet_2pt, float(d.std() / d.mean()), float(d.mean())


def table(S, serv, idx):
    rows = [(s["Astar"],) + predictors(s, serv, idx) for s in S if not s["censored"]]
    return np.array(rows, float)                          # A*, lin, 2pt, psi, total


def design(T):
    cap = np.log(1.0)                                     # cap infinite estimates at 1 m^2 (far beyond 200 cm^2)
    return np.column_stack([np.ones(len(T)), np.minimum(np.log(T[:, 2]), cap), np.minimum(np.log(T[:, 1]), cap), T[:, 3], T[:, 4]])


def score(pred, A):
    rel = np.abs(pred - A) / A
    return dict(median_rel_error=float(np.median(rel)), spearman=float(spearmanr(pred, A).correlation),
                within_50pct=float(np.mean(rel < 0.5)))


out = {}
for net in NETS:
    cal_seed, test_seed = SEEDS[net]
    serv, idx = in_service(net)
    cal = table(np.load(f"sfp2_{net}_s{cal_seed}.npy", allow_pickle=True), serv, idx)
    test = table(np.load(f"sfp2_{net}_s{test_seed}.npy", allow_pickle=True), serv, idx)
    beta, *_ = np.linalg.lstsq(design(cal), np.log(cal[:, 0]), rcond=None)
    reg = np.exp(design(test) @ beta)
    out[net] = dict(n_calibration=int(len(cal)), n_test=int(len(test)),
                    regression=dict(zip(["intercept", "log_NEET_2PT", "log_NEET_LIN", "PSI_N", "TOTAL"], map(float, beta))),
                    NEET_LIN=score(test[:, 1], test[:, 0]), NEET_2PT=score(test[:, 2], test[:, 0]), NEET_REG=score(reg, test[:, 0]))
    print(f"\n{net}: {len(cal)} calibration and {len(test)} test scenarios (uncensored)")
    for k in ("NEET_LIN", "NEET_2PT", "NEET_REG"):
        v = out[net][k]
        print(f"  {k:9s} median relative error {v['median_rel_error']:.3f}   Spearman {v['spearman']:.3f}   within 50% {v['within_50pct']:.2f}")
verdict = {}
for k in ("NEET_2PT", "NEET_REG"):
    verdict[k] = all(out[n][k]["median_rel_error"] < 0.5 and out[n][k]["median_rel_error"] < out[n]["NEET_LIN"]["median_rel_error"]
                     for n in out)
out["decision"] = {k: ("solves the problem on all networks" if v else "does not meet the pre-specified rule") for k, v in verdict.items()}
print("\nPre-specified decision:", out["decision"])
json.dump(out, open("ext_failure_size.json", "w"), indent=1)
