"""Daily early-warning indicators from BattLeDIM residuals (see ews_protocol.md).

Output: ews_features.pkl with raw daily indicators and causal z-scores for 2018-2019.
"""
import numpy as np
import config  # noqa: F401
import pandas as pd

res = pd.read_pickle("bd_resid.pkl")["all"]            # 5-min residuals, 2018-2019, 33 sensors
ref = res.loc[: "2018-01-08 12:00"]
s = ref.std().values
days = pd.date_range(res.index[0].normalize(), res.index[-1].normalize(), freq="D")
X = res.values
day_of = res.index.normalize()


def gaussian_te(Z):
    """Lag-1 Gaussian transfer entropy j->i for all ordered pairs (rows/cols = sensors).
    TE = 0.5 log( var(x_i(t)|x_i(t-1)) / var(x_i(t)|x_i(t-1), x_j(t-1)) )."""
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    a, b = Z[1:], Z[:-1]                                   # x(t), x(t-1)
    n = len(a)
    C_ab = a.T @ b / n                                     # cov(x_i(t), x_j(t-1))
    C_bb = b.T @ b / n                                     # cov(x_i(t-1), x_j(t-1))
    r_ii = np.diag(C_ab)                                   # corr(x_i(t), x_i(t-1))
    v1 = 1 - r_ii ** 2                                     # residual var given own past
    # partial regression on [x_i(t-1), x_j(t-1)]
    ci = r_ii[:, None]                                     # cov(x_i(t), x_i(t-1))
    cj = C_ab                                              # cov(x_i(t), x_j(t-1))
    rij = C_bb                                             # cov(x_i(t-1), x_j(t-1))
    det = 1 - rij ** 2
    explained = (ci ** 2 + cj ** 2 - 2 * ci * cj * rij) / np.maximum(det, 1e-9)
    v2 = np.maximum(1 - explained, 1e-9)
    te = 0.5 * np.log(np.maximum(v1[:, None], 1e-9) / v2)
    np.fill_diagonal(te, 0)
    return np.maximum(te, 0)                               # te[i, j] = j -> i


def graph_metrics(T):
    n = len(T)
    W = T.copy()
    # global efficiency with edge length 1/TE (directed shortest paths)
    L = np.where(W > 1e-6, 1.0 / np.maximum(W, 1e-6), np.inf)
    np.fill_diagonal(L, 0)
    D = L.copy()
    for k in range(n):                                     # Floyd-Warshall, n = 33
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    off = ~np.eye(n, dtype=bool)
    eff = np.mean(1.0 / D[off])
    S = (W + W.T) / 2
    Lap = np.diag(S.sum(1)) - S
    ev = np.linalg.eigvalsh(Lap)
    lam2 = ev[1]
    spec = np.abs(np.linalg.eigvalsh(S)).sum()
    return dict(TE_MEAN=W[off].mean(), EFF=eff, LAMBDA2=lam2, SPEC=spec)


rows = []
for d in days:
    m = day_of == d
    Z = X[m]
    if len(Z) < 200:
        continue
    E = ((Z / s) ** 2).sum(1)
    dE = np.diff(E)
    osc = np.mean(np.sign(dE[1:]) != np.sign(dE[:-1]))
    zc = Z - Z.mean(0)
    var = np.mean(zc.var(0))
    ac1 = np.mean([np.corrcoef(zc[1:, i], zc[:-1, i])[0, 1] for i in range(Z.shape[1])])
    row = dict(day=d, ENERGY=E.mean(), OSC=osc, VAR=var, AC1=ac1, MEANRES=Z.mean(0))
    row.update(graph_metrics(gaussian_te(Z)))
    rows.append(row)
F = pd.DataFrame(rows).set_index("day")

# REACT: causal drop statistic of the detector
M = np.vstack(F["MEANRES"].values)
react = np.full(len(F), np.nan)
for t in range(60, len(F)):
    base = M[t - 60:t - 7].mean(0)
    rec = M[t - 7:t + 1].mean(0)
    react[t] = np.linalg.norm(np.maximum(base - rec, 0))
F["REACT"] = react
# DESCENT: 7-day slope of daily energy
e = F["ENERGY"].values
F["DESCENT"] = [np.polyfit(np.arange(7), e[t - 6:t + 1], 1)[0] if t >= 6 else np.nan for t in range(len(F))]
F = F.drop(columns="MEANRES")

IND = ["REACT", "VAR", "AC1", "ENERGY", "DESCENT", "OSC", "TE_MEAN", "EFF", "LAMBDA2", "SPEC"]
ABS = {"OSC", "TE_MEAN", "EFF", "LAMBDA2", "SPEC"}
Zs = pd.DataFrame(index=F.index)
for c in IND:
    v = F[c].values
    z = np.full(len(v), np.nan)
    for t in range(60, len(v)):
        w = v[t - 60:t]
        w = w[np.isfinite(w)]
        if len(w) > 20 and np.std(w) > 0:
            z[t] = (v[t] - np.mean(w)) / np.std(w)
    Zs[c] = np.abs(z) if c in ABS else z
pd.to_pickle(dict(raw=F, z=Zs), "ews_features.pkl")
print(F.describe().T[["mean", "std"]])
print("days with z:", int(Zs.notna().all(axis=1).sum()))
