"""BattLeDIM step 1: pressure residuals from a hydraulics-shaped regression.

Each sensor's pressure is regressed on the network inflows (reservoir pipes p227,
p235 and PUMP_1), their squares (head loss grows ~ q^1.85), the tank level, and
hour-of-day harmonics, fitted on a leak-free reference window at the start of 2018
(before the first 2018 leak, p257, began on 2018-01-08 13:30).
Residual = observed - predicted (a leak makes it negative near the leak).
Outputs daily residual means for 2018 and 2019: bd_resid_daily.pkl
"""
import numpy as np
import pandas as pd

import config  # noqa: F401
D = "scada_"
REF_END = pd.Timestamp("2018-01-08 12:00")


def load(y):
    P = pd.read_pickle(D + f"{y}_pressures.pkl")
    F = pd.read_pickle(D + f"{y}_flows.pkl")
    L = pd.read_pickle(D + f"{y}_levels.pkl")
    return P, F, L


def features(F, L):
    h = F.index.hour + F.index.minute / 60
    X = pd.DataFrame(index=F.index)
    for c in F.columns:
        X[c] = F[c]
        X[c + "_sq"] = F[c] ** 2
    X["T1"] = L["T1"]
    for k in (1, 2):
        X[f"s{k}"] = np.sin(2 * np.pi * k * h / 24)
        X[f"c{k}"] = np.cos(2 * np.pi * k * h / 24)
    X["one"] = 1.0
    return X


P18, F18, L18 = load(2018)
P19, F19, L19 = load(2019)
X18, X19 = features(F18, L18), features(F19, L19)
ref = X18.index < REF_END
mu, sd = X18[ref].mean(), X18[ref].std().replace(0, 1)
mu["one"], sd["one"] = 0.0, 1.0
Z = lambda X: ((X - mu) / sd).values
A = Z(X18[ref])
lam = 1e-2 * len(A)
B = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ P18[ref].values)
R18 = P18 - Z(X18) @ B
R19 = P19 - Z(X19) @ B
print("reference rows", ref.sum(), " in-sample residual std (m):", float(R18[ref].values.std()))
print("2019 mean |residual| first week vs last week:",
      float(R19.iloc[:2016].abs().mean().mean()), float(R19.iloc[-2016:].abs().mean().mean()))
res = pd.concat([R18, R19])
daily = res.resample("D").mean()
night = res[(res.index.hour >= 1) & (res.index.hour < 5)].resample("D").mean()
pd.to_pickle(dict(daily=daily, night=night, all=res), "bd_resid.pkl")
print("saved; daily shape", daily.shape)
