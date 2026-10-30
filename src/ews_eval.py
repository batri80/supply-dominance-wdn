"""Evaluate early-warning indicators exactly as in ews_protocol.md.
Usage: python ews_eval.py 2018   (tuning year: thresholds, combined model)
       python ews_eval.py 2019   (test year: frozen choices from the 2018 run)
"""
import json
import sys
import numpy as np
import config  # noqa: F401
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

YEAR = int(sys.argv[1])
IND = ["REACT", "VAR", "AC1", "ENERGY", "DESCENT", "OSC", "TE_MEAN", "EFF", "LAMBDA2", "SPEC"]
CSTNIG = ["ENERGY", "DESCENT", "OSC", "TE_MEAN", "EFF", "LAMBDA2", "SPEC"]
Z = pd.read_pickle("ews_features.pkl")["z"]

LEAKS = {
    2018: [("p257", "2018-01-08", "2018-12-31", "incipient"), ("p461", "2018-01-23", "2018-04-02", "incipient"),
           ("p232", "2018-01-31", "2018-02-10", "incipient"), ("p427", "2018-02-13", "2018-12-31", "incipient"),
           ("p673", "2018-03-05", "2018-03-23", "abrupt"), ("p628", "2018-05-02", "2018-05-29", "incipient"),
           ("p538", "2018-05-18", "2018-06-02", "abrupt"), ("p866", "2018-06-01", "2018-06-12", "abrupt"),
           ("p31", "2018-06-28", "2018-08-12", "incipient"), ("p654", "2018-07-05", "2018-12-31", "incipient"),
           ("p810", "2018-07-28", "2018-12-31", "incipient"), ("p183", "2018-08-07", "2018-09-01", "abrupt"),
           ("p158", "2018-10-06", "2018-10-23", "abrupt"), ("p369", "2018-10-26", "2018-11-08", "abrupt")],
    2019: [("p523", "2019-01-15", "2019-02-01", "abrupt"), ("p827", "2019-01-24", "2019-02-07", "abrupt"),
           ("p280", "2019-02-10", "2019-12-31", "abrupt"), ("p653", "2019-03-03", "2019-05-05", "incipient"),
           ("p710", "2019-03-24", "2019-12-31", "abrupt"), ("p514", "2019-04-02", "2019-05-23", "abrupt"),
           ("p331", "2019-04-20", "2019-12-31", "abrupt"), ("p193", "2019-05-19", "2019-12-31", "incipient"),
           ("p277", "2019-05-30", "2019-12-31", "incipient"), ("p142", "2019-06-12", "2019-07-17", "abrupt"),
           ("p680", "2019-07-10", "2019-12-31", "abrupt"), ("p586", "2019-07-26", "2019-09-16", "incipient"),
           ("p721", "2019-08-02", "2019-12-31", "incipient"), ("p800", "2019-08-16", "2019-10-01", "incipient"),
           ("p123", "2019-09-13", "2019-12-31", "incipient"), ("p455", "2019-10-03", "2019-12-31", "incipient"),
           ("p762", "2019-10-09", "2019-12-31", "incipient"), ("p426", "2019-10-25", "2019-12-31", "abrupt"),
           ("p879", "2019-11-20", "2019-12-31", "incipient")]}
# repairs that fall in each year (end dates before the year's last day)
REPAIRS = {y: [pd.Timestamp(e) for _, _, e, _ in LEAKS[y] if not e.endswith("12-31")] for y in LEAKS}
REPAIRS[2019] += []   # 2018 leaks running into 2019 end on 2019-12-31 (no repair inside 2019)


def labels(year):
    z = Z.loc[str(year)].dropna()
    ons = [pd.Timestamp(s) for _, s, _, _ in LEAKS[year]]
    y, ev = {}, {}
    for d in z.index:
        recent = [k for k, o in enumerate(ons) if pd.Timedelta(0) <= d - o < pd.Timedelta(days=14)]
        if recent:
            y[d] = 1; ev[d] = recent[0]
            continue
        any30 = any(pd.Timedelta(0) <= d - o < pd.Timedelta(days=30) for o in ons)
        rep7 = any(pd.Timedelta(0) <= d - r < pd.Timedelta(days=7) for r in REPAIRS[year])
        if not any30 and not rep7:
            y[d] = 0; ev[d] = -1 - (d.dayofyear // 7)     # negative "event" = calendar week
    idx = list(y)
    return z.loc[idx], np.array([y[d] for d in idx]), np.array([ev[d] for d in idx])


def boot_auc(s, y, ev, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = {g: np.where(ev == g)[0] for g in np.unique(ev)}
    pos = [g for g in groups if g >= 0]; neg = [g for g in groups if g < 0]
    out = []
    for _ in range(n):
        gs = list(rng.choice(pos, len(pos))) + list(rng.choice(neg, len(neg)))
        ii = np.concatenate([groups[g] for g in gs])
        if len(set(y[ii])) == 2:
            out.append(roc_auc_score(y[ii], s[ii]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


z, y, ev = labels(YEAR)
print(f"{YEAR}: {int(y.sum())} early-phase days from {len(set(ev[ev>=0]))} onsets, {int((y==0).sum())} negative days")
res = {"year": YEAR, "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()), "auroc": {}}
for c in IND:
    s = z[c].values
    res["auroc"][c] = dict(auc=float(roc_auc_score(y, s)), ci=boot_auc(s, y, ev))

if YEAR == 2018:
    frozen = {"thresholds": {}, "models": {}}
    for c in IND:
        frozen["thresholds"][c] = float(np.percentile(z[c].values[y == 0], 100 * (1 - 1 / 30)))
    for name, cols in (("CST_NIG", CSTNIG), ("ALL", IND)):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(z[cols].values, y)
        frozen["models"][name] = dict(cols=cols, coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]))
    json.dump(frozen, open("ews_frozen_2018.json", "w"), indent=1)
else:
    frozen = json.load(open("ews_frozen_2018.json"))

for name, mdl in frozen["models"].items():
    s = z[mdl["cols"]].values @ np.array(mdl["coef"]) + mdl["intercept"]
    res["auroc"]["model_" + name] = dict(auc=float(roc_auc_score(y, s)), ci=boot_auc(s, y, ev))

# event lead time: days from onset to first alarm (within 60 days), per indicator
lead = {}
zall = Z.loc[str(YEAR)]
for c in IND:
    thr = frozen["thresholds"][c]
    ds = []
    for link, on, _, typ in LEAKS[YEAR]:
        o = pd.Timestamp(on)
        w = zall[c].loc[o: o + pd.Timedelta(days=60)].dropna()
        hit = w[w > thr]
        ds.append(None if hit.empty else int((hit.index[0] - o).days))
    lead[c] = dict(detected=sum(d is not None for d in ds), median_days=float(np.median([d for d in ds if d is not None])) if any(d is not None for d in ds) else None,
                   per_leak=ds)
res["lead"] = lead
json.dump(res, open(f"ews_results_{YEAR}.json", "w"), indent=1)
print(f"{'indicator':14s} AUROC  [95% CI]        alarms<=60d  median days")
for c, v in res["auroc"].items():
    l = lead.get(c, {})
    print(f"{c:14s} {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]   {l.get('detected','-'):>3}/{len(LEAKS[YEAR])}      {l.get('median_days','-')}")
