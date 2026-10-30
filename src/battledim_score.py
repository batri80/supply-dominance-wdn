"""Python port of the official BattLeDIM 2020 scoring algorithm (KIOS, EUPL-1.1).

Mirrors Scoring_Algorithm.m and functions/scoring_function.m:
  - a detection is a (pipe ID, start time); detections before 2019 are ignored;
  - distance x = 0 for the same pipe, else the shortest pipe-length path between
    the two pipes' end nodes plus half of each pipe's length;
  - a detection inside a leak's [start, end] window with x <= 300 m scores the
    water saved from detection time to the end of 2019 (EUR 0.80/m3) minus a crew
    cost of (x/300)*EUR 500; otherwise it is a false positive costing EUR 500;
  - detections are processed in time order; a detection matching only leaks
    already detected scores 0; a new true detection is credited to the closest
    not-yet-detected matching leak.
"""
import os
from datetime import datetime
import config
import numpy as np
import pandas as pd
import networkx as nx
import wntr

BASE = config.BATTLEDIM_SCORING + os.sep
T0 = datetime(2019, 1, 1)
XMAX, COST_WATER, COST_CREW = 300.0, 0.80, 500.0


def steps(ts):
    """5-minute steps from 2019-01-01 00:00 (negative in 2018), as time2steps.m."""
    t = pd.Timestamp(ts).to_pydatetime()
    return int(round((t - T0).total_seconds() / 300))


def load_leaks():
    leaks = []
    for line in open(BASE + "competition_data/leakages_info.yalm"):
        if line.startswith("#") or not line.strip():
            continue
        f = [x.strip() for x in line.split(",")]
        ts = pd.read_excel(BASE + f"competition_leakages/Leak_{f[0]}.xlsx", sheet_name=1).iloc[:, 1].values
        leaks.append(dict(link=f[0], start=steps(f[1]), end=steps(f[2]), diameter=float(f[3]),
                          type=f[4], series=ts.astype(float)))
    return leaks


class Scorer:
    def __init__(self, inp=config.LTOWN_INP):
        self.wn = wntr.network.WaterNetworkModel(inp)
        G = nx.Graph()
        for name, l in self.wn.links():
            length = getattr(l, "length", 0.0)
            G.add_edge(l.start_node_name, l.end_node_name, w=length)
        self.G = G
        self.leaks = load_leaks()
        self._d = {}

    def _dist_nodes(self, a):
        if a not in self._d:
            self._d[a] = nx.single_source_dijkstra_path_length(self.G, a, weight="w")
        return self._d[a]

    def pipe_distance(self, p, q):
        if p == q:
            return 0.0
        lp, lq = self.wn.get_link(p), self.wn.get_link(q)
        ends_p = (lp.start_node_name, lp.end_node_name)
        ends_q = (lq.start_node_name, lq.end_node_name)
        best = min(self._dist_nodes(a).get(b, np.inf) for a in ends_p for b in ends_q)
        return best + getattr(lp, "length", 0.0) / 2 + getattr(lq, "length", 0.0) / 2

    def score_one(self, leak, link, td):
        x = self.pipe_distance(leak["link"], link)
        if td < 0:
            return 0.0, False, x
        if leak["start"] <= td <= leak["end"] and abs(x) <= XMAX:
            t = max(td, 1)
            saved = leak["series"][t - 1:].sum() / 12.0      # MATLAB 1-based td:end
            return saved * COST_WATER - (x / XMAX) * COST_CREW, True, x
        return -COST_CREW, False, x

    def score(self, detections):
        """detections: list of (pipe_id, timestamp). Returns a summary dict."""
        dets = [(link, steps(t)) for link, t in detections]
        dets = [d for d in dets if d[1] >= 0]
        dets.sort(key=lambda d: d[1])
        detected, total, fp, delays, rows = [], 0.0, 0, [], []
        for link, td in dets:
            res = [self.score_one(lk, link, td) for lk in self.leaks]
            ok = [k for k, r in enumerate(res) if r[1]]
            if ok:
                new = [k for k in ok if k not in detected]
                if not new:
                    s, who = 0.0, None
                else:
                    who = min(new, key=lambda k: res[k][2])
                    s = res[who][0]
                    detected.append(who)
                    delays.append(td - self.leaks[who]["start"])
            else:
                fp += 1
                s, who = -COST_CREW, None
            total += s
            rows.append(dict(link=link, step=td, score=s, leak=None if who is None else self.leaks[who]["link"],
                             distance=None if who is None else res[who][2]))
        return dict(total_score=round(total, 2), TP=len(detected), FP=fp, FN=len(self.leaks) - len(detected),
                    detected=[self.leaks[k]["link"] for k in detected], rows=rows)


def read_submission(path):
    out = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        a, b = [x.strip() for x in line.split(",")[:2]]
        out.append((a, b))
    return out


if __name__ == "__main__":
    # Validation against the reference reports shipped with the official MATLAB code.
    sc = Scorer()
    expected = {"PERFECT_TEAM": 523153.82, "test_team": 523153.82, "test_team_multiple": 9547.81}
    ok = True
    for team, value in expected.items():
        r = sc.score(read_submission(BASE + f"SUBMITTED_files/{team}.txt"))
        match = abs(r["total_score"] - value) < 0.01
        ok &= match
        print(f"{team:20s} score {r['total_score']:>12,.2f}  expected {value:>12,.2f}  TP {r['TP']} FP {r['FP']}  {'OK' if match else 'MISMATCH'}")
    print("Scoring port validated." if ok else "Scoring port does NOT reproduce the reference reports.")
    raise SystemExit(0 if ok else 1)
