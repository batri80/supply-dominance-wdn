"""Test networks: a controlled synthetic asset network and the KY4 benchmark."""
import config  # noqa: F401  (first: sets warnings filter and the outputs folder)
import numpy as np
import wntr
from core import LIB


def synthetic(seed=0):
    """6x6 meshed grid fed by two reservoirs, a split-merge bypass on the feed
    trunk, and an 8-node dead-end branch hanging off one grid node."""
    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("R1", base_head=70.0, coordinates=(-3, 0))
    wn.add_reservoir("R2", base_head=62.0, coordinates=(8, 5))

    def J(name, xy):
        wn.add_junction(name, base_demand=rng.uniform(0.5e-3, 2.0e-3),
                        elevation=0.0, coordinates=xy)

    # feed trunk with a split-merge bypass (two parallel branches)
    J("T1", (-2, 0)); J("S", (-1, 0))            # S: split
    for k in range(3):
        J(f"A{k}", (-1 + 0.3 * (k + 1), 0.6))    # upper branch
        J(f"B{k}", (-1 + 0.3 * (k + 1), -0.6))   # lower branch
    J("M", (0.2, 0))                              # M: merge
    wn.add_pipe("pR1", "R1", "T1", 200, 0.40, 110)
    wn.add_pipe("pT1", "T1", "S", 200, 0.40, 110)
    prevA, prevB = "S", "S"
    for k in range(3):
        wn.add_pipe(f"pA{k}", prevA, f"A{k}", 150, 0.25, 100)
        wn.add_pipe(f"pB{k}", prevB, f"B{k}", 150, 0.25, 100)
        prevA, prevB = f"A{k}", f"B{k}"
    wn.add_pipe("pAM", prevA, "M", 150, 0.25, 100)
    wn.add_pipe("pBM", prevB, "M", 150, 0.25, 100)

    # meshed grid
    n = 6
    for i in range(n):
        for j in range(n):
            J(f"G{i}_{j}", (1 + i, j - 2.5))
    wn.add_pipe("pMG", "M", "G0_2", 200, 0.35, 110)
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                wn.add_pipe(f"pg{i}_{j}h", f"G{i}_{j}", f"G{i+1}_{j}", 250, 0.20, 100)
            if j + 1 < n:
                wn.add_pipe(f"pg{i}_{j}v", f"G{i}_{j}", f"G{i}_{j+1}", 250, 0.20, 100)
    wn.add_pipe("pR2", "R2", "G5_5", 300, 0.30, 110)

    # dead-end branch off G3_0
    prev = "G3_0"
    for k in range(8):
        J(f"D{k}", (4, -3.5 - k * 0.5))
        wn.add_pipe(f"pD{k}", prev, f"D{k}", 200, 0.15, 100)
        prev = f"D{k}"
    return wn


def ky4():
    return wntr.network.WaterNetworkModel(LIB + "ky4.inp")


def ltown():
    """BattLeDIM L-Town nominal model (as distributed with EPyT / EPANET-Matlab-Toolkit)."""
    return wntr.network.WaterNetworkModel(config.LTOWN_INP)


def load(name):
    return {"synthetic": synthetic, "ky4": ky4, "ltown": ltown}[name]()
