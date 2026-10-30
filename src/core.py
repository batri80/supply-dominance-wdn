"""Core utilities for the merge/split fault-propagation prototype.

Steady-state (single time step) pressure-dependent simulations with WNTR.
A fault is an orifice leak at a junction. Influence of a leak at j on node i
is the head drop dh_i = h_i(baseline) - h_i(leak at j).
"""
import config  # noqa: F401  (first: sets warnings filter and the outputs folder)
import copy
import os
import numpy as np
import networkx as nx
import wntr

LIB = os.path.join(os.path.dirname(wntr.__file__), "library", "networks") + os.sep


def prep(wn, demand_mult=1.0):
    wn = copy.deepcopy(wn)
    wn.options.time.duration = 0
    wn.options.time.hydraulic_timestep = 3600
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.required_pressure = 20.0
    wn.options.hydraulic.minimum_pressure = 0.0
    wn.options.hydraulic.demand_multiplier = demand_mult
    # drop controls so a single snapshot is not perturbed by rule switching
    for name in list(wn.control_name_list):
        wn.remove_control(name)
    return wn


def simulate(wn, leaks=None, demand_mult=1.0):
    """leaks: dict node -> orifice area (m^2). Returns (head Series, flow Series)."""
    w = prep(wn, demand_mult)
    for node, area in (leaks or {}).items():
        j = w.get_node(node)
        j.add_leak(w, area=area, discharge_coeff=0.75, start_time=0)
    res = wntr.sim.WNTRSimulator(w).run_sim()
    return res.node["head"].iloc[0], res.link["flowrate"].iloc[0]


def flow_digraph(wn, flow):
    """Directed graph oriented by baseline flow; edge attr g = incremental conductance."""
    G = nx.DiGraph()
    for name, link in wn.links():
        q = flow[name]
        a, b = link.start_node_name, link.end_node_name
        if q < 0:
            a, b = b, a
        G.add_edge(a, b, link=name, q=abs(q))
    return G


def junction_types(wn, flow, tol=1e-6):
    """Classify junctions by baseline flow direction."""
    G = flow_digraph(wn, flow)
    types = {}
    for n in wn.junction_name_list:
        if n not in G:
            continue
        ins = [u for u, _, d in G.in_edges(n, data=True) if d["q"] > tol]
        outs = [v for _, v, d in G.out_edges(n, data=True) if d["q"] > tol]
        if len(ins) >= 2:
            types[n] = "merge"
        elif len(outs) >= 2:
            types[n] = "split"
        elif len(outs) == 0:
            types[n] = "end"
        else:
            types[n] = "pass"
    return types, G


def influence_matrix(wn, head0, flow0, nodes, gmax=0.1):
    """Linearised influence G = L_g^{-1}; see grounded_laplacian."""
    L, idx = grounded_laplacian(wn, head0, flow0, nodes, gmax)
    return np.linalg.inv(L), idx


def grounded_laplacian(wn, head0, flow0, nodes, gmax=0.1):
    """Conductance-weighted Laplacian L_g over junctions, grounded at fixed-head nodes.

    Edge conductance g = dq/dh for Hazen-Williams h = r q^1.852:
    g = 1 / (1.852 r |q|^0.852), estimated as q / (1.852 |dh|) from the baseline.
    Pipes with negligible flow get a small floor conductance.
    """
    idx = {n: k for k, n in enumerate(nodes)}
    N = len(nodes)
    L = np.zeros((N, N))
    fixed = set(wn.reservoir_name_list) | set(wn.tank_name_list)
    GMAX = gmax  # cap for near-zero-flow pipes (true dq/dh -> infinity as q -> 0); default 0.1
    for name, link in wn.links():
        a, b = link.start_node_name, link.end_node_name
        q = abs(flow0[name])
        dh = abs(head0[a] - head0[b])
        if link.link_type == "Pipe":
            g = GMAX if dh < 1e-9 else min(q / (1.852 * dh), GMAX)
        elif link.link_type == "Pump":
            if q < 1e-7:
                g = 0.0                      # closed pump carries no perturbation
            elif link.pump_type == "POWER":  # H = P/(rho g q)  =>  dq/dH = q/H
                g = q / max(dh, 1e-6)
            else:                            # head curve H = A - B q^C
                A, B, C = link.get_head_curve_coefficients()
                g = 1.0 / max(B * C * q ** (C - 1), 1e-9)
        else:  # valves: treat as stiff connections
            g = GMAX
        for u, v in ((a, b), (b, a)):
            if u in idx:
                L[idx[u], idx[u]] += g
                if v in idx:
                    L[idx[u], idx[v]] -= g
    # nodes adjacent to fixed heads are grounded through those edges already
    return L, idx
