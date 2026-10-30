# Service-failure prediction: pre-specified protocol

Written 27 October 2026, before any scenario was simulated.

## Question

When a leak grows slowly at a known location, can the onset of service failure
be predicted early, from the hydraulic state at a small leak size? Two targets:

1. **Which node fails first**: the first in-service junction whose pressure
   falls below the service threshold.
2. **When it fails**: the leak size (orifice area) at which that first failure
   occurs. For a leak growing linearly in time this is proportional to time to failure.

The NEET principle tested is that failure is governed by the distribution of
depletion relative to the remaining margin, not by totals or absolute values.

## Networks and conditions

- L-Town (782 junctions) and KY4 (959 junctions), steady state, pressure-dependent demand.
- Service threshold p_min = 20 m. In-service nodes: baseline pressure >= 22 m.
- 300 scenarios per network. Leak node drawn uniformly from in-service candidate
  junctions; seed 2026 (L-Town) and 2027 (KY4).
- Ground truth: the leak area A* in (0, 200] cm^2 at which the first in-service
  node falls below p_min, found by bisection (9 steps, about 0.4 cm^2 resolution),
  and the node that fails at A*. Scenarios with no failure up to 200 cm^2 are censored
  (excluded from node accuracy; reported as a share).
- Amendment before any scenario run: the upper limit was raised from 50 to 200 cm^2
  after a 12-leak feasibility check showed that only 42-58% of leaks caused a
  failure by 50 cm^2 (all did by 200 cm^2). No predictor was computed in that check.

## Information available to the predictor

The hydraulic model, the leak location, and the state at the early stage
A0 = 2 cm^2 (simulated): pressures p_i and the observed head drops dh_i(A0) at
all nodes. No information beyond A0 is used.

## Predictors of the first failing node

- MARGIN: the in-service node with the lowest current margin m_i = p_i(A0) - p_min.
- NEAREST: the in-service node nearest the leak by pipe length.
- DROP: the node with the largest observed drop dh_i(A0) (impact only).
- NEET: the node with the largest relative depletion d_i = dh_i(A0) / m_i,
  that is, drop per unit of remaining margin.
- NEET_LIN: as NEET, with the drop predicted by the linearised influence
  G_ij / G_jj * dh_j(A0) instead of observed (model-only variant).

Metrics: top-1 accuracy, top-5 accuracy, pipe-length distance from predicted to
true first-failing node.

## Predictors of the critical leak size A*

Extrapolation assumes the drop grows in proportion to leak outflow, with outflow
following the orifice law at the leak node's pressure.

- AT_LEAK: failure assumed at the leak node itself, A* from its own margin.
- MARGIN_MIN: the lowest margin anywhere, divided by the drop rate at that node.
- NEET: A* = A0 * min_i (m_i / dh_i(A0)) corrected for the orifice law
  (the node with the largest relative depletion sets the time to failure).

Metric: median absolute relative error |A_pred - A*| / A*, and Spearman rank
correlation of A_pred with A* across scenarios.

## NEET disequilibrium test (distribution versus total)

Across scenarios, with d_i as above over in-service nodes:
- TOTAL: mean relative depletion, mean(d).
- PSI_N: normalised disequilibrium, std(d) / mean(d).
Spearman partial correlation of A* with PSI_N controlling for TOTAL. NEET
predicts a negative partial correlation (more concentrated depletion, earlier failure).

## Decision rule stated in advance

The NEET node predictor counts as useful if its top-1 accuracy exceeds every
baseline on both networks. The NEET size predictor counts as useful if its
median relative error is lower than both baselines on both networks. The
disequilibrium claim holds if the partial correlation is negative with p < 0.01
on both networks. Each result is reported whichever way it falls.
