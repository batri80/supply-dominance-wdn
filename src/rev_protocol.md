# Revision experiments: pre-specified protocol

Written 29 October 2026, in response to a pre-submission review, before any of the
experiments below was run. Every experiment is run once with the settings given here.
Any later change is reported as a deviation together with the original result.

## R1. Robustness of the two-point failure-size estimate (NEET-2PT)

Question: does NEET-2PT stay accurate when the network that fails differs from the
model used to predict, when the observations are noisy, and when the leak location is
slightly wrong?

Scenarios: 100 per network (L-Town, KY4). Leak junctions are drawn uniformly from
in-service junctions of the nominal model with seeds 6026 (L-Town) and 6027 (KY4).
The same leak junctions are used in every condition, so conditions are paired. Service
threshold 20 m, in service if the baseline pressure is at least 22 m, A0 = 2 cm^2,
A1 = 4 cm^2, A* found by bisection on (0, 200] cm^2 with 9 steps, censored if no
failure by 200 cm^2. In-service sets and A* are always those of the "true" network.

Conditions (the "true" network that fails; the nominal model is unchanged):
- nominal: true network = nominal model.
- roughness: every pipe roughness coefficient multiplied by exp(N(0, 0.15^2)).
- demand: all demands multiplied by U(0.8, 1.2), and each junction's demand further by
  exp(N(0, 0.2^2)).
- pdd: required pressure U(15, 30) m and minimum pressure U(0, 5) m (nominal 20 and 0).
- cd: leak discharge coefficient U(0.6, 0.9) (nominal 0.75).
- combined: roughness, demand, pdd and cd together.
- drift: observations at nominal demand; failure (A*) at demand multiplier U(1.1, 1.3).
Perturbations are drawn per scenario from seed 7000 + scenario index.

Predictors, all NEET-2PT (exponent clipped to [0.5, 3]):
- MODEL: drops at A0 and A1 simulated on the nominal model at the leak junction
  (the deployment route: a model run at the located leak).
- OBS: drops at A0 and A1 observed on the true network at all junctions (the
  measurement route, an upper bound on what sensors can provide).
Additional analyses on the nominal condition:
- noise: OBS with independent N(0, s^2) noise added to every pressure at baseline, A0
  and A1, s in {0.01, 0.05, 0.10} m, 20 draws per scenario.
- location error: MODEL computed at a wrong junction, drawn uniformly from in-service
  junctions 1 to 3 hops from the true leak.

Metrics: median relative error, share within 50%, Spearman with A*, median log ratio
(bias), top-1 of the first failing junction, and top-1 counting near-ties (true node
among junctions whose estimated area is within 5% of the minimum).

Decision rule: NEET-2PT (MODEL) is called robust to a condition if its median relative
error stays below 0.5 on both networks. Each condition is judged separately and every
result is reported.

## R2. Sensor placement on an independent evaluation library

Question: does the ranking of placement rules hold when layouts are evaluated on leaks
not used to select them?

Evaluation library: a 5 cm^2 leak at every eligible junction at demand multiplier 1.2,
simulated separately from the 1 cm^2 nominal selection library. Layouts are selected
exactly as in placement.py (same budgets). Localisation: the angle method with the
selection library as dictionary and observations from the evaluation library with
noise s in {0.01, 0.05} m (10 draws). A second dictionary, the linearised influence,
is also reported. Random placement: mean and standard deviation over 10 layouts.
Metrics: mean hop error, top-1, top-5, and the share of scenarios with a tied top score.

Decision rule: the earlier conclusion (even spreading better than the shadow-rule
variant and PageRank) is confirmed if it holds at every budget on both networks with
the selection-library dictionary at s = 0.01 m.

## R3. Scenario-level uncertainty for the localiser comparison

Question: is the hybrid's difference from the physics prior robust to the choice of
test scenarios as well as training seeds?

The hybrid and hybrid_nodom models of X1 are retrained (same data, seeds 0-4, same
settings) and their per-scenario results are saved together with those of the
deterministic methods. Intervals: (a) t-interval over seeds, as in X1; (b) paired
scenario bootstrap within each seed; (c) hierarchical bootstrap resampling seeds and
test scenarios together (2,000 resamples), with the same scenarios used for every
method in a resample. Per-seed results and McNemar tests (hybrid vs prior, per seed)
are reported. A concurrent-leak scenario counts as a success if the single estimate is
within 300 m of either leak.

Decision rule: unchanged from X1, applied to the hierarchical interval (c). A
difference whose interval includes zero is reported as "no reliable difference", not
as equivalence.

## R4. Generalisation of NEET-REG and checks of NEET-2PT (existing X2 data)

Reported from the X2 calibration and test sets, without new simulation: regression
coefficients with standard errors; a learning curve (calibration sizes 10, 20, 40, 80,
160 and all, 200 random subsets each); transfer (fit on one network, test on the
other) and a pooled fit; the share of junction fits at the exponent bounds and of
scenarios whose estimate comes from a clipped fit; NEET-2PT error with bounds
[0.5, 3], [0.25, 5] and none; NEET-2PT error when junctions with a drop below 1 mm at
A0 are excluded; first-failing-junction accuracy with near-ties; errors by tercile of
A*; and the share of censored scenarios for which NEET-2PT predicts failure below
200 cm^2.

## R5. Checks (deterministic, no decision rule)

- Conductance cap: physics-prior localisation (X1 test sets) and agreement of the
  linearised influence with the simulated leak libraries for caps 0.01, 0.1 (used),
  1 and 10.
- Worked example of Proposition 2 on the synthetic network: G_ij/G_jj, Monte Carlo
  hitting probabilities (20,000 walks) and simulated drop ratios at 0.1, 1 and 10 cm^2.
- Proposition 3 contrast on KY4: leaks outside the largest shadow with all sensors
  inside it, and with half of them moved outside.
- Proposition 4 sign: for 30 pairs below a common split and 30 random pairs per network
  at 10 cm^2, the signed nodal interaction (combined drop minus the sum of single drops)
  and the second-order cross term n(n-1) r |q0|^(n-2) fa fb on every pipe, split into
  supply links (both leaks add flow along the base direction), return links (both remove
  it, e.g. less inflow to a lower reservoir or a filling tank) and opposed links. Run
  with fixed leak outflows under demand-driven hydraulics (the setting of Proposition 4)
  and with orifice leaks under pressure-dependent demand.
  Note added while testing the code, before the run: a test on this machine showed a
  negative interaction on the synthetic network, so the check was extended from link
  signs to this decomposition to explain the sign.
- BattLeDIM: per-leak detection table with delays, and false-alarm times.
