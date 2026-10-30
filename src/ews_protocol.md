# Early-warning experiment: pre-specified protocol

Written 27 October 2026, before any 2019 early-warning indicator was computed.
2018 is used for all choices; 2019 is evaluated once with the choices frozen.

## Question

Do stability indicators derived from the author's earlier frameworks (Cognitive
Stability Theory energy and oscillation; Neuromotor Interaction Graph signatures)
rise during the early phase of a leak, before or more reliably than (a) the reactive
drop statistic of the detection pipeline and (b) generic early-warning signals
(variance, lag-1 autocorrelation)?

## Data

BattLeDIM SCADA, 33 pressure sensors at 5-minute resolution, 2018 and 2019.
Residuals: observed pressure minus the regression on inflows, tank level and
time of day fitted on the leak-free first week of 2018 (bd_residuals.py).

## Daily indicators (all causal: day d uses data up to the end of day d only)

Reactive baseline
- REACT: norm of max(baseline - recent, 0), baseline = mean daily residual over
  [d-60, d-7), recent = mean over [d-7, d]. This is the existing detector's statistic.

Generic early-warning signals (critical slowing down)
- VAR: mean over sensors of the within-day variance of 5-minute residuals.
- AC1: mean over sensors of the within-day lag-1 autocorrelation of residuals.

CST-derived (energy = parabolic deviation from the balance point)
- ENERGY: within-day mean of E(t) = sum_i r_i(t)^2 / s_i^2, with s_i the
  residual standard deviation of sensor i in the reference week.
- DESCENT: 7-day least-squares slope of daily ENERGY (a sustained rise is a
  failure of energy descent).
- OSC: within-day fraction of sign reversals of the 5-minute energy increment
  dE(t), the CST oscillation index.

NIG-derived (functional interaction graph, estimated per day from that day only)
- Directed Gaussian transfer entropy at lag 1 between every ordered sensor pair,
  from the within-day residual series (Gaussian TE = Granger form).
- TE_MEAN: mean TE over pairs.
- EFF: global efficiency of the TE graph (edge length 1/TE).
- LAMBDA2: algebraic connectivity of the symmetrised TE graph.
- SPEC: spectral energy (sum of absolute eigenvalues of the symmetrised adjacency).

Each indicator is converted to a causal z-score against its own trailing 60-day
window [d-60, d-1]. Direction: REACT, VAR, AC1, ENERGY, DESCENT are scored by z
(increase = warning); OSC, TE_MEAN, EFF, LAMBDA2, SPEC by |z| (direction not
assumed in advance).

## Labels (per day)

- Positive (early phase): a leak onset (any type) in (d-14, d].
- Negative: no leak onset in (d-30, d] and no leak repair in (d-7, d].
- All other days are excluded. The series run continuously from 2018 into 2019,
  so only the first 60 days of 2018 are excluded (z-score warm-up); trailing
  windows at the start of 2019 use late-2018 data, as a real monitor would.
  The four leaks running from 2018 into 2019 have no 2019 onset.

## Metrics

1. AUROC of each indicator separating positive from negative days, per year.
   Uncertainty: 1,000 bootstrap resamples over leak events and negative weeks.
2. Combined model: logistic regression on all indicators, fitted on 2018 labels,
   applied unchanged to 2019 (CST+NIG only, and all indicators).
3. Event lead time: alarm threshold per indicator set on 2018 to give at most
   one false alarm per 30 negative days; for each 2019 onset, days from onset
   to the first alarm, compared with the same quantity for REACT.

## Decision rule stated in advance

The CST/NIG indicators are counted as useful only if, on 2019, their AUROC or
their combined model exceeds both REACT and the better of VAR/AC1, with
bootstrap intervals that do not overlap. Otherwise the result is reported as negative.
