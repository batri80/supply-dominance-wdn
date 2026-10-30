# Extension experiments: pre-specified protocol

Written 28 October 2026, before any extension experiment was run. The two experiments
below are run once, with the settings given here. Any later change is reported as a
deviation together with the original result.

## Experiment X1: localiser robustness, baselines and a second network

### Question

1. Is the hybrid localiser's gain over the linear physics prior real across training
   seeds?
2. Do the dominance flags add anything?
3. How does the hybrid compare with a data-only graph network, a reconstructor-style
   baseline and a physics-loss baseline?
4. Do the results hold on a second network?

### Networks and sensors

- L-Town: the utility's 33 pressure sensors.
- KY4: 48 sensors (5% of 959 junctions) placed by farthest-point spreading in hop
  distance, the best structural placement found earlier. Selected by `ext_sensors.py`
  from the network alone, before any KY4 scenario is generated.

### Data

Simulated scenarios as before, for each network: 3,000 training scenarios (seed 0)
and 800 test scenarios (seed 1). 30% of scenarios have two concurrent leaks, areas
are 1-10 cm^2, and demand is 0.6-1.4 times nominal. Sensor noise is 0.01 m. The test
scenarios are never used for training or model selection.

### Methods

All learned models share one architecture and training budget: a graph network with
4 conductance-weighted message-passing layers, hidden size 64, Adam (learning rate
2e-3), 30 epochs, batch 32.

Every learned model's node features begin with the harmonic interpolation of the
sensor drops over the pipe graph, scaled by its largest absolute value, and the sensor
mask. This replaces the earlier input of drops at sensor nodes only, which left most
nodes without information and stopped the plain network from training. All learned
models are therefore retrained in this experiment.

Non-learned methods:
- nearest_sensor: the sensor with the largest drop.
- graph_interp: the node with the largest interpolated drop.
- linear: the angle method against the linearised influence (physics prior).

Learned methods:
- gnn_plain: interpolated drops and sensor mask only (data-only graph network).
- reconstructor: the same network trained to reconstruct the full head-drop field
  (mean squared error against the simulated field); it localises at the node with the
  largest reconstructed drop. This adapts the reconstructor of Garðarsson et al.
  (2022) to steady-state scenarios. Their predictor network, which estimates the
  leak-free state, is not needed here because the leak-free state at the scenario's
  demand is known.
- physics_loss: the reconstructor, with an added loss requiring the implied nodal
  outflow s = L_g f to be sparse (L1 norm, weight 0.1 after scaling). Here L_g is the
  grounded conductance Laplacian and f the reconstructed field. It localises at the
  node with the largest reconstructed drop, as the reconstructor does, so the physics
  term acts as a regulariser. The same trained model is also scored by the node with
  the largest implied outflow s, reported as physics_loss_outflow.
- Amendment (28 October 2026, before the experiment was run): the protocol first
  specified localising by the largest implied outflow only. A one-epoch smoke test on
  L-Town (its only use) placed 19% of estimates within 300 m with that rule, against 65%
  for the reconstructor. Applying the Laplacian to an imperfect field amplifies its
  small-scale errors. Both scorings of the one model are now reported, and the field
  scoring is the primary physics-loss baseline.
- hybrid_nodom: interpolated drops, sensor mask and linear-prior scores.
- hybrid: as hybrid_nodom plus the two dominance flags (in a shadow; shadow entry).

### Seeds and metrics

Each learned model is trained with five seeds (0-4) on each network. Metrics on the
test scenarios: the share of estimates within 300 m of a true leak (primary), top-1,
top-5 and median distance. Each is reported for all scenarios and for single and
concurrent leaks. For each network the following are reported:
- the mean and the 95% t-interval over seeds of every metric for each learned method
  (non-learned methods are deterministic);
- the paired per-seed difference (hybrid - linear) and (hybrid - hybrid_nodom) in the
  primary metric, with its 95% t-interval.

### Decision rules

- The hybrid improves on the physics prior if the interval of (hybrid - linear) lies
  above zero on both networks.
- The dominance flags help if the interval of (hybrid - hybrid_nodom) lies above zero
  on both networks. Otherwise the paper states that no benefit was found.
- The hybrid is the strongest learned method if its mean primary metric exceeds that
  of gnn_plain, reconstructor and physics_loss (field scoring) on both networks.

## Experiment X2: nonlinear correction of the NEET failure-size estimate

### Question

Can the leak size at first service failure, A*, be estimated with a median relative
error below 0.5 from early-stage observations, using the NEET relative-depletion
principle with a nonlinear correction?

### Data

The same settings as `sfp_protocol.md` (20 m service threshold, in-service if the
baseline is at least 22 m, leak area up to 200 cm^2, bisection with 9 steps). The
scenarios are new, with different seeds from the original experiment:
- calibration set, 300 scenarios per network: seeds 3026 (L-Town) and 3027 (KY4);
- test set, 300 scenarios per network: seeds 4026 (L-Town) and 4027 (KY4).

Each scenario also records the state at a second early leak size, A1 = 4 cm^2.

### Predictors of A*, all using only the states at A0 = 2 cm^2 and A1 = 4 cm^2

- NEET_LIN: the original linear NEET estimate from A0 (baseline from the earlier
  protocol).
- NEET_2PT: for each in-service node, fit dh_i(A) = c_i A^b_i through the drops at A0
  and A1 (b_i limited to [0.5, 3]). Solve for the area at which the drop equals the
  node's baseline margin. The estimate is the minimum over in-service nodes, the node
  that runs out of margin first.
- NEET_REG: a linear regression of log A* on log NEET_2PT, log NEET_LIN, the
  disequilibrium Psi_N and the mean relative depletion, all at A0. It is fitted
  separately for each network on that network's calibration set only, and applied
  unchanged to the test set.

### Metrics and decision rule

On the test set: the median absolute relative error and the Spearman correlation
with A*, per network. A predictor solves the problem if its median relative error is
below 0.5 on both networks and lower than that of NEET_LIN on both. NEET_2PT and
NEET_REG are judged separately.
