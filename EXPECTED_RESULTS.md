# Expected results

These are the numbers from the reference run (27 October 2026; Linux, Python 3.13,
WNTR 1.5.0, PyTorch 2.14). Use them to check your run.

With WNTR 1.4.0 (the newest version for Intel Macs) the hydraulics agree with 1.5.0 to
within simulation precision. Deterministic stages should then match to the decimals
shown. The graph-network results (stage `localiser`) can differ by about two
percentage points between PyTorch versions and machines.

## prototype (outputs/results_synthetic.json, results_ky4.json)

| Quantity | Synthetic | KY4 |
| --- | --- | --- |
| Spearman, influence vs linearised influence (median) | 0.989 | 0.992 |
| Spearman, influence vs hop distance (median) | -0.587 | -0.855 |
| Profile R² of linear model (median) | 0.998 | 0.974 |
| Leaks with profile R² < 0.9 | 0.00 | 0.12 |
| Shadow-node relative influence (median) | 1.000 | 1.000 |
| Same hop distance, outside shadow (median) | 0.416 | 0.841 |
| Superposition error at 10 cm², split siblings | 0.152 | 0.149 |
| Superposition error at 10 cm², random pairs | 0.118 | 0.008 |
| Sibling confusion, merge / split | 0.044 / 0.059 | 0.168 / 0.243 |

## placement (outputs/placement_*.json), mean localisation error in hops

| Network, budget | Random | PageRank | Greedy on library | Even spreading (fps_all) | With shadow rule (dom_struct) | Utility layout |
| --- | --- | --- | --- | --- | --- | --- |
| KY4, 10% | 2.52 | 2.71 | 2.12 | 1.85 | 1.96 | |
| KY4, 20% | 1.66 | 1.96 | 1.73 | 1.37 | 1.65 | |
| L-Town, 33 sensors | 4.65 | 6.08 | 3.23 | 3.41 | 4.08 | 3.47 |
| L-Town, 10% | 2.52 | 2.96 | 1.99 | 1.91 | 2.99 | |

prop3_check_ky4.json: for all ten shadows, outside_max is at most about 3e-8.

## localiser (outputs/localiser_ltown.json), share of estimates within 300 m

| Method | All | Median error (m) |
| --- | --- | --- |
| nearest_sensor | 0.701 | 197 |
| graph_interp | 0.701 | 197 |
| linear | 0.751 | 165 |
| gnn_plain | 0.662 | 218 |
| hybrid_nodom | 0.786 | 165 |
| hybrid | 0.810 | 158 |

In the Python 3.11 test environment (PyTorch 2.2.2, the Intel-Mac version) the four
non-learned rows matched (linear 0.750), while the learned rows came out as gnn_plain
0.662, hybrid 0.789 and hybrid_nodom 0.794. The hybrid's gain over the linear prior
reproduces. The ordering of hybrid and hybrid_nodom does not, so treat differences of
about 2 points between learned models as noise unless several training seeds agree.

## battledim

battledim_score.py prints "Scoring port validated." (523,153.82 and 9,547.81).

outputs/bd_results_2019.json:

| Configuration | Score (EUR) | TP | FP |
| --- | --- | --- | --- |
| same_argmax | 184,714.59 | 13 | 4 |
| primary_medoid | 173,276.77 | 13 | 5 |
| strict_argmax | 148,403.16 | 11 | 3 |
| strict_medoid | 142,400.19 | 11 | 3 |

The two medoid variants depend on how near-tied candidates are ordered. In the Python 3.11
test environment they scored 173,194.63 and 142,318.05, about EUR 80 lower. The other two
variants match to the cent.

## early_warning (outputs/ews_results_2019.json), AUROC on 2019

REACT 0.802, VAR 0.669, AC1 0.502, ENERGY 0.108, DESCENT 0.352, OSC 0.538,
TE_MEAN 0.373, EFF 0.389, LAMBDA2 0.551, SPEC 0.500, model_CST_NIG 0.755, model_ALL 0.860.

## service_failure (outputs/sfp_results_*.json)

| Quantity | L-Town | KY4 |
| --- | --- | --- |
| Censored scenarios (of 300) | 2 | 6 |
| Partial Spearman, A* vs disequilibrium given total | -0.734 | -0.828 |
| Spearman, A* vs total depletion | -0.100 | -0.406 |
| Critical size, median relative error: NEET / AT_LEAK | 1.669 / 3.686 | 1.055 / 1.740 |
| First failing node, top-1: NEET / DROP / NEAREST / MARGIN | 0.520 / 0.685 / 0.664 / 0.034 | 0.707 / 0.364 / 0.378 / 0.058 |

## Extension experiments

These have no reference values: they are new experiments, and their outcome is decided
by the rules in `src/ext_protocol.md`. The deterministic rows printed by
`localiser2.py ltown` (nearest_sensor 0.701, graph_interp 0.701, linear 0.751) match the
`localiser` stage above, which confirms the test data are the same.
