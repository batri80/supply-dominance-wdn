# Running the experiments

Code for the experiments behind the planned Process Safety and Environmental Protection
paper (VSI: Intelligent Asset) on supply structure, leak localisation and service-failure
prediction in pipeline networks. Every result in the framework document can be
regenerated with this folder.

## 1. Install (once)

The code was tested in a clean Python 3.11 environment with the versions in
`requirements.txt`. These are the newest versions with Intel-Mac builds: WNTR 1.4.0 and
PyTorch 2.2.2. Their hydraulic results are identical to WNTR 1.5.0 to within 1e-13 m.

```bash
cd psep_pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3.11` is not available, install Python 3.11 from python.org or with
`brew install python@3.11`. Apple Silicon Macs, Linux and Windows may instead use
`wntr==1.5.0` and a current `torch`.

When PyTorch 2.2.2 is imported alongside NumPy 2, it prints a warning that a module was
"compiled using NumPy 1.x". The code avoids the affected NumPy–PyTorch conversion, so the
warning is harmless, and the scripts suppress it where they can.

## 2. Put the data in place

```
psep_pipeline/
  data/
    L-TOWN.inp              included (BattLeDIM nominal model)
    scada/
      2018_SCADA.xlsx       the two files you downloaded from Zenodo record 4017659
      2019_SCADA.xlsx
    BattLeDIM/              clone of the official repository (step below)
```

```bash
mkdir -p data/scada
cp /path/to/2018_SCADA.xlsx /path/to/2019_SCADA.xlsx data/scada/
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/KIOS-Research/BattLeDIM.git data/BattLeDIM
```

From the repository, only the leak list, the leak flow files and the reference
submissions in `Scoring Algorithm/` are used. These are ordinary files, so Git LFS is not
needed. The `GIT_LFS_SKIP_SMUDGE=1` prefix stops git from failing if LFS is installed.
The KY4 network comes with WNTR itself.

Then check everything:

```bash
python run_all.py check
```

It should end with `Setup OK.`

## 3. Run

```bash
python run_all.py --list                 # show the stages
python run_all.py                        # run all main stages in order
python run_all.py battledim early_warning   # run selected stages
python run_all.py --skip-existing        # rerun, skipping finished steps
```

Results, intermediate files and one log per command are written to `outputs/`
(logs in `outputs/logs/`). Each stage is listed below with the times measured on a
2-core machine. On an 8-core MacBook Pro the simulation stages should run roughly
three to five times faster.

| Stage | What it tests | Needs | Time (2 cores) |
| --- | --- | --- | --- |
| check | Environment and data files | | < 1 min |
| prototype | Decay with hydraulic distance, shadow invariance, merge masking, nonlinearity | | about 1 h |
| libraries | Simulates a 1 cm² leak at every junction of three networks | | about 1 h |
| placement | Sensor placement comparison and isolability-limit check | libraries | about 20 min |
| scenarios | 3,800 simulated L-Town leak scenarios | | about 35 min |
| localiser | Hybrid graph-network localiser and baselines | scenarios | about 15 min |
| battledim | BattLeDIM benchmark on the official SCADA data | | about 30 min |
| early_warning | CST and NIG early-warning indicators | battledim | about 2 min |
| service_failure | NEET service-failure prediction | | about 4 h |

The stages without a dependency can run in any order, or on different days.
`sfp_sim.py` saves a checkpoint every 25 scenarios and resumes if restarted.

To run a single script by hand, start it from anywhere, for example
`python src/placement.py ky4`. Every script reads and writes the `outputs/` folder.

## 4. Extension experiments

Two further experiments are included but are not part of a default run. Their
protocol, including the decision rules, is in `src/ext_protocol.md` and was written
before either was run.

- **X1, `ext_localiser`.** Repeats the localiser comparison over five training seeds,
  fixes the data-only graph network, and adds a reconstructor baseline (after
  Garðarsson et al., 2022) and a physics-loss baseline. It runs on L-Town and on KY4.
  It needs the `scenarios` stage.
- **X2, `ext_failure`.** Tests two nonlinear corrections to the NEET estimate of the
  leak size at first service failure, using new calibration and test scenarios.

Run them after the main run has finished:

```bash
python run_all.py extensions                 # both experiments
python run_all.py ext_localiser              # or one at a time
python run_all.py ext_failure
```

| Stage | Main outputs | Time (2 cores) | Expected on an 8-core Mac |
| --- | --- | --- | --- |
| ext_localiser | `ext_localiser_ltown.json`, `ext_localiser_ky4.json` | about 4 h | 1 to 1.5 h |
| ext_failure | `ext_failure_size.json` | about 7.5 h | 2 to 3 h |

Both resume where they stopped if interrupted. Rerun the same command, or add
`--skip-existing` to also skip finished simulations. The final lines of each log state
the result and the pre-specified decision. Please send these files back for
analysis:

```
outputs/ext_localiser_ltown.json   outputs/ext_localiser_ky4.json   outputs/ext_failure_size.json
outputs/logs/localiser2_ltown_5.log   outputs/logs/localiser2_ky4_5.log   outputs/logs/sfp2_eval.log
```

## 4b. Revision experiments

These answer the pre-submission review. Their protocol and decision rules are in
`src/rev_protocol.md`, written before any of them was run. They use files from the
main run and the extensions, so run them in the same folder:

```bash
caffeinate -i python run_all.py revision
```

| Stage | What it does | Time on your Mac (estimate) |
| --- | --- | --- |
| rev_checks | Conductance-cap sensitivity, Proposition 2 worked example, Proposition 3 contrast, signed Proposition 4 interaction, BattLeDIM detection table | about 10 min |
| rev_failure_checks | NEET-REG coefficients, learning curve and transfer; NEET-2PT clipping, near-ties, censoring (X2 data, no new simulation) | under 2 min |
| rev_placement | Placement layouts evaluated on a separate 5 cm² library at 1.2 x demand | about 20 min |
| rev_robustness | NEET-2PT under perturbed roughness, demand, pressure-dependent-demand settings, discharge coefficient, demand drift, noise and a wrong leak junction | about 2.5 h |
| rev_localiser | Retrains the two hybrids over five seeds and adds scenario-level and hierarchical bootstrap intervals | about 3.5 h |

In total about 6.5 h, so an overnight run suits. Every stage resumes if interrupted.
Please send back:

```
outputs/rev_r1_results.json   outputs/rev_r2_placement_ky4.json   outputs/rev_r2_placement_ltown.json
outputs/rev_r3_ltown.json     outputs/rev_r3_ky4.json             outputs/rev_r4_failure_checks.json
outputs/rev_r1_ci.json
outputs/rev_r5_cap.json  outputs/rev_r5_walk.json  outputs/rev_r5_prop3.json  outputs/rev_r5_prop4.json  outputs/rev_r5_battledim.json
outputs/logs/   (the whole folder, or at least the logs whose names start with rev_, sfp3_, placement2_, localiser3_, sfp2_gen)
```

## 5. Settings

| Variable | Meaning | Default |
| --- | --- | --- |
| `PSEP_WORKERS` | Parallel simulation processes | CPU count − 1 |
| `PSEP_DATA` | Data folder | `data/` |
| `PSEP_OUT` | Outputs folder | `outputs/` |

For example: `PSEP_WORKERS=6 python run_all.py service_failure`

## 6. Checking your results

Compare the result files with `EXPECTED_RESULTS.md`. Simulation stages are
deterministic and should match exactly. The graph-network results can differ by about
two percentage points between PyTorch versions.

Two experiments follow protocols written before their test data were analysed:
`src/ews_protocol.md` (early warning) and `src/sfp_protocol.md` (service failure).
Their thresholds and models are fixed on 2018, or on the stated rules. Changing them
after looking at 2019 results would break the protocol, so record any change as a
deviation.

## 7. Files

| File | Purpose |
| --- | --- |
| `src/config.py` | Paths and settings |
| `src/core.py`, `src/networks.py` | WNTR simulation, linearised influence, network loaders |
| `src/experiments.py`, `src/e1_recompute.py` | Prototype tests 1 to 4 |
| `src/build_library.py`, `src/placement.py`, `src/prop3_check.py` | Leak libraries, placement, isolability limit |
| `src/gen_train.py`, `src/localiser.py` | Scenario generation, hybrid localiser |
| `src/scada_convert.py`, `src/battledim_score.py`, `src/bd_residuals.py`, `src/bd_explore.py`, `src/bd_detect.py`, `src/bd_selected.json` | BattLeDIM pipeline and the Python port of the official scoring |
| `src/ews_features.py`, `src/ews_eval.py`, `src/ews_protocol.md` | Early-warning experiment |
| `src/sfp_sim.py`, `src/sfp_eval.py`, `src/sfp_protocol.md` | Service-failure prediction |
| `src/ext_protocol.md`, `src/ext_sensors.py`, `src/localiser2.py` | Extension X1: localiser robustness, baselines, KY4 |
| `src/sfp2_sim.py`, `src/sfp2_eval.py` | Extension X2: nonlinear failure-size estimate |
| `src/rev_protocol.md`, `src/sfp3_sim.py`, `src/sfp3_eval.py` | Revision R1: robustness of the failure-size estimate |
| `src/placement2.py` | Revision R2: placement on an independent library |
| `src/localiser3.py` | Revision R3: scenario-level uncertainty for the localiser comparison |
| `src/sfp2_gen.py` | Revision R4: NEET-REG generalisation and NEET-2PT checks |
| `src/rev_r1_ci.py` | Bootstrap intervals for R1 and X2 median errors (second review) |
| `src/rev_checks.py` | Revision R5: cap sensitivity, worked example, Propositions 3 and 4 checks, BattLeDIM table |

## Credits

The L-Town network, the SCADA data and the scoring rules are from the Battle of the
Leakage Detection and Isolation Methods (Vrachimis et al., 2022, J. Water Resour. Plann.
Manage. 148(12), 04022068), released by the KIOS Research and Innovation Center of
Excellence under the EUPL. `battledim_score.py` is a Python port of their MATLAB scoring
code and is checked against their reference reports on every run. Simulations use WNTR
(Klise et al., 2017).
