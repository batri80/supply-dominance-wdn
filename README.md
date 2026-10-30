# Supply dominance, leak localisation and service-failure prediction

Code, pre-specified protocols and results for the paper

> **Physics-first intelligent pipeline asset management: supply dominance, leak localisation and service-failure prediction**
> Batri Krishnan, Lakshmi S, Surbhi B. Khan, Fayez Alfayez, Alhanof Almutairi, Ahlam Almusharraf.
> Submitted to *Process Safety and Environmental Protection*, special issue "Intelligent Asset Management for Process Safety and Environmental Protection".

Every table and figure in the paper can be regenerated from this repository. The results
reported in the paper are stored in `results/` together with the run logs, so they can be
checked without rerunning anything.

## Contents

| Folder or file | What it holds |
| --- | --- |
| `src/` | All experiment scripts, and the pre-specified protocols (`*_protocol.md`) |
| `run_all.py` | Runs the experiments in stages, with logs |
| `scripts/make_figures.py` | Produces Figures 2 to 5 |
| `results/` | Result files (JSON) and run logs of the runs reported in the paper |
| `data/L-TOWN.inp` | BattLeDIM L-Town nominal model (EUPL licence, KIOS Research and Innovation Center of Excellence) |
| `RUNNING.md` | Installation, data download, stages and run times |
| `EXPECTED_RESULTS.md` | Reference values for the main stages |

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py check                       # environment and data
python run_all.py                             # main stages (about 3 h on a 16-thread laptop)
python run_all.py extensions                  # X1 and X2 (about 11 h)
python run_all.py revision                    # R1 to R5 (about 7 h)
python scripts/make_figures.py
```

The BattLeDIM SCADA files and the organisers' repository are not redistributed here; see
`RUNNING.md` for where to download them. The KY4 network ships with WNTR.

## Where each result comes from

| Paper item | Stage (`run_all.py`) | Script | Result file in `results/` |
| --- | --- | --- | --- |
| Table 4 (prototype tests) | `prototype` | `experiments.py`, `e1_recompute.py` | `results_synthetic.json`, `results_ky4.json` |
| Table 5 (worked example, Prop. 2) | `rev_checks` | `rev_checks.py walk` | `rev_r5_walk.json` |
| Table 6 (concurrent-leak interaction, Prop. 4) | `rev_checks` | `rev_checks.py prop4` | `rev_r5_prop4.json` |
| Isolability limit (Prop. 3, Sec. 7.1) | `placement`, `rev_checks` | `prop3_check.py`, `rev_checks.py prop3` | `prop3_check_ky4.json`, `rev_r5_prop3.json` |
| Conductance-cap sensitivity (Sec. 7.1) | `rev_checks` | `rev_checks.py cap` | `rev_r5_cap.json` |
| Table 7 (placement), selection library | `placement` | `placement.py` | `placement_ky4.json`, `placement_ltown.json` |
| Table 7 (placement), independent library | `rev_placement` | `placement2.py` | `rev_r2_placement_ky4.json`, `rev_r2_placement_ltown.json` |
| Table 8, Figure 3 (localiser X1) | `ext_localiser` | `localiser2.py` | `ext_localiser_ltown.json`, `ext_localiser_ky4.json` |
| Table 9 (localiser R3, bootstrap) | `rev_localiser` | `localiser3.py` | `rev_r3_ltown.json`, `rev_r3_ky4.json` |
| Table 10 (BattLeDIM) and per-leak delays | `battledim`, `rev_checks` | `bd_detect.py`, `rev_checks.py battledim` | `bd_results_2019.json`, `rev_r5_battledim.json` |
| Table 11 (first service-failure experiment) | `service_failure` | `sfp_eval.py` | `sfp_results_ltown.json`, `sfp_results_ky4.json` |
| Table 12, Figure 4 (failure size X2) | `ext_failure` | `sfp2_eval.py` | `ext_failure_size.json` |
| Table 13 (NEET-REG generalisation) | `rev_failure_checks` | `sfp2_gen.py` | `rev_r4_failure_checks.json` |
| Table 14, Figure 5 (robustness R1) | `rev_robustness` | `sfp3_eval.py`, `rev_r1_ci.py` | `rev_r1_results.json`, `rev_r1_ci.json` |
| Figure 2 | | `scripts/make_figures.py` | simulated directly |

Tables 1 to 3 are descriptive (novelty boundary, decision pathway, experimental protocols).

## Protocols

The service-failure experiment, X1, X2 and R1 to R3 followed protocols written before their
test data were generated or analysed. They are in `src/sfp_protocol.md`, `src/ext_protocol.md`
and `src/rev_protocol.md`, with every amendment recorded. The early-warning experiment
(`src/ews_protocol.md`, stage `early_warning`) is included for completeness; its result was
negative and it is not reported in this paper.

## Reproducibility notes

Simulations are deterministic, and WNTR 1.4.0 (macOS) and 1.5.0 (Linux) agree to within
1e-13 m. Results that depend on ties between candidate junctions with identical signatures
(junctions in one shadow) can differ slightly between environments: the BattLeDIM score
(EUR 184,715 to 184,879 for the same 13 detections and 4 false positives) and the largest-drop
and nearest-junction baselines. Graph-network results vary with the training seed and the
PyTorch version, which is why they are reported over five seeds.

## Licence and citation

Code: MIT licence (see `LICENSE`). The L-Town model, SCADA data and scoring rules belong to the
BattLeDIM organisers (Vrachimis et al., 2022, J. Water Resour. Plan. Manag. 148(12), 04022068)
and are used under the EUPL. If you use this code, please cite the paper (see `CITATION.cff`).
