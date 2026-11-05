# Supply dominance, leak identifiability and service-failure prediction

Code, pre-specified protocols, results and Supplemental Materials for the paper

> **Supply Dominance, Leak Identifiability and Service-Failure Prediction in Water Distribution Networks**
> Batri Krishnan and Lakshmi S. Submitted to the *Journal of Water Resources Planning and Management* (ASCE).

Every table and figure in the paper can be regenerated from this repository. The results
reported in the paper are stored in `results/` together with the run logs, so they can be
checked without rerunning anything. The Supplemental Materials (Tables S1–S7, Fig. S1) are in
`supplement/Supplemental_Materials.pdf`.

## Contents

| Folder or file | What it holds |
| --- | --- |
| `src/` | All experiment scripts, and the pre-specified protocols (`*_protocol.md`) |
| `run_all.py` | Runs the experiments in stages, with logs |
| `scripts/make_figures.py` | Produces the figures (Figures 1–3 and Fig. S1 of the paper) |
| `supplement/` | Supplemental Materials (Tables S1–S7, Fig. S1) |
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
| Table 1 (prototype tests), Fig. 1 | `prototype` | `experiments.py`, `e1_recompute.py`; `scripts/make_figures.py` | `results_synthetic.json`, `results_ky4.json` |
| Isolability limit (Prop. 3) | `placement`, `rev_checks` | `prop3_check.py`, `rev_checks.py prop3` | `prop3_check_ky4.json`, `rev_r5_prop3.json` |
| Conductance-cap sensitivity | `rev_checks` | `rev_checks.py cap` | `rev_r5_cap.json` |
| Table 2 (placement, independent library) | `rev_placement` | `placement2.py` | `rev_r2_placement_ky4.json`, `rev_r2_placement_ltown.json` |
| Table 3, Fig. 2 (localisers, X1 and R3) | `ext_localiser`, `rev_localiser` | `localiser2.py`, `localiser3.py` | `ext_localiser_*.json`, `rev_r3_*.json` |
| Table 4 (BattLeDIM) and detection delays | `battledim`, `rev_checks` | `bd_detect.py`, `rev_checks.py battledim` | `bd_results_2019.json`, `rev_r5_battledim.json` |
| Table 5, Fig. 3 (failure size X2, robustness R1) | `ext_failure`, `rev_robustness` | `sfp2_eval.py`, `sfp3_eval.py`, `rev_r1_ci.py` | `ext_failure_size.json`, `rev_r1_results.json`, `rev_r1_ci.json` |
| Table S3 (worked example, Prop. 2) | `rev_checks` | `rev_checks.py walk` | `rev_r5_walk.json` |
| Table S4 (concurrent-leak interaction, Prop. 4) | `rev_checks` | `rev_checks.py prop4` | `rev_r5_prop4.json` |
| Table S5 (per-seed localiser results) | `rev_localiser` | `localiser3.py` | `rev_r3_*.json` |
| Table S6 (first service-failure experiment) | `service_failure` | `sfp_eval.py` | `sfp_results_*.json` |
| Table S7 (regression generalisation) | `rev_failure_checks` | `sfp2_gen.py` | `rev_r4_failure_checks.json` |
| Placement on the selection library (Sec. 6) | `placement` | `placement.py` | `placement_*.json` |

Tables S1 (novelty boundary) and S2 (protocols) are descriptive.

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
