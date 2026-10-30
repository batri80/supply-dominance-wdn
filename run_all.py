"""Run the PSEP pipeline experiments in order.

  python run_all.py                       run every stage
  python run_all.py battledim early_warning   run the named stages only
  python run_all.py --list                list the stages
  python run_all.py --skip-existing ...   skip any command whose output file already exists
  python run_all.py extensions            run the extension experiments (ext_localiser, ext_failure)
  python run_all.py revision              run the revision experiments (rev_*; protocol src/rev_protocol.md)

Each command's output is shown on screen and saved to outputs/logs/.
Stages depend on earlier ones as listed under "needs".
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
import config  # noqa: E402

W = str(config.WORKERS)
# (stage, description, needs, [(script args..., output file that marks completion)])
STAGES = [
    ("check", "Check environment and data files", [], [
        (["check_setup.py"], None)]),
    ("prototype", "Decay, shadow, masking and nonlinearity tests (Tests 1-4)", [], [
        (["experiments.py", "synthetic"], "results_synthetic.json"),
        (["experiments.py", "ky4", "150"], "results_ky4.json"),
        (["e1_recompute.py", "synthetic"], None),
        (["e1_recompute.py", "ky4"], None)]),
    ("libraries", "Simulate a 1 cm2 leak at every eligible junction", [], [
        (["build_library.py", "synthetic", "1e-4", W], "lib_synthetic_0.0001.npz"),
        (["build_library.py", "ky4", "1e-4", W], "lib_ky4_0.0001.npz"),
        (["build_library.py", "ltown", "1e-4", W], "lib_ltown_0.0001.npz")]),
    ("placement", "Sensor placement comparison and isolability-limit check", ["libraries"], [
        (["placement.py", "synthetic"], "placement_synthetic.json"),
        (["placement.py", "ky4"], "placement_ky4.json"),
        (["placement.py", "ltown"], "placement_ltown.json"),
        (["prop3_check.py"], "prop3_check_ky4.json")]),
    ("scenarios", "Simulated L-Town training (3,000) and test (800) scenarios", [], [
        (["gen_train.py", "ltown", "3000", W, "0"], "train_ltown_s0.npz"),
        (["gen_train.py", "ltown", "800", W, "1"], "train_ltown_s1.npz")]),
    ("localiser", "Hybrid graph-network localiser and baselines", ["scenarios"], [
        (["localiser.py"], "localiser_ltown.json")]),
    ("battledim", "BattLeDIM benchmark on the official SCADA data", [], [
        (["scada_convert.py"], "scada_2019_levels.pkl"),
        (["battledim_score.py"], None),
        (["bd_residuals.py"], "bd_resid.pkl"),
        (["bd_explore.py"], "ltown_Ginv.npy"),
        (["bd_detect.py", "tune", "daily"], "bd_tune_2018_daily.json"),
        (["bd_detect.py", "tune", "night"], "bd_tune_2018_night.json"),
        (["bd_detect.py", "test", "daily"], "bd_results_2019.json")]),
    ("early_warning", "CST and NIG early-warning indicators (protocol: src/ews_protocol.md)", ["battledim"], [
        (["ews_features.py"], "ews_features.pkl"),
        (["ews_eval.py", "2018"], "ews_results_2018.json"),
        (["ews_eval.py", "2019"], "ews_results_2019.json")]),
    ("service_failure", "NEET service-failure prediction (protocol: src/sfp_protocol.md)", [], [
        (["sfp_sim.py", "ltown", "300", W], "sfp_ltown.npy"),
        (["sfp_sim.py", "ky4", "300", W], "sfp_ky4.npy"),
        (["sfp_eval.py", "ltown"], "sfp_results_ltown.json"),
        (["sfp_eval.py", "ky4"], "sfp_results_ky4.json")]),
    # ---- extension experiments (protocol: src/ext_protocol.md); not part of a default run
    ("ext_localiser", "X1: localiser over 5 seeds, new baselines, L-Town and KY4", ["scenarios"], [
        (["ext_sensors.py"], "ext_sensors.json"),
        (["gen_train.py", "ky4", "3000", W, "0"], "train_ky4_s0.npz"),
        (["gen_train.py", "ky4", "800", W, "1"], "train_ky4_s1.npz"),
        (["localiser2.py", "ltown", "5"], None),
        (["localiser2.py", "ky4", "5"], None)]),
    ("ext_failure", "X2: nonlinear NEET estimate of the failure size", [], [
        (["sfp2_sim.py", "ltown", "3026", "300", W], "sfp2_ltown_s3026.npy"),
        (["sfp2_sim.py", "ltown", "4026", "300", W], "sfp2_ltown_s4026.npy"),
        (["sfp2_sim.py", "ky4", "3027", "300", W], "sfp2_ky4_s3027.npy"),
        (["sfp2_sim.py", "ky4", "4027", "300", W], "sfp2_ky4_s4027.npy"),
        (["sfp2_eval.py"], "ext_failure_size.json")]),
    # ---- revision experiments (protocol: src/rev_protocol.md); run with "python run_all.py revision"
    ("rev_checks", "R5: cap sensitivity, Prop. 2 worked example, Prop. 3 contrast, Prop. 4 sign, BattLeDIM table",
     ["libraries", "battledim", "ext_localiser"], [
        (["rev_checks.py", "battledim", "prop3", "walk", "cap", "prop4"], "rev_r5_prop4.json")]),
    ("rev_failure_checks", "R4: NEET-REG generalisation and NEET-2PT checks on the X2 data", ["ext_failure"], [
        (["sfp2_gen.py"], "rev_r4_failure_checks.json")]),
    ("rev_placement", "R2: placement evaluated on an independent leak library", ["libraries"], [
        (["placement2.py", "ky4", W], "rev_r2_placement_ky4.json"),
        (["placement2.py", "ltown", W], "rev_r2_placement_ltown.json")]),
    ("rev_robustness", "R1: robustness of the two-point failure-size estimate to model mismatch", [], [
        (["sfp3_sim.py", "ltown", "100", W], "rev_r1_ltown.npy"),
        (["sfp3_sim.py", "ky4", "100", W], "rev_r1_ky4.npy"),
        (["sfp3_eval.py"], "rev_r1_results.json"),
        (["rev_r1_ci.py"], "rev_r1_ci.json")]),
    ("rev_localiser", "R3: scenario-level uncertainty for the localiser comparison", ["ext_localiser"], [
        (["localiser3.py", "ltown", "5"], None),
        (["localiser3.py", "ky4", "5"], None)]),
]
EXTENSIONS = {"ext_localiser", "ext_failure"}
REVISION = ["rev_checks", "rev_failure_checks", "rev_placement", "rev_robustness", "rev_localiser"]


def run(args, log_path):
    cmd = [sys.executable, "-u", os.path.join(SRC, args[0])] + args[1:]
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=config.OUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write("    " + line)
            log.write(line)
        return p.wait()


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for name, desc, needs, cmds in STAGES:
            print(f"{name:16s} {desc}" + (f"  (needs: {', '.join(needs)})" if needs else ""))
        return
    skip = "--skip-existing" in argv
    wanted = [a for a in argv if not a.startswith("--")]
    if "extensions" in wanted:
        wanted = [w for w in wanted if w != "extensions"] + sorted(EXTENSIONS)
    if "revision" in wanted:
        wanted = [w for w in wanted if w != "revision"] + REVISION
    wanted = wanted or [s[0] for s in STAGES if s[0] not in EXTENSIONS and s[0] not in REVISION]
    unknown = set(wanted) - {s[0] for s in STAGES}
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(sorted(unknown))}. Use --list.")
    logs = os.path.join(config.OUT, "logs")
    os.makedirs(logs, exist_ok=True)
    print(f"Outputs: {config.OUT}\nWorkers: {config.WORKERS}\n")
    t_all = time.time()
    for name, desc, needs, cmds in STAGES:
        if name not in wanted:
            continue
        print(f"== {name}: {desc}")
        for args, marker in cmds:
            label = " ".join(args)
            if skip and marker and os.path.exists(os.path.join(config.OUT, marker)):
                print(f"  - {label}: skipped ({marker} exists)")
                continue
            print(f"  - {label}")
            t = time.time()
            log = os.path.join(logs, "_".join(a.replace(".py", "") for a in args) + ".log")
            code = run(args, log)
            print(f"    ({time.time() - t:.0f} s, exit {code}, log: {log})")
            if code != 0:
                raise SystemExit(f"Stopped: '{label}' failed. See {log}.")
        print()
    print(f"All requested stages finished in {(time.time() - t_all) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
