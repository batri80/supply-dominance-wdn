"""Check the Python environment and the data files before a run."""
import contextlib
import importlib
import io
import os
import sys
import config

ok = True
print(f"Python {sys.version.split()[0]}")
for mod in ("numpy", "scipy", "pandas", "networkx", "openpyxl", "sklearn", "wntr", "torch"):
    try:
        with contextlib.redirect_stderr(io.StringIO()):     # torch 2.2 prints a harmless NumPy 2 notice
            m = importlib.import_module(mod)
        print(f"  {mod:10s} {getattr(m, '__version__', '?')}")
    except Exception as e:                                   # noqa: BLE001
        ok = False
        print(f"  {mod:10s} MISSING ({e.__class__.__name__}: {e})")

print(f"\nData folder:    {config.DATA}")
print(f"Outputs folder: {config.OUT}")
print(f"Workers:        {config.WORKERS}")
need = {
    "L-Town network model": config.LTOWN_INP,
    "SCADA 2018": os.path.join(config.SCADA_DIR, "2018_SCADA.xlsx"),
    "SCADA 2019": os.path.join(config.SCADA_DIR, "2019_SCADA.xlsx"),
    "BattLeDIM leak list": os.path.join(config.BATTLEDIM_SCORING, "competition_data", "leakages_info.yalm"),
    "BattLeDIM leak flows": os.path.join(config.BATTLEDIM_SCORING, "competition_leakages", "Leak_p257.xlsx"),
    "BattLeDIM reference submission": os.path.join(config.BATTLEDIM_SCORING, "SUBMITTED_files", "PERFECT_TEAM.txt"),
}
print()
for label, path in need.items():
    exists = os.path.isfile(path)
    small = exists and os.path.getsize(path) < 1000 and path.endswith((".xlsx", ".inp"))
    status = "OK" if exists and not small else ("GIT-LFS POINTER, NOT THE REAL FILE" if small else "MISSING")
    ok &= status == "OK"
    print(f"  {label:32s} {status:8s} {path}")

try:
    from networks import ltown, ky4
    print(f"\n  L-Town loads: {len(ltown().junction_name_list)} junctions")
    print(f"  KY4 loads:    {len(ky4().junction_name_list)} junctions")
except Exception as e:                                       # noqa: BLE001
    ok = False
    print(f"\n  Network loading failed: {e}")

print("\nSetup OK." if ok else "\nSetup incomplete: fix the items above (see README.md).")
raise SystemExit(0 if ok else 1)
