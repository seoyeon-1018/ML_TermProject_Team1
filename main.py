"""Run the whole project with one command.

    python main.py                      # every stage (about 2 hours)
    python main.py --quick              # fast end-to-end check (1 seed, 2 epochs, smaller grids)
    python main.py --stages db cbf      # only some stages
    python main.py --list               # show the stages and steps

Stages (in order)
  check    raw CSVs in data/ and the Python / LightFM versions
  db       preprocessing: raw CSVs -> allrecipes.db, foodcom.db              (team1_data_cbf)
  cbf      CBF evaluation on Allrecipes test and Food.com + report analysis  (team1_data_cbf)
  h5       main experiment, H = first 5 positives                            (team_5H)
  fullh    additional experiment, H = all past positives                     (team_fullH)
  compare  H5 vs H-all comparison table and figures                          (comparison)

Nothing in the repository is overwritten. Every stage works in its own folder under
--out (default outputs/, or outputs_quick/ with --quick): the original scripts are copied there
and run from that folder, so their result files land there. Databases are hard-linked
(copied if linking is impossible). Logs: <out>/logs/.

A step whose output files already exist is skipped (re-run it with --force).
Hyperparameter tuning (lightfm_tune.py, 30+ min) is skipped by default and the saved
tuning.csv is copied; use --retune to run it again.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILES = ["core-data_recipe.csv", "core-data-train_rating.csv", "core-data-valid_rating.csv",
              "core-data-test_rating.csv", "RAW_recipes.csv", "RAW_interactions.csv"]
STAGES = ["check", "db", "cbf", "h5", "fullh", "compare"]
CBF_CODE = ROOT / "team1_data_cbf" / "code"
PREP = CBF_CODE / "preprocessing_handoff"
H5_SRC, FULLH_SRC = ROOT / "team_5H", ROOT / "team_fullH"
DBS = ["allrecipes.db", "foodcom.db"]


# --------------------------------------------------------------------------- helpers
class Step:
    def __init__(self, name, cmd, cwd, outputs=(), prepare=None):
        self.name, self.cmd, self.cwd, self.outputs, self.prepare = name, cmd, Path(cwd), list(outputs), prepare


def py(*args):
    return [sys.executable, "-W", "ignore", *[str(a) for a in args]]


def py_patched(module, patches):
    """Run ``module.main()`` after overriding module-level settings (used by --quick)."""
    code = f"import {module} as m\n" + "".join(f"{p}\n" for p in patches) + "m.main()"
    return [sys.executable, "-W", "ignore", "-c", code]


def link_or_copy(src, dst):
    """Hard-link src to dst (copy if the file system does not allow it); refresh stale files."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        same = False
        try:
            same = os.path.samefile(src, dst)
        except OSError:
            pass
        if same or (dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime):
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_scripts(src_dir, work, extra=()):
    """Copy the *.py files (always refreshed) and extra files of an experiment folder into ``work``."""
    work.mkdir(parents=True, exist_ok=True)
    for f in list(src_dir.glob("*.py")) + [src_dir / e for e in extra]:
        shutil.copy2(f, work / f.name)


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def run_steps(stage, steps, logs, force):
    for i, step in enumerate(steps, 1):
        label = f"[{stage} {i}/{len(steps)}] {step.name}"
        if step.outputs and not force and all(Path(o).exists() for o in step.outputs):
            print(f"{label:<58} skipped (output exists; --force to re-run)")
            continue
        if step.prepare:
            step.prepare()
        if step.cmd is None:          # preparation-only step
            print(f"{label:<58} done")
            continue
        log = logs / f"{stage}_{i:02d}_{step.name.split()[0].replace('.py', '')}.log"
        print(f"{label:<58} running ...", end="", flush=True)
        t = time.time()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                   MPLBACKEND="Agg")
        with open(log, "w", encoding="utf-8") as f:
            f.write(f"$ (cd {step.cwd}) {' '.join(map(str, step.cmd))}\n\n")
            f.flush()
            code = subprocess.run(step.cmd, cwd=step.cwd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        print(f"\r{label:<58} {'done' if code == 0 else 'FAILED'} ({fmt_time(time.time() - t)})")
        if code != 0:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join("    " + line for line in lines[-25:]))
            raise SystemExit(f"\nStopped: '{step.name}' failed (exit code {code}). Full log: {log}")


# --------------------------------------------------------------------------- stages
def stage_check(a, out):
    print(f"Python {platform.python_version()} ({sys.executable})")
    missing = [f for f in DATA_FILES if not (a.data / f).exists()]
    if missing:
        raise SystemExit(f"Missing raw files in {a.data}: {missing}\nSee data/README.md for the download links.")
    print(f"Raw data: all {len(DATA_FILES)} files found in {a.data}")
    try:
        import lightfm
        version = lightfm.__version__
    except ImportError:
        version = None
    if version is None:
        print("LightFM: not installed (needed for the h5 and fullh stages; see environment.yml)")
    elif version != "1.16":
        print(f"LightFM {version}: the project is tested with 1.16"
              + (" - on Windows, 1.17 crashes with warp loss (h5 stage)" if sys.platform == "win32" else ""))
    else:
        print("LightFM 1.16: OK")


def stage_db(a, out):
    db, prep = out / "db", out / "db" / "prep"
    steps = [
        Step("run_preprocessing.py allrecipes", py("run_preprocessing.py", "allrecipes", "--data-dir", a.data,
                                                    "--output", prep), PREP,
             [prep / "data" / "allrecipes" / "test_evaluation" / "H.csv.gz"]),
        Step("run_preprocessing.py foodcom", py("run_preprocessing.py", "foodcom", "--source", a.data,
                                                 "--output", prep), PREP,
             [prep / "data" / "foodcom" / "H.csv.gz"]),
        Step("build_db.py", py("build_db.py", "--prepared", prep, "--dest", db), PREP, [db / d for d in DBS],
             prepare=lambda: (db.mkdir(parents=True, exist_ok=True),
                              shutil.copy2(CBF_CODE / "cbf_model.json", db / "cbf_model.json"))),
    ]
    return steps


def need_db(out):
    missing = [d for d in DBS if not (out / "db" / d).exists()]
    if missing:
        raise SystemExit(f"{out / 'db'} has no {missing}. Run the db stage first: python main.py --stages db"
                         + (" --quick" if out.name.endswith("quick") else ""))


def stage_cbf(a, out):
    work = out / "cbf"

    def prepare():
        need_db(out)
        copy_scripts(CBF_CODE, work, ["cbf_model.json"])
        for d in DBS:
            link_or_copy(out / "db" / d, work / d)

    return [
        Step("prepare workspace", None, work, prepare=prepare),
        Step("cbf.py evaluate allrecipes", py("cbf.py", "evaluate", "--db", "allrecipes.db", "--split", "test",
                                              "--output", "cbf_allrecipes_result"), work,
             [work / "cbf_allrecipes_result" / "metrics.json"]),
        Step("cbf.py evaluate foodcom", py("cbf.py", "evaluate", "--db", "foodcom.db", "--split", "test",
                                           "--output", "cbf_foodcom_result"), work,
             [work / "cbf_foodcom_result" / "metrics.json"]),
        Step("cbf_report_analysis.py", py("cbf_report_analysis.py"), work, [work / "cbf_analysis" / "analysis.json"]),
    ]


def tuning_step(src, work, a):
    res = work / "lightfm_results"
    if a.retune:
        return Step("lightfm_tune.py", py("lightfm_tune.py"), work, [res / "tuning.csv"])

    def copy():
        res.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "lightfm_results" / "tuning.csv", res / "tuning.csv")
    return Step("tuning.csv (saved result copied; --retune to run)", None, work, prepare=copy)


def stage_h5(a, out):
    work, res = out / "team_5H", out / "team_5H" / "lightfm_results"
    q = a.quick

    def prepare():
        need_db(out)
        copy_scripts(H5_SRC, work, ["cbf_model.json"])
        for d in DBS:
            link_or_copy(out / "db" / d, work / d)

    quick_final = ["--max-epochs", "2", "--seeds", "42"] if q else []
    smoke = ["--loss", "warp", "--dim", "32", "--lr", "0.02", "--alpha", "0.0"] + (
        ["--max-epochs", "2", "--seeds", "42"] if q else ["--max-epochs", "20", "--seeds", "42", "43", "44"])
    diag = (py_patched("lightfm_diagnose", ["m.CONFIG.max_epochs = 2"]) if q else py("lightfm_diagnose.py"))
    cos = (py_patched("lightfm_final_cosine", ["m.CONFIG.max_epochs = 2", "m.SEEDS = [42]"]) if q
           else py("lightfm_final_cosine.py"))
    rew = (py_patched("lightfm_final_reweight", ["m.CONFIG.max_epochs = 2", "m.BETAS = [0.0, 0.5]",
                                                  "m.ALPHAS = [0.0]", "m.FINAL_SEEDS = [42]"]) if q
           else py("lightfm_final_reweight.py"))
    return [
        Step("prepare workspace", None, work, prepare=prepare),
        Step("split_train.py", py("split_train.py"), work, [work / "train_split" / "train_targets.csv"]),
        Step("lightfm_smoke.py (valid learning curves)", py("lightfm_smoke.py", *smoke), work,
             [res / "valid_warp_d32_lr0.02_a0.0.json"]),
        tuning_step(H5_SRC, work, a),
        Step("lightfm_final.py", py("lightfm_final.py", *quick_final), work, [res / "final_test.json"]),
        Step("lightfm_diagnose.py", diag, work, []),
        Step("lightfm_final_cosine.py", cos, work, [res / "final_test_cosine.json"]),
        Step("lightfm_final_reweight.py", rew, work, [res / "final_test_reweight.json"]),
        Step("make_plots.py", py("make_plots.py"), work, []),
    ]


def stage_fullh(a, out):
    work, res = out / "team_fullH", out / "team_fullH" / "lightfm_results"
    quick_final = ["--max-epochs", "2", "--seeds", "42"] if a.quick else []

    def prepare():
        need_db(out)
        copy_scripts(FULLH_SRC, work)

    return [
        Step("prepare workspace", None, work, prepare=prepare),
        Step("build_full_H.py", py("build_full_H.py", "--db-dir", out / "db", "--allrecipes", a.data,
                                   "--foodcom", a.data, "--out", "."), work, [work / d for d in DBS]),
        Step("split_train.py", py("split_train.py"), work, [work / "train_split" / "train_targets.csv"]),
        tuning_step(FULLH_SRC, work, a),
        Step("lightfm_final.py (logistic, dim 16, lr 0.005, alpha 1e-4)",
             py("lightfm_final.py", "--loss", "logistic", "--dim", "16", "--lr", "0.005", "--alpha", "0.0001",
                *quick_final), work, [res / "final_test.json"]),
    ]


def stage_compare(a, out):
    h5, hall = out / "team_5H", out / "team_fullH"

    def prepare():
        need = [h5 / "lightfm_results" / f for f in ["final_test.json", "final_test_cosine.json",
                                                     "final_test_reweight.json"]] + [hall / "lightfm_results" / "final_test.json"]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            raise SystemExit("compare needs the h5 and fullh stages first. Missing:\n  " + "\n  ".join(missing))

    return [Step("compare.py", py(ROOT / "comparison" / "compare.py", "--h5", h5, "--hall", hall,
                                  "--out", out / "comparison"), ROOT, [], prepare=prepare)]


BUILDERS = {"db": stage_db, "cbf": stage_cbf, "h5": stage_h5, "fullh": stage_fullh, "compare": stage_compare}


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES, help="stages to run (default: all)")
    p.add_argument("--quick", action="store_true", help="fast check: 1 seed, 2 epochs, small grids (not the reported numbers)")
    p.add_argument("--retune", action="store_true", help="run hyperparameter tuning instead of copying tuning.csv")
    p.add_argument("--force", action="store_true", help="re-run steps whose outputs already exist")
    p.add_argument("--data", type=Path, default=ROOT / "data", help="folder with the raw CSVs (default: data/)")
    p.add_argument("--out", type=Path, help="output folder (default: outputs/, or outputs_quick/ with --quick)")
    p.add_argument("--list", action="store_true", help="show the steps and exit")
    a = p.parse_args()
    a.data = a.data.resolve()
    out = (a.out or ROOT / ("outputs_quick" if a.quick else "outputs")).resolve()
    stages = [s for s in STAGES if s in a.stages]

    if a.list:
        for s in STAGES:
            print(s)
            for step in ([] if s == "check" else BUILDERS[s](a, out)):
                print(f"   - {step.name}")
        return

    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out}{'   (QUICK MODE: numbers are not the reported results)' if a.quick else ''}")
    t0 = time.time()
    for s in stages:
        print(f"\n=== {s} " + "=" * (60 - len(s)))
        if s == "check":
            stage_check(a, out)
        else:
            run_steps(s, BUILDERS[s](a, out), logs, a.force)
    print(f"\nAll requested stages finished in {fmt_time(time.time() - t0)}. Results: {out}")


if __name__ == "__main__":
    main()
