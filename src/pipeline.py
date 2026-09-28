"""Single entry point that reproduces every deliverable end to end.

    python -m src.pipeline          # EDA -> cleaning log -> final model -> predictions -> December
                                    #   -> score.py -> DOCX report                      (~1 min)
    python -m src.pipeline --cv     # additionally re-runs every validation experiment and
                                    #   regenerates the metric tables first            (~1.5 h)

Without --cv the report is built from the committed metric CSVs in reports/metrics/.
All randomness is seeded (config.SEED).
"""
import argparse
import subprocess
import sys

from . import december, eda, ensemble, predict, prepare, report, train, validate
from .config import DECEMBER_CSV, PREDICTIONS_CSV, ROOT


def step(name):
    print(f"\n=== {name} ===", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", action="store_true", help="re-run all validation experiments (slow)")
    args = ap.parse_args()

    step("1/7 EDA + data-quality audit"); eda.main()
    step("2/7 cleaning log"); prepare.main()
    if args.cv:
        step("CV: time / city / random folds")
        for scheme in ("time", "city", "random"):
            validate.run(scheme)
        validate.report()
        ensemble.main(validate.FINAL)
        december.variant_stats()
    step("3/7 train final model"); model = train.main()
    step("4/7 predict validation + December"); predict.main(model)
    step("5/7 December drivers"); december.main()
    step("6/7 score.py")
    subprocess.run([sys.executable, "score.py", "--predictions", str(PREDICTIONS_CSV.relative_to(ROOT)),
                    "--december-predictions", str(DECEMBER_CSV.relative_to(ROOT))], cwd=ROOT, check=True)
    step("7/7 DOCX report"); report.main()


if __name__ == "__main__":
    main()
