"""Launcher for the preprocessing modules in src/recipe_cbf.

Usage (raw CSVs are read from the project data folder ML_TermProject_1/data by default):
  python run_preprocessing.py allrecipes --output out
  python run_preprocessing.py foodcom --output out          # add --profile-rule recent for the sensitivity run
  # other sources: allrecipes --data-dir <folder> or --recipe-archive/--train-archive/... <CSV or ZIP>;
  #                foodcom --source <folder or ZIP>  (--zip still works)
  python build_db.py --prepared out --dest ..                                   # pack into allrecipes.db / foodcom.db
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # make the recipe_cbf package importable

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_preprocessing.py foodcom|allrecipes [options]")
    command = sys.argv.pop(1)
    if command == "foodcom":
        from recipe_cbf.prepare import main
    elif command == "allrecipes":
        from recipe_cbf.allrecipes import main
    else:
        raise SystemExit("Expected foodcom or allrecipes")
    main()
