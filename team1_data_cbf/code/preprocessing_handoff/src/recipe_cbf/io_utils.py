"""Portable UTF-8 JSONL/CSV readers. IDs are always strings."""
import contextlib
import csv
import gzip
import json
import zipfile
from pathlib import Path

# Raw source files live in the project-level data folder:
# ML_TermProject_1/data/{core-data_recipe.csv, core-data-*_rating.csv, RAW_recipes.csv, RAW_interactions.csv}
DATA_DIR = Path(__file__).resolve().parents[5] / "data"


def raw_member(z, basename):
    """Find exactly one member named ``basename`` in a ZIP, ignoring macOS __MACOSX copies."""
    names = [n for n in z.namelist() if Path(n).name == basename and not n.startswith("__MACOSX/")]
    if len(names) != 1:
        raise ValueError(f"Expected exactly one {basename}; found {names}")
    return names[0]


def source_file(source, basename):
    """The file that holds ``basename``: ``source/basename`` for a folder, else ``source`` itself."""
    source = Path(source)
    return source / basename if source.is_dir() else source


@contextlib.contextmanager
def open_source(source, basename):
    """Open raw file ``basename`` as a binary stream.

    ``source`` may be a folder that contains the CSV (default: DATA_DIR),
    the CSV file itself, or a ZIP archive that contains it.
    """
    path = source_file(source, basename)
    if not path.exists():
        raise FileNotFoundError(f"{basename} not found: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z, z.open(raw_member(z, basename)) as f:
            yield f
    else:
        with path.open("rb") as f:
            yield f


def open_text(path, mode="rt"):
    """Open a text file as UTF-8, transparently gunzipping *.gz files."""
    path = Path(path)
    return (gzip.open(path, mode, encoding="utf-8", newline="") if path.suffix == ".gz"
            else path.open(mode.replace("t", ""), encoding="utf-8", newline=""))


def read_jsonl(path):
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    """Write one JSON object per line via a temp file, so a crash never leaves a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.stem + ".tmp" + path.suffix)
    with open_text(temp, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def write_frame(path, frame):
    """Close the complete CSV/gzip before replacing the final output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    frame.to_csv(temp, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temp.replace(path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_csv(path):
    with open_text(path) as f:
        yield from csv.DictReader(f)
