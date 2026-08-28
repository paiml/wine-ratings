#!/usr/bin/env python3
"""Repository checks for wine-ratings.

Every check prints its DENOMINATOR ("checked N ...") and FAILS when that
denominator is zero. A check that silently inspected nothing must never read
as a pass -- that is the failure mode this gate exists to avoid.

Usage: python3 ci/checks.py <workflows|notebooks|dataset|crossref|requirements>

Deliberately NOT checked here: that the pinned environment imports and runs.
Measured 2026-08-28 in python:3.9-slim -- `pip install -r requirements.txt`
exits 0 and `pip check` reports "No broken requirements found", yet
`import pandas` raises "numpy.dtype size changed" because pandas==1.1.5 pulls
an unpinned numpy 2.0.2. An install-only check would therefore report a green
it had not measured. Add an import/pipeline check once the pin is repaired.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "wine-ratings.csv"
REQUIREMENTS = ROOT / "requirements.txt"

# csv fields can be very large (the `notes` column holds free prose).
csv.field_size_limit(10 * 1024 * 1024)


class CheckFailed(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def require_nonzero(n: int, noun: str) -> None:
    """A zero denominator is a failure, never a pass."""
    require(n > 0, f"checked 0 {noun} -- nothing was inspected, so this is not a pass")


def notebooks() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.ipynb") if ".git" not in p.parts)


def strip_ipython_magics(source: str) -> str | None:
    """Return plain-Python source for a cell, or None if it is a cell magic."""
    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return None  # whole-cell magic: not Python, nothing to compile
    kept = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            kept.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            kept.append(line)
    return "\n".join(kept)


def check_workflows() -> None:
    """Every GitHub Actions workflow parses as YAML and declares jobs."""
    import yaml  # provided by the job that runs this check

    files = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    jobs = 0
    for path in files:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise CheckFailed(f"{path.name}: is not valid YAML: {exc}") from exc
        require(isinstance(doc, dict), f"{path.name}: not a YAML mapping")
        require("jobs" in doc and doc["jobs"], f"{path.name}: declares no jobs")
        require(True in doc or "on" in doc, f"{path.name}: declares no triggers")
        jobs += len(doc["jobs"])
        print(f"  ok {path.relative_to(ROOT)} ({len(doc['jobs'])} job(s))")
    print(f"checked {len(files)} workflow file(s), {jobs} job(s)")
    require_nonzero(len(files), "workflow files")
    require_nonzero(jobs, "workflow jobs")


def check_notebooks() -> None:
    """Every notebook is valid nbformat and every code cell compiles."""
    files = notebooks()
    cells = 0
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        require(doc.get("nbformat") == 4, f"{path.name}: unsupported nbformat {doc.get('nbformat')!r}")
        require(isinstance(doc.get("cells"), list), f"{path.name}: no cell list")
        compiled = 0
        for index, cell in enumerate(doc["cells"]):
            require(cell.get("cell_type") in {"code", "markdown", "raw"},
                    f"{path.name}: cell {index} has unknown type {cell.get('cell_type')!r}")
            if cell["cell_type"] != "code":
                continue
            source = strip_ipython_magics("".join(cell.get("source", [])))
            if source is None or not source.strip():
                continue
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise CheckFailed(f"{path.name}: cell {index} is not valid Python: {exc}") from exc
            compiled += 1
        print(f"  ok {path.relative_to(ROOT)} ({compiled} code cell(s) compiled)")
        cells += compiled
    print(f"checked {len(files)} notebook(s), {cells} code cell(s)")
    require_nonzero(len(files), "notebooks")
    require_nonzero(cells, "notebook code cells")


def check_dataset() -> None:
    """The committed dataset parses and every row honours the header."""
    require(DATASET.exists(), f"{DATASET.name} is missing from the repository")
    with DATASET.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        require(header is not None, "wine-ratings.csv is empty")
        width = len(header)
        require(width > 1, f"wine-ratings.csv header has {width} column(s)")
        require("rating" in header, "wine-ratings.csv has no 'rating' column")
        rating_at = header.index("rating")
        rows = 0
        rated = 0
        for line, row in enumerate(reader, start=2):
            rows += 1
            require(len(row) == width,
                    f"wine-ratings.csv line {line}: {len(row)} field(s), header has {width}")
            value = row[rating_at]
            if value == "":
                continue
            try:
                rating = float(value)
            except ValueError as exc:
                raise CheckFailed(f"wine-ratings.csv line {line}: rating {value!r} is not numeric") from exc
            require(0 <= rating <= 100,
                    f"wine-ratings.csv line {line}: rating {rating} outside 0..100")
            rated += 1
    print(f"  header: {header}")
    print(f"checked {rows} dataset row(s), {rated} numeric rating(s), {width} column(s)")
    require_nonzero(rows, "dataset rows")
    require_nonzero(rated, "numeric ratings")


def dataset_header() -> list[str]:
    require(DATASET.exists(), f"{DATASET.name} is missing from the repository")
    with DATASET.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


REPO_URL = re.compile(
    r"https://raw\.githubusercontent\.com/paiml/wine-ratings/[^/]+/([^\s\"')]+)")
README_LOCAL_REF = re.compile(r'(?:src|href)="(?!https?:|#|mailto:)([^"]+)"')


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _code_cell_tree(cell: dict) -> ast.Module | None:
    """Parsed AST for a code cell, or None if it is not plain Python.

    Syntax errors are swallowed here on purpose: check_notebooks owns syntax
    and reports it precisely, so re-reporting it here would double-count.
    """
    if cell.get("cell_type") != "code":
        return None
    plain = strip_ipython_magics(_cell_source(cell))
    if plain is None:
        return None
    try:
        return ast.parse(plain)
    except SyntaxError:
        return None


def _dropped_columns(tree: ast.Module) -> list[str]:
    """Column names passed to any `.drop([...])` call in a cell."""
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "drop"):
            continue
        if node.args:
            names.extend(_string_literals(node.args[0]))
    return names


def _check_notebook_urls(path: Path, cells: list[dict]) -> int:
    """Every raw.githubusercontent URL into this repo resolves to a real file."""
    found = 0
    for index, cell in enumerate(cells):
        for match in REPO_URL.finditer(_cell_source(cell)):
            require((ROOT / match.group(1)).exists(),
                    f"{path.name}: cell {index} loads {match.group(1)!r}, which is not in this repo")
            print(f"  ok {path.name} -> {match.group(1)}")
            found += 1
    return found


def _check_notebook_columns(path: Path, cells: list[dict], header: list[str]) -> int:
    """Every column the notebook drops exists in the dataset header."""
    found = 0
    for index, cell in enumerate(cells):
        tree = _code_cell_tree(cell)
        if tree is None:
            continue
        for name in _dropped_columns(tree):
            require(name in header,
                    f"{path.name}: cell {index} drops column {name!r}, "
                    f"absent from the dataset header {header}")
            print(f"  ok {path.name} drops existing column {name!r}")
            found += 1
    return found


def _check_readme_refs() -> int:
    """Every local path the README links or embeds exists."""
    readme = ROOT / "README.md"
    require(readme.exists(), "README.md is missing from the repository")
    found = 0
    for match in README_LOCAL_REF.finditer(readme.read_text(encoding="utf-8")):
        require((ROOT / match.group(1)).exists(),
                f"README.md references {match.group(1)!r}, which does not exist")
        print(f"  ok README.md -> {match.group(1)}")
        found += 1
    return found


def check_crossref() -> None:
    """Paths and columns referenced by the notebooks and README really exist.

    Column names are read back out of the notebook's own AST rather than
    hand-listed here, so this stays true when the notebook changes.
    """
    header = dataset_header()
    refs = 0
    columns = 0
    for path in notebooks():
        cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
        refs += _check_notebook_urls(path, cells)
        columns += _check_notebook_columns(path, cells, header)
    refs += _check_readme_refs()

    print(f"checked {refs} local reference(s) and {columns} notebook column reference(s)")
    require_nonzero(refs, "local references")
    require_nonzero(columns, "notebook column references")


def _string_literals(node: ast.AST | None) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def parse_requirements() -> list[tuple[str, str]]:
    pinned = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.*+!-]*)", line)
        require(match is not None,
                f"requirements.txt: {line!r} is not a name==version exact pin")
        pinned.append((match.group(1), match.group(2)))
    return pinned


def check_requirements() -> None:
    """Every dependency is an exact pin (this repo is teaching material)."""
    pinned = parse_requirements()
    for name, version in pinned:
        print(f"  ok {name} pinned to {version}")
    print(f"checked {len(pinned)} requirement(s), all exact pins")
    require_nonzero(len(pinned), "requirements")


CHECKS = {
    "workflows": check_workflows,
    "notebooks": check_notebooks,
    "dataset": check_dataset,
    "crossref": check_crossref,
    "requirements": check_requirements,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in CHECKS:
        print(f"usage: {argv[0]} <{'|'.join(CHECKS)}>", file=sys.stderr)
        return 2
    name = argv[1]
    print(f"== {name} ==")
    try:
        CHECKS[name]()
    except CheckFailed as exc:
        print(f"FAIL [{name}]: {exc}", file=sys.stderr)
        return 1
    print(f"PASS [{name}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
