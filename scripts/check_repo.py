import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "homeworks").glob("**/*.ipynb"))
PLACEHOLDERS = [
    "scratch cell",
    "replace this",
    "next: implement",
    "todo",
]
ALLOWED_MARKDOWN = {
    "README.md",
    "agent.md",
}
SKIP_DIRS = {
    ".git",
    ".miniforge",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "executed",
}


def fail(message):
    print(f"ERROR: {message}")
    return 1


def check_markdown_files():
    failures = 0
    for path in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel not in ALLOWED_MARKDOWN:
            failures += fail(f"unexpected markdown file: {rel}")
    return failures


def check_notebooks():
    failures = 0
    if len(NOTEBOOKS) != 12:
        failures += fail(f"expected 12 notebooks, found {len(NOTEBOOKS)}")
    for path in NOTEBOOKS:
        rel = path.relative_to(ROOT).as_posix()
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures += fail(f"{rel} is not valid JSON: {exc}")
            continue
        if nb.get("nbformat") != 4:
            failures += fail(f"{rel} is not nbformat 4")
        text = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
        lower = text.lower()
        for token in PLACEHOLDERS:
            if token in lower:
                failures += fail(f"{rel} contains placeholder text: {token}")
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source)
            except SyntaxError as exc:
                failures += fail(f"{rel} code cell {idx} has syntax error: {exc}")
    return failures


def check_readme_links():
    failures = 0
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        path = ROOT / clean
        if not path.exists():
            failures += fail(f"README link target does not exist: {target}")
    return failures


def main():
    failures = 0
    failures += check_markdown_files()
    failures += check_notebooks()
    failures += check_readme_links()
    if failures:
        print(f"failed with {failures} issue(s)")
        return 1
    print("repo check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
