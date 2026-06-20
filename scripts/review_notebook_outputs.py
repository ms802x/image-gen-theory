import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = 0
    for path in sorted((ROOT / "homeworks").glob("**/*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        executed = 0
        images = 0
        errors = 0
        has_cpu_note = False
        for cell in nb["cells"]:
            source = "".join(cell.get("source", []))
            has_cpu_note = has_cpu_note or "CPU smoke run" in source
            if cell.get("cell_type") == "code":
                executed += cell.get("execution_count") is not None
            for output in cell.get("outputs", []):
                errors += output.get("output_type") == "error"
                images += "image/png" in output.get("data", {})
        rel = path.relative_to(ROOT)
        print(
            f"{rel}: executed_code={executed} png_outputs={images} "
            f"errors={errors} cpu_note={has_cpu_note}"
        )
        if errors:
            failures += 1
        if "hw0_" not in str(path) and "hw11_" not in str(path) and not has_cpu_note:
            print(f"  missing CPU-smoke interpretation note: {rel}")
            failures += 1
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
