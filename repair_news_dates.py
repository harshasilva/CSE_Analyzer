"""Rewrite placeholder 1 January dates using the real createdDate text."""

from __future__ import annotations

import csv
from pathlib import Path

from cse_common import parse_cse_date

OUTPUT_DIR = Path("cse_output")
DATE_FIELDS = ("createdDate", "announcementDate", "publishedDate", "date")


def _is_placeholder(value: str | None, parsed) -> bool:
    if parsed is None:
        return True
    text = str(value or "").strip()
    return parsed.month == 1 and parsed.day == 1 and text in {parsed.isoformat(), f"{parsed.year}-01-01"}


def _better_date(row: dict):
    for field in DATE_FIELDS:
        parsed = parse_cse_date(row.get(field))
        if parsed is not None:
            return parsed
    return None


def repair_file(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "_parsed_date" not in fieldnames:
        return 0, 0

    changed = 0
    for row in rows:
        current_raw = row.get("_parsed_date")
        current = parse_cse_date(current_raw)
        if not _is_placeholder(current_raw, current):
            continue
        better = _better_date(row)
        if better is None or better == current:
            continue
        row["_parsed_date"] = better.isoformat()
        if "_year" in fieldnames:
            row["_year"] = str(better.year)
        changed += 1

    if changed:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return changed, len(rows)


def main() -> None:
    if not OUTPUT_DIR.exists():
        print(f"No output folder: {OUTPUT_DIR}")
        return
    total_changed = 0
    for path in sorted(OUTPUT_DIR.rglob("*.csv")):
        changed, rows = repair_file(path)
        if rows and "_parsed_date" in path.name or changed:
            print(f"{path.relative_to(OUTPUT_DIR)}: updated {changed} of {rows}")
            total_changed += changed
    print(f"Updated {total_changed} announcement dates.")


if __name__ == "__main__":
    main()
