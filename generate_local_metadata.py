#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq


FILE_RE = re.compile(
    r"^(?:RC|RS)_(?P<year>\d{4})-(?P<month>\d{2})(?:_part\d+_\d+)?\.(?P<ext>csv|csv\.gz|parquet|parq)$",
    re.IGNORECASE,
)


@dataclass
class FileInfo:
    path: Path
    rel_path: str
    year: int
    month: int
    ext: str
    size_bytes: int


def parse_file_info(path: Path, root: Path) -> Optional[FileInfo]:
    m = FILE_RE.match(path.name)
    if not m:
        return None
    rel_path = path.relative_to(root).as_posix()
    return FileInfo(
        path=path,
        rel_path=rel_path,
        year=int(m.group("year")),
        month=int(m.group("month")),
        ext=m.group("ext").lower(),
        size_bytes=path.stat().st_size,
    )


def count_csv_rows(path: Path, progress_every: int = 1_000_000) -> int:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    print(f"    ? Counting CSV rows: {path.name}")

    t0 = time.time()
    count = 0

    with opener(path, "rt", encoding="utf-8", newline="") as f:
        for i, _ in enumerate(f, 1):
            count = i
            if progress_every and i % progress_every == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"      ... {i:,} lines read ({rate:,.0f} lines/sec)")

    elapsed = time.time() - t0
    print(f"    ? Done CSV scan: {count:,} total lines in {elapsed:.2f}s")

    return max(0, count - 1)  # subtract header


def parquet_stats(path: Path) -> tuple[int, int, Optional[str], Optional[str]]:
    print(f"    ? Reading parquet metadata: {path.name}")
    t0 = time.time()

    pf = pq.ParquetFile(path)
    md = pf.metadata
    num_rows = md.num_rows if md else 0
    num_row_groups = md.num_row_groups if md else 0

    ts_min = None
    ts_max = None

    if md:
        try:
            schema_names = [md.schema.column(i).name for i in range(md.schema.num_columns)]
        except Exception:
            schema_names = []

        if "time" in schema_names:
            time_idx = schema_names.index("time")
            mins = []
            maxs = []
            for rg_idx in range(md.num_row_groups):
                col = md.row_group(rg_idx).column(time_idx)
                stats = col.statistics
                if not stats or not stats.has_min_max:
                    continue
                if stats.min is not None:
                    mins.append(stats.min)
                if stats.max is not None:
                    maxs.append(stats.max)
            if mins:
                ts_min = min(mins)
            if maxs:
                ts_max = max(maxs)

    if ts_min is not None:
        ts_min = str(ts_min)
    if ts_max is not None:
        ts_max = str(ts_max)

    elapsed = time.time() - t0
    print(
        f"    ? Done parquet metadata: {num_rows:,} rows, "
        f"{num_row_groups:,} row groups in {elapsed:.2f}s"
    )

    return num_rows, num_row_groups, ts_min, ts_max


def build_metadata_rows(data_root: Path) -> list[dict]:
    rows: list[dict] = []
    next_id = 1

    social_group_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    total_groups = len(social_group_dirs)

    print(f"Found {total_groups} social group folders under {data_root}")

    for group_idx, social_group_dir in enumerate(social_group_dirs, start=1):
        social_group = social_group_dir.name
        group_start = time.time()

        print(f"\n[{group_idx}/{total_groups}] Processing group: {social_group}")

        by_month: dict[tuple[int, int], dict[str, FileInfo]] = {}

        files = sorted(social_group_dir.iterdir())
        total_entries = len(files)
        matched_files = 0

        for file_idx, path in enumerate(files, start=1):
            if not path.is_file():
                continue

            print(f"  [{file_idx}/{total_entries}] Scanning: {path.name}")
            info = parse_file_info(path, data_root)
            if info is None:
                print("    - Skipped (filename did not match expected pattern)")
                continue

            matched_files += 1
            month_key = (info.year, info.month)
            bucket = by_month.setdefault(month_key, {})
            if info.ext in {"parquet", "parq"}:
                bucket["parquet"] = info
            elif info.ext in {"csv", "csv.gz"}:
                bucket["csv"] = info

        group_months = sorted(by_month)
        total_months = len(group_months)

        print(
            f"  Group summary: {matched_files} matching files, "
            f"{total_months} year-month buckets"
        )

        for month_idx, (year, month) in enumerate(group_months, start=1):
            print(f"  [{month_idx}/{total_months}] Building row for {year:04d}-{month:02d}")

            bucket = by_month[(year, month)]
            parquet_info = bucket.get("parquet")
            csv_info = bucket.get("csv")

            primary = parquet_info or csv_info
            if primary is None:
                print("    - Skipped (no usable primary file)")
                continue

            if parquet_info is not None:
                num_rows, row_groups, ts_min, ts_max = parquet_stats(parquet_info.path)
                file_format = "parquet"
                file_path = parquet_info.rel_path
                size_bytes = parquet_info.size_bytes
            else:
                num_rows = count_csv_rows(csv_info.path)
                row_groups = 0
                ts_min = None
                ts_max = None
                file_format = "csv.gz" if csv_info.ext == "csv.gz" else "csv"
                file_path = csv_info.rel_path
                size_bytes = csv_info.size_bytes

            rows.append(
                {
                    "id": next_id,
                    "social_group": social_group,
                    "date": f"{year:04d}-{month:02d}-01",
                    "file_path": file_path,
                    "num_rows": num_rows,
                    "file_format": file_format,
                    "size_bytes": size_bytes,
                    "row_groups": row_groups,
                    "ts_min": ts_min,
                    "ts_max": ts_max,
                    "year": year,
                    "month": month,
                    "csv_file_path": csv_info.rel_path if csv_info else "",
                    "csv_available": bool(csv_info),
                }
            )
            print(
                f"    ? Added row #{next_id}: group={social_group}, "
                f"month={year:04d}-{month:02d}, format={file_format}, rows={num_rows:,}"
            )
            next_id += 1

        group_elapsed = time.time() - group_start
        print(f"[{group_idx}/{total_groups}] Finished group: {social_group} in {group_elapsed:.2f}s")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a local metadata table from bulk/data."
    )
    parser.add_argument(
        "data_root",
        help="Path to the bulk/data directory"
    )
    parser.add_argument(
        "--output",
        default="metadata_local.csv",
        help="Output CSV path (default: metadata_local.csv)"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()

    if not data_root.exists() or not data_root.is_dir():
        raise SystemExit(f"Data root does not exist or is not a directory: {data_root}")

    total_start = time.time()
    rows = build_metadata_rows(data_root)

    fieldnames = [
        "id",
        "social_group",
        "date",
        "file_path",
        "num_rows",
        "file_format",
        "size_bytes",
        "row_groups",
        "ts_min",
        "ts_max",
        "year",
        "month",
        "csv_file_path",
        "csv_available",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_elapsed = time.time() - total_start
    print(f"\nWrote {len(rows)} rows to {output}")
    print(f"Completed in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()