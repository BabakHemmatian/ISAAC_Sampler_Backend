from pathlib import Path
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

READ_CSV_KW = dict(
    usecols=[
        "id", "parent id", "text", "author", "time",
        "subreddit", "score", "matched patterns", "source_row"
    ],
    dtype={
        "score": "Int32",
        "source_row": "Int32",
        "id": "string",
        "parent id": "string",
        "text": "string",
        "author": "string",
        "time": "string",
        "subreddit": "string",
        "matched patterns": "string",
    },
    parse_dates=False,
    engine="c",
    low_memory=False,
)

def csv_folder_to_parquet(folder: Path, chunksize: int, compression: str):
    csv_files = sorted(folder.glob("RC_*.csv"))

    if not csv_files:
        print(f"No RC_*.csv files found in {folder}")
        return

    for csv_path in csv_files:
        parquet_path = csv_path.with_suffix(".parquet")

        if parquet_path.exists():
            print(f"[SKIP] {parquet_path.name} already exists")
            continue

        print(f"[CONVERT] {csv_path.name} -> {parquet_path.name}")
        writer = None

        try:
            for chunk in pd.read_csv(csv_path, chunksize=chunksize, **READ_CSV_KW):
                table = pa.Table.from_pandas(chunk, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(
                        parquet_path,
                        table.schema,
                        compression=compression,
                        use_dictionary=True,
                    )

                writer.write_table(table, row_group_size=min(len(chunk), chunksize))

            print(f"[DONE] {parquet_path.name}")

        except Exception as e:
            print(f"[ERROR] Failed on {csv_path.name}: {e}")

        finally:
            if writer:
                writer.close()


def main():
    parser = argparse.ArgumentParser(description="Convert RC_*.csv files to Parquet")

    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing RC_*.csv files"
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per chunk (default: 250000)"
    )

    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "brotli", "none"],
        help="Parquet compression (default: snappy)"
    )

    args = parser.parse_args()

    folder = Path(args.folder)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    csv_folder_to_parquet(
        folder=folder,
        chunksize=args.chunksize,
        compression=None if args.compression == "none" else args.compression,
    )


if __name__ == "__main__":
    main()