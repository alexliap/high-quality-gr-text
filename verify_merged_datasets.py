#!/usr/bin/env python3
"""
Verify merged Greek datasets - check schema, row counts, file sizes, and tokenization.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def verify_file(file_path: Path) -> Dict[str, Any]:
    """Verify a single parquet file and return statistics."""
    logger.info(f"Verifying {file_path.name}...")

    # Read file
    df = pl.read_parquet(file_path)

    # Get basic stats
    num_rows = len(df)
    file_size = file_path.stat().st_size
    file_size_gb = file_size / (1024 ** 3)

    # Check schema
    expected_columns = {"text", "url", "language", "source", "token_count"}
    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns

    # Check token_count column
    token_count_is_int = df.schema["token_count"] == pl.Int64 if "token_count" in df.columns else False

    # Calculate token statistics
    token_stats = {}
    if "token_count" in df.columns:
        # Get token count statistics
        token_counts = df["token_count"]

        token_stats = {
            "min": token_counts.min(),
            "max": token_counts.max(),
            "mean": round(token_counts.mean(), 2),
            "median": token_counts.median(),
        }

    # Check for nulls in critical columns
    null_counts = {}
    for col in ["text", "url", "token_count"]:
        if col in df.columns:
            null_counts[col] = df[col].null_count()

    # Sample first row
    sample_row = None
    if num_rows > 0:
        first_row = df.row(0, named=True)
        sample_row = {
            "text_length": len(first_row.get("text", "")),
            "url": first_row.get("url", ""),
            "language": first_row.get("language", ""),
            "source": first_row.get("source", ""),
            "token_count": first_row.get("token_count", 0),
        }

    return {
        "file_name": file_path.name,
        "num_rows": num_rows,
        "file_size_gb": round(file_size_gb, 3),
        "schema_valid": len(missing_columns) == 0 and len(extra_columns) == 0,
        "missing_columns": list(missing_columns),
        "extra_columns": list(extra_columns),
        "token_count_is_int_type": token_count_is_int,
        "token_stats": token_stats,
        "null_counts": null_counts,
        "sample": sample_row,
    }


def verify_subset(subset_dir: Path) -> Dict[str, Any]:
    """Verify all files in a subset directory."""
    logger.info("=" * 80)
    logger.info(f"Verifying subset: {subset_dir.name}")
    logger.info("=" * 80)

    parquet_files = sorted(subset_dir.glob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {subset_dir}")
        return {
            "subset_name": subset_dir.name,
            "num_files": 0,
            "files": [],
            "total_rows": 0,
            "total_size_gb": 0,
        }

    logger.info(f"Found {len(parquet_files)} file(s)")

    file_stats = []
    total_rows = 0
    total_size_gb = 0.0

    for file_path in parquet_files:
        try:
            stats = verify_file(file_path)
            file_stats.append(stats)

            total_rows += stats["num_rows"]
            total_size_gb += stats["file_size_gb"]

            # Log summary for this file
            logger.info(f"  ✓ {stats['file_name']}: {stats['num_rows']:,} rows, {stats['file_size_gb']:.2f} GB")

            if not stats["schema_valid"]:
                logger.warning(f"    Schema issue - Missing: {stats['missing_columns']}, Extra: {stats['extra_columns']}")

            if stats["null_counts"]:
                null_info = ", ".join([f"{col}: {count}" for col, count in stats["null_counts"].items() if count > 0])
                if null_info:
                    logger.warning(f"    Null values - {null_info}")

            if stats["token_stats"]:
                logger.info(f"    Token stats - Mean: {stats['token_stats']['mean']}, "
                           f"Min: {stats['token_stats']['min']}, Max: {stats['token_stats']['max']}")

        except Exception as e:
            logger.error(f"  ✗ Error verifying {file_path.name}: {e}")
            file_stats.append({
                "file_name": file_path.name,
                "error": str(e),
            })

    logger.info(f"\nSubset summary: {total_rows:,} total rows, {total_size_gb:.2f} GB")

    return {
        "subset_name": subset_dir.name,
        "num_files": len(parquet_files),
        "files": file_stats,
        "total_rows": total_rows,
        "total_size_gb": round(total_size_gb, 3),
    }


def check_file_sizes(subset_stats: Dict[str, Any], max_size_gb: float) -> bool:
    """Check if any file exceeds the maximum size."""
    all_valid = True
    for file_stat in subset_stats.get("files", []):
        if file_stat.get("file_size_gb", 0) > max_size_gb:
            logger.error(f"  ✗ {file_stat['file_name']} exceeds {max_size_gb} GB: {file_stat['file_size_gb']} GB")
            all_valid = False
    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Verify merged Greek datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --merged-dir data/merged
  %(prog)s --merged-dir data/merged --stats-output merge_stats.json
        """,
    )

    parser.add_argument(
        "--merged-dir",
        type=str,
        default="data/merged",
        help="Path to merged dataset directory. Default: data/merged",
    )

    parser.add_argument(
        "--stats-output",
        type=str,
        default="merge_stats.json",
        help="Path to output statistics JSON file. Default: merge_stats.json",
    )

    parser.add_argument(
        "--max-size-gb",
        type=float,
        default=5.0,
        help="Maximum file size in GB. Default: 5.0",
    )

    args = parser.parse_args()

    # Validate merged directory exists
    merged_dir = Path(args.merged_dir)
    if not merged_dir.exists():
        parser.error(f"Merged directory not found: {args.merged_dir}")

    if not merged_dir.is_dir():
        parser.error(f"Not a directory: {args.merged_dir}")

    # Get subset folders
    subset_folders = sorted([d for d in merged_dir.iterdir() if d.is_dir()])
    if not subset_folders:
        logger.error(f"No subset folders found in {args.merged_dir}")
        return

    logger.info(f"Found {len(subset_folders)} subset(s): {[f.name for f in subset_folders]}")
    logger.info(f"Maximum file size: {args.max_size_gb} GB")
    logger.info("")

    # Verify each subset
    all_stats = {}
    for subset_dir in subset_folders:
        subset_stats = verify_subset(subset_dir)
        all_stats[subset_dir.name] = subset_stats

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)

    total_files = 0
    total_rows = 0
    total_size_gb = 0.0
    all_schemas_valid = True
    all_sizes_valid = True

    for subset_name, stats in all_stats.items():
        total_files += stats["num_files"]
        total_rows += stats["total_rows"]
        total_size_gb += stats["total_size_gb"]

        logger.info(f"\n{subset_name}:")
        logger.info(f"  Files: {stats['num_files']}")
        logger.info(f"  Rows: {stats['total_rows']:,}")
        logger.info(f"  Size: {stats['total_size_gb']:.2f} GB")

        # Check schemas
        for file_stat in stats.get("files", []):
            if not file_stat.get("schema_valid", True):
                logger.warning(f"  ⚠ Schema issue in {file_stat['file_name']}")
                all_schemas_valid = False

        # Check file sizes
        if not check_file_sizes(stats, args.max_size_gb):
            all_sizes_valid = False

    logger.info("")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Total size: {total_size_gb:.2f} GB")

    # Final validation
    logger.info("")
    if all_schemas_valid:
        logger.info("✓ All schemas are valid")
    else:
        logger.error("✗ Some files have schema issues")

    if all_sizes_valid:
        logger.info(f"✓ All files are under {args.max_size_gb} GB")
    else:
        logger.error(f"✗ Some files exceed {args.max_size_gb} GB")

    # Write statistics to JSON
    output_path = Path(args.stats_output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Statistics written to {output_path}")


if __name__ == "__main__":
    main()
