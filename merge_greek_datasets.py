#!/usr/bin/env python3
"""
Merge Greek datasets from multiple sources into consolidated files (max 5GB each) with GPT2 tokenization.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import polars as pl
import tiktoken

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Initialize GPT-4 tokenizer globally
logger.info("Initializing GPT-4 tokenizer...")
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
logger.info("GPT-4 tokenizer initialized")


def get_token_count(text: str) -> int:
    """Get token count for text using GPT-4 tokenizer."""
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.warning(f"Tokenization error: {e}. Returning 0.")
        return 0


def get_parquet_files(source_dir: Path) -> List[Path]:
    """Get all parquet files in a directory."""
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return []

    parquet_files = sorted(source_dir.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files in {source_dir}")
    return parquet_files


def calculate_file_sizes(files: List[Path]) -> List[Tuple[Path, int]]:
    """Calculate sizes of files and return list of (path, size) tuples."""
    file_sizes = []
    for file_path in files:
        size = file_path.stat().st_size
        file_sizes.append((file_path, size))
    return file_sizes


def group_files_by_size(file_sizes: List[Tuple[Path, int]], max_size: int) -> List[List[Path]]:
    """Group files into batches where each batch is approximately max_size."""
    groups = []
    current_group = []
    current_size = 0

    for file_path, size in file_sizes:
        # If adding this file would exceed max_size and current_group is not empty, start new group
        if current_size + size > max_size and current_group:
            groups.append(current_group)
            current_group = [file_path]
            current_size = size
        else:
            current_group.append(file_path)
            current_size += size

    # Add the last group if not empty
    if current_group:
        groups.append(current_group)

    return groups


def standardize_schema(lf: pl.LazyFrame, source_name: str, available_columns: List[str]) -> pl.LazyFrame:
    """
    Standardize schema across different sources.
    - Keep only: text, url, language
    - Add source column
    - Add token_count column (number of GPT-4 tokens)
    """
    # Select available columns in order
    select_cols = []
    if "text" in available_columns:
        select_cols.append(pl.col("text"))
    if "url" in available_columns:
        select_cols.append(pl.col("url"))

    # Handle language column (might be named differently)
    if "language" in available_columns:
        select_cols.append(pl.col("language"))
    elif "in_language" in available_columns:
        select_cols.append(pl.col("in_language").alias("language"))

    # Start with selected columns
    lf = lf.select(select_cols)

    # Add source column
    lf = lf.with_columns(pl.lit(source_name).alias("source"))

    # Add token_count column using map_elements
    logger.info(f"Adding token counts for {source_name}...")
    lf = lf.with_columns(
        pl.col("text").map_elements(
            get_token_count,
            return_dtype=pl.Int64
        ).alias("token_count")
    )

    return lf


def merge_source_dataset(
    source_name: str,
    source_dir: Path,
    output_dir: Path,
    max_size_bytes: int,
) -> List[Path]:
    """
    Merge parquet files from a single source, splitting into max_size_bytes chunks.

    Returns list of output file paths created.
    """
    logger.info("=" * 80)
    logger.info(f"Processing source: {source_name}")
    logger.info(f"Source directory: {source_dir}")
    logger.info("=" * 80)

    # Get all parquet files
    parquet_files = get_parquet_files(source_dir)
    if not parquet_files:
        logger.warning(f"No parquet files found in {source_dir}")
        return []

    # Calculate file sizes
    file_sizes = calculate_file_sizes(parquet_files)
    total_size = sum(size for _, size in file_sizes)
    total_size_gb = total_size / (1024 ** 3)
    logger.info(f"Total size: {total_size_gb:.2f} GB")

    # Group files by size
    file_groups = group_files_by_size(file_sizes, max_size_bytes)
    logger.info(f"Split into {len(file_groups)} group(s) based on {max_size_bytes / (1024**3):.1f} GB limit")

    # Create output directory
    output_subdir = output_dir / source_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    output_files = []

    # Process each group
    for group_idx, file_group in enumerate(file_groups, 1):
        logger.info(f"Processing group {group_idx}/{len(file_groups)} with {len(file_group)} file(s)")

        # Scan all files in the group
        lazy_frames = []
        for file_path in file_group:
            logger.info(f"  Scanning: {file_path.name}")
            lf = pl.scan_parquet(file_path)
            lazy_frames.append(lf)

        # Concatenate all lazy frames
        logger.info(f"Concatenating {len(lazy_frames)} file(s)...")
        combined_lf = pl.concat(lazy_frames, how="vertical_relaxed")

        # Get available columns from first file to determine schema
        first_file_columns = pl.read_parquet_schema(file_group[0]).names()

        # Standardize schema and add source + token_count columns
        logger.info("Standardizing schema and adding source/token_count columns...")
        standardized_lf = standardize_schema(combined_lf, source_name, first_file_columns)

        # Generate output filename
        output_file = output_subdir / f"data_{group_idx:05d}.parquet"

        # Sink to parquet
        logger.info(f"Writing group {group_idx} to {output_file}...")
        standardized_lf.sink_parquet(str(output_file))

        # Log output file size
        output_size = output_file.stat().st_size
        output_size_gb = output_size / (1024 ** 3)
        logger.info(f"✓ Created {output_file.name} ({output_size_gb:.2f} GB)")

        output_files.append(output_file)

    logger.info(f"✓ Completed {source_name}: created {len(output_files)} file(s)")
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Merge Greek datasets with GPT2 tokenization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --max-size 5GB --output-dir data/merged
  %(prog)s --max-size 10GB --output-dir data/merged
        """,
    )

    parser.add_argument(
        "--max-size",
        type=str,
        default="5GB",
        help="Maximum size per output file (e.g., '5GB', '10GB'). Default: 5GB",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/merged",
        help="Output directory for merged files. Default: data/merged",
    )

    args = parser.parse_args()

    # Parse max size
    max_size_str = args.max_size.upper()
    if max_size_str.endswith("GB"):
        max_size_bytes = int(float(max_size_str[:-2]) * 1024 ** 3)
    elif max_size_str.endswith("MB"):
        max_size_bytes = int(float(max_size_str[:-2]) * 1024 ** 2)
    else:
        logger.error(f"Invalid max size format: {args.max_size}. Use format like '5GB' or '500MB'")
        return

    logger.info(f"Max size per file: {max_size_bytes / (1024**3):.2f} GB")

    output_dir = Path(args.output_dir)

    # Define Greek dataset sources
    sources = [
        ("finewiki_el", Path("data/finewiki/elwiki/parquet")),
        ("fineweb_hq_el", Path("data/fineweb_hq/gr/parquet")),
        ("finepdfs_el", Path("data/finepdfs_edu/gr/parquet")),
        ("wikipedia_el", Path("data/wikipedia/gr/parquet")),
    ]

    all_output_files = []

    # Process each source
    for source_name, source_dir in sources:
        try:
            output_files = merge_source_dataset(
                source_name=source_name,
                source_dir=source_dir,
                output_dir=output_dir,
                max_size_bytes=max_size_bytes,
            )
            all_output_files.extend(output_files)
        except Exception as e:
            logger.error(f"Error processing {source_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    logger.info("=" * 80)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Created {len(all_output_files)} merged file(s) in {output_dir}")

    # Calculate total output size
    total_output_size = sum(f.stat().st_size for f in all_output_files)
    total_output_gb = total_output_size / (1024 ** 3)
    logger.info(f"Total output size: {total_output_gb:.2f} GB")

    # List all output files
    logger.info("\nOutput files:")
    for output_file in all_output_files:
        size_gb = output_file.stat().st_size / (1024 ** 3)
        logger.info(f"  {output_file.relative_to(output_dir.parent)} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
