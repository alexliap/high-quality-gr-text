import argparse
import logging
import os
import time
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    retry=retry_if_exception_type((IOError, OSError, ConnectionError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after error: {retry_state.outcome.exception()}"
    ),
)
def sink_to_parquet(chunk_lf: pl.LazyFrame, output_file: Path) -> None:
    """Sink LazyFrame to parquet with retry logic."""
    logger.info(f"Writing to {output_file}")
    chunk_lf.sink_parquet(str(output_file))


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    retry=retry_if_exception_type((IOError, OSError, ConnectionError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retry {retry_state.attempt_number} after error: {retry_state.outcome.exception()}"
    ),
)
def sink_to_ndjson(chunk_lf: pl.LazyFrame, output_file: Path) -> None:
    """Sink LazyFrame to NDJSON with retry logic."""
    logger.info(f"Writing to {output_file}")
    chunk_lf.sink_ndjson(str(output_file))


def process_language(repo_id: str, language_path: str, output_dir: str, output_format: str, lang_name: str, hf_token: str = None):
    """Process parquet files for a specific language using lazy Polars operations."""
    logger.info(f"Processing {lang_name} from {repo_id}")

    # Initialize HuggingFace filesystem
    fs = HfFileSystem(token=hf_token) if hf_token else HfFileSystem()

    # List all parquet files for this language
    full_path = f"datasets/{repo_id}/{language_path}"
    file_paths = fs.glob(f"{full_path}/**/*.parquet")

    if not file_paths:
        logger.error(f"No parquet files found in {full_path}")
        return

    logger.info(f"Found {len(file_paths)} parquet files")

    # Convert to HTTP URLs for Polars
    urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path.split(f'datasets/{repo_id}/')[-1]}"
            for path in file_paths]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_extension = "parquet" if output_format == "parquet" else "jsonl"

    # Process each parquet file separately (no batching)
    logger.info(f"Processing {len(urls)} parquet files one at a time...")

    for idx, url in enumerate(urls, 1):
        logger.info(f"[{idx}/{len(urls)}] Processing: {url}")

        try:
            # Scan and select columns (lazy)
            single_lf = pl.scan_parquet(url, retries=10)
            single_lf = single_lf.select([
                "text", "url", "has_math",
                pl.col("in_language").alias("language"),
            ])

            # Generate output filename
            output_file = output_path / f"data_{idx:05d}.{file_extension}"

            # Sink to output format with retry
            if output_format == "parquet":
                sink_to_parquet(single_lf, output_file)
            else:
                sink_to_ndjson(single_lf, output_file)

            # Log file size
            file_size = output_file.stat().st_size
            file_size_gb = file_size / (1024 ** 3)
            logger.info(f"[{idx}/{len(urls)}] Wrote {output_file} ({file_size_gb:.3f} GB)")

            # Delay between files to avoid rate limiting (except for last file)
            if idx < len(urls):
                delay = 240
                logger.info(f"Waiting {delay}s before next file ...")
                time.sleep(delay)

        except Exception as e:
            logger.error(f"[{idx}/{len(urls)}] Failed to process: {e}")
            continue

    logger.info(f"Finished downloading {lang_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWiki dataset using Polars")
    parser.add_argument("-w", type=str, default="parquet", choices=["parquet", "jsonl"],
                        help="Output format (default: parquet)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token for authentication")

    args = parser.parse_args()

    # Get token from args or environment (.env file loaded at top)
    hf_token = args.token or os.getenv("HF_TOKEN")

    if not hf_token:
        logger.warning("No HF token provided. You may hit rate limits.")

    repo_id = "HuggingFaceFW/finewiki"

    # Process Greek finewiki
    process_language(
        repo_id=repo_id,
        language_path="data/elwiki",
        output_dir=f"data/finewiki/elwiki/{args.w}",
        output_format=args.w,
        lang_name="Greek finewiki",
        hf_token=hf_token
    )

    delay = 240
    logger.info(f"Waiting {delay}s before next file ...")
    time.sleep(delay)

    logger.info("Moving onto downloading English finewiki ...")

    # Process English finewiki
    process_language(
        repo_id=repo_id,
        language_path="data/enwiki",
        output_dir=f"data/finewiki/enwiki/{args.w}",
        output_format=args.w,
        lang_name="English finewiki",
        hf_token=hf_token
    )
