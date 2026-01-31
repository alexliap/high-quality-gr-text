#!/usr/bin/env python3
"""
Upload merged Greek datasets to Hugging Face Hub with subset configurations.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def get_folder_size(folder_path):
    """Calculate total size of all files in a folder"""
    total_size = 0
    for root, _dirs, files in os.walk(folder_path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size


def generate_dataset_card(merged_dir: Path, repo_name: str) -> str:
    """Generate dataset card content with YAML frontmatter."""

    # Get subset folders
    subset_folders = sorted([d for d in merged_dir.iterdir() if d.is_dir()])

    # Calculate statistics for each subset
    subset_stats = {}
    for folder in subset_folders:
        subset_name = folder.name
        files = list(folder.glob("*.parquet"))
        total_size = get_folder_size(folder)
        total_size_gb = total_size / (1024 ** 3)
        subset_stats[subset_name] = {
            "num_files": len(files),
            "size_gb": total_size_gb
        }

    # Generate YAML frontmatter
    yaml_configs = []
    for subset_name in subset_stats.keys():
        yaml_configs.append(f"""- config_name: {subset_name}
  data_files: "{subset_name}/*.parquet\"""")

    configs_str = "\n".join(yaml_configs)

    # Calculate total size
    total_size_gb = sum(stats["size_gb"] for stats in subset_stats.values())

    # Determine size category
    if total_size_gb < 1:
        size_category = "n<1G"
    elif total_size_gb < 10:
        size_category = "1G<n<10G"
    elif total_size_gb < 100:
        size_category = "10G<n<100G"
    else:
        size_category = "100G<n<1T"

    card = f"""---
language:
- el
license: apache-2.0
size_categories:
- {size_category}
task_categories:
- text-generation
configs:
{configs_str}
---

# {repo_name}

This dataset contains Greek language text data from multiple high-quality sources, preprocessed and tokenized with GPT-4 tokenizer.

## Dataset Structure

The dataset consists of {len(subset_stats)} subsets, each representing a different data source:

"""

    # Add subset descriptions
    for subset_name, stats in subset_stats.items():
        card += f"### {subset_name}\n\n"
        card += f"- **Files:** {stats['num_files']}\n"
        card += f"- **Size:** {stats['size_gb']:.2f} GB\n"

        # Add source descriptions
        if "finewiki" in subset_name:
            card += "- **Source:** FineWiki - High-quality Greek Wikipedia articles\n"
            card += "- **Repository:** [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)\n"
        elif "fineweb_hq" in subset_name:
            card += "- **Source:** FineWeb2-HQ - Filtered high-quality Greek web content\n"
            card += "- **Repository:** [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ)\n"
        elif "finepdfs" in subset_name:
            card += "- **Source:** FinePDFs-Edu - Educational PDF content in Greek\n"
            card += "- **Repository:** [HuggingFaceFW/finepdfs-edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu)\n"
        elif "wikipedia" in subset_name:
            card += "- **Source:** Greek Wikipedia and Wikisource\n"
            card += "- **Snapshots:** Wikipedia (20231101.el) + Wikisource (20231201.el)\n"
            card += "- **Repository:** [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)\n"

        card += "\n"

    card += f"""
## Schema

Each record in the dataset contains the following fields:

- **text** (string): The text content
- **url** (string): Source URL of the content
- **language** (string): Language code (always 'el' for Greek)
- **source** (string): Source dataset name (finewiki_el, fineweb_hq_el, finepdfs_el, or wikipedia_el)
- **token_count** (integer): Number of GPT-4 tokens in the text

## Usage

Load a specific subset:

```python
from datasets import load_dataset

# Load FineWiki Greek subset
ds = load_dataset("{repo_name}", "finewiki_el", split="train")

# Load FineWeb2-HQ Greek subset
ds = load_dataset("{repo_name}", "fineweb_hq_el", split="train")

# Access text and token count
print(ds[0]["text"])
print(ds[0]["token_count"])
```

## Tokenization

All text has been tokenized using the GPT2 tokenizer (via tiktoken library). The `token_count` field contains the number of tokens in each text, which can be used for filtering or statistics.

## Total Dataset Size

**Total size:** {total_size_gb:.2f} GB

## License

Apache 2.0 (inherits from source datasets)

## Code Repository

The code used to build this dataset is available on GitHub:
[https://github.com/alexliap/high-quality-gr-text](https://github.com/alexliap/high-quality-gr-text)

## Citation

If you use this dataset, please cite the original sources:

- FineWiki: HuggingFaceFW/finewiki
- FineWeb2-HQ: epfml/FineWeb2-HQ
- FinePDFs-Edu: HuggingFaceFW/finepdfs-edu
- Wikipedia: Wikimedia Foundation
"""

    return card


def main():
    parser = argparse.ArgumentParser(
        description="Upload merged Greek datasets to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --username alexliap --token hf_xxxxx --repo-name greek-pretraining-data
  %(prog)s -u alexliap -t hf_xxxxx --repo-name my-greek-data --public
        """,
    )

    parser.add_argument(
        "-u", "--username",
        required=True,
        help="Your Hugging Face username"
    )

    parser.add_argument(
        "-t", "--token",
        required=True,
        help="Your Hugging Face token (from https://huggingface.co/settings/tokens)",
    )

    parser.add_argument(
        "--repo-name",
        type=str,
        default="greek-pretraining-data",
        help="Repository name for the dataset. Default: greek-pretraining-data",
    )

    parser.add_argument(
        "--merged-dir",
        type=str,
        default="data/merged",
        help="Path to merged dataset directory. Default: data/merged",
    )

    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public (default: private)",
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
        parser.error(f"No subset folders found in {args.merged_dir}")

    # Initialize the API with the provided token
    api = HfApi(token=args.token)

    repo_id = f"{args.username}/{args.repo_name}"

    print(f"Uploading dataset to {repo_id}...")
    print(f"Merged directory: {merged_dir}")
    print(f"Subsets: {[f.name for f in subset_folders]}")
    print(f"Visibility: {'Public' if args.public else 'Private'}")
    print()

    try:
        # Create the dataset repository
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True
        )
        print(f"✓ Repository created/verified: https://huggingface.co/datasets/{repo_id}\n")

        # Generate and upload dataset card
        print("Generating dataset card...")
        dataset_card = generate_dataset_card(merged_dir, repo_id)

        # Write to temporary file
        readme_path = merged_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(dataset_card)

        print(f"Uploading dataset card...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("✓ Dataset card uploaded\n")

        # Upload each subset folder
        for subset_folder in subset_folders:
            subset_name = subset_folder.name

            folder_size_gb = get_folder_size(subset_folder) / (1024**3)
            num_files = len(list(subset_folder.glob("*.parquet")))

            print(f"Uploading {subset_name}/ folder ({num_files} file(s), {folder_size_gb:.2f} GB) ...")

            # Upload the entire subset folder
            api.upload_folder(
                folder_path=str(subset_folder),
                path_in_repo=subset_name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"✓ {subset_name}/ folder uploaded\n")

        print("=" * 60)
        print("✓ Upload complete!")
        print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during upload: {e}")
        raise


if __name__ == "__main__":
    main()
