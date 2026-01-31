# Greek Dataset Builder

Project for creating the HF dataset 
[high-quality-gr-text](https://huggingface.co/datasets/alexliap/high-quality-gr-text). The plan is to enrich it with additional curated Greek datasets in the future if any come out.

## Overview

This repository contains Python scripts to:
1. Download Greek language datasets from HuggingFace Hub
2. Merge Greek language parquet datasets from multiple sources
3. Add GPT-4 tokenization (token counts) to text data
4. Split merged files at 5GB threshold for optimal HuggingFace hosting
5. Upload datasets to HuggingFace Hub with multi-subset configuration
6. Verify merged datasets for quality assurance

## Features

- **Lazy evaluation**: Uses Polars lazy frames for memory-efficient processing of large datasets
- **GPT-4 tokenization**: Adds token count column using tiktoken (cl100k_base encoding)
- **Automatic file splitting**: Splits datasets into ~5GB chunks for HuggingFace compatibility
- **Schema standardization**: Normalizes different source schemas to consistent format
- **Multi-subset support**: Creates HuggingFace dataset with separate subsets per source
- **Comprehensive verification**: Validates schemas, file sizes, token statistics, and data quality

## Usage

### 0. Download Source Datasets (Optional)

If you don't have the source datasets yet, download them using the provided scripts.

**Option 1: Download all at once (recommended)**
```bash
./download_data.sh
```

**Option 2: Download individually**
```bash
# Download FineWiki Greek data
python cli/get_finewiki.py

# Download FineWeb2-HQ Greek data
python cli/get_fineweb_hq.py

# Download FineWeb-Edu Greek data (if needed)
python cli/get_fineweb_edu.py

# Download FinePDFs-Edu Greek data
python cli/get_finepdfs.py

# Download Wikipedia Greek data
python cli/get_wikipedia.py
```

These scripts will download the respective datasets to the `data/` directory with the following structure:
```
data/
├── finewiki/elwiki/parquet/*.parquet
├── fineweb_hq/gr/parquet/*.parquet
├── fineweb_edu/gr/parquet/*.parquet
├── finepdfs_edu/gr/parquet/*.parquet
└── wikipedia/gr/parquet/*.parquet
```

**Note:** You can also manually download or prepare your own datasets following this directory structure.

### 1. Merge Greek Datasets

Merge parquet files from multiple sources with tokenization:

```bash
python merge_greek_datasets.py --max-size 5GB --output-dir data/merged
```

**Arguments:**
- `--max-size`: Maximum size per output file (default: 5GB)
- `--output-dir`: Output directory for merged files (default: data/merged)

**Expected input structure:**
```
data/
├── finewiki/elwiki/parquet/*.parquet
├── fineweb_hq/gr/parquet/*.parquet
├── finepdfs_edu/gr/parquet/*.parquet
└── wikipedia/gr/parquet/*.parquet
```

**Output structure:**
```
data/merged/
├── finewiki_el/
│   └── data_00001.parquet
├── fineweb_hq_el/
│   ├── data_00001.parquet
│   └── data_00002.parquet
├── finepdfs_el/
│   └── data_00001.parquet
└── wikipedia_el/
    └── data_00001.parquet
```

**Output schema:**
- `text` (string): The text content
- `url` (string): Source URL
- `language` (string): Language code (always 'el' for Greek)
- `source` (string): Source dataset name
- `token_count` (integer): Number of GPT-4 tokens in the text

### 2. Verify Merged Datasets (Optional)

Run validation checks on merged data:

```bash
python verify_merged_datasets.py --merged-dir data/merged --stats-output merge_stats.json
```

**Arguments:**
- `--merged-dir`: Path to merged dataset directory (default: data/merged)
- `--stats-output`: Path to output statistics JSON (default: merge_stats.json)
- `--max-size-gb`: Maximum file size in GB (default: 5.0)

**Verification checks:**
- Row count and file size statistics
- Schema consistency (all expected columns present)
- Token count validation (type and statistics)
- Null value detection
- File size compliance (<5GB per file)

### 3. Upload to HuggingFace

Upload merged datasets to HuggingFace Hub:

```bash
python upload_greek_datasets.py \
  --username YOUR_USERNAME \
  --token YOUR_HF_TOKEN \
  --repo-name greek-pretraining-data \
  --merged-dir data/merged
```

**Arguments:**
- `-u, --username`: Your HuggingFace username (required)
- `-t, --token`: Your HuggingFace token (required, get from https://huggingface.co/settings/tokens)
- `--repo-name`: Repository name (default: greek-pretraining-data)
- `--merged-dir`: Path to merged dataset directory (default: data/merged)
- `--public`: Make dataset public (default: private)

**Environment variable alternative:**
```bash
export HF_TOKEN="hf_xxxxx"
python upload_greek_datasets.py --username YOUR_USERNAME --token "$HF_TOKEN" --repo-name greek-pretraining-data
```

## Dataset Loading

After uploading, load specific subsets from HuggingFace:

```python
from datasets import load_dataset

# Load FineWiki Greek subset
ds = load_dataset("alexliap/high-quality-gr-text", "finewiki_el", split="train")

# Load FineWeb2-HQ Greek subset
ds = load_dataset("alexliap/high-quality-gr-text", "fineweb_hq_el", split="train")

# Access data
print(ds[0]["text"])
print(ds[0]["token_count"])
```

## Source Datasets

This tool is designed to work with Greek language data from:
- **FineWiki**: High-quality Greek Wikipedia articles
- **FineWeb2-HQ**: Filtered high-quality Greek web content
- **FinePDFs-Edu**: Educational PDF content in Greek
- **Wikipedia**: Greek Wikipedia and Wikisource
