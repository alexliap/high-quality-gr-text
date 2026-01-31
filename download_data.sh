#!/bin/bash
# Download all datasets in parquet format (one output file per source parquet file)

source .venv/bin/activate

echo "=================================================="
echo "Downloading all datasets"
echo "Output format: parquet"
echo "Note: Each source parquet file saved separately"
echo "=================================================="

echo ""
echo "1/5 Downloading FineWiki (Greek & English) ..."
python cli/get_finewiki.py -w "parquet"

echo ""
echo "Waiting 300s before next dataset ..."
sleep 300

echo ""
echo "2/5 Downloading FineWeb2-HQ (Greek) ..."
python cli/get_fineweb_hq.py -w "parquet"

echo ""
echo "Waiting 300s before next dataset ..."
sleep 300

echo ""
echo "3/5 Downloading FineWeb-EDU (English) ..."
python cli/get_fineweb_edu.py -w "parquet"

echo ""
echo "Waiting 300s before next dataset ..."
sleep 300

echo ""
echo "4/5 Downloading FinepdfsEdu (Greek) ..."
python cli/get_finepdfs.py -w "parquet"

echo ""
echo "Waiting 300s before next dataset ..."
sleep 300

echo ""
echo "5/5 Downloading Wikipedia & Wikisource (Greek) ..."
python cli/get_wikipedia.py -w "parquet"

echo ""
echo "=================================================="
echo "All downloads completed!"
echo "=================================================="
