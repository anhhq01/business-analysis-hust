#!/bin/bash
# ---------------------------------------------------------------------------
# Downloads the Kaggle PaySim base dataset into data/raw/ under the exact
# filename config.py expects (online_fraud_detection.csv).
#
# The public API endpoint below currently serves the file WITHOUT a Kaggle
# login. If that ever stops working, use the fallbacks noted at the bottom.
# ---------------------------------------------------------------------------
set -euo pipefail

RAW_DIR="data/raw"
TARGET="$RAW_DIR/online_fraud_detection.csv"
URL="https://www.kaggle.com/api/v1/datasets/download/rupakroy/online-payments-fraud-detection-dataset"

mkdir -p "$RAW_DIR"

if [ -f "$TARGET" ]; then
  echo "Already present: $TARGET"
  exit 0
fi

echo "Downloading PaySim dataset..."
curl -L -o "$RAW_DIR/paysim.zip" "$URL"

echo "Unzipping..."
unzip -o "$RAW_DIR/paysim.zip" -d "$RAW_DIR"

# Kaggle ships it as PS_..._log.csv; rename to the config.py filename.
mv "$RAW_DIR"/PS_*_log.csv "$TARGET"
rm -f "$RAW_DIR/paysim.zip"

echo "Done -> $TARGET"
echo "Rows: $(( $(wc -l < "$TARGET") - 1 ))"

# ---------------------------------------------------------------------------
# FALLBACKS if the direct URL stops working:
#   1) Manual: download from
#      https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
#      unzip, and place/rename the CSV as data/raw/online_fraud_detection.csv
#   2) Kaggle CLI (needs ~/.kaggle/kaggle.json API token):
#      kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset -p data/raw
#      then unzip + rename as above.
# ---------------------------------------------------------------------------