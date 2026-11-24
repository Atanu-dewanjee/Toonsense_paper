#!/usr/bin/env bash
# Full ToonSense pipeline for one dataset. Each step caches its outputs, so re-running resumes where it stopped.
#   bash run_all.sh [cmban|banglaabusememe|memosen]        (default: cmban)
# After all datasets are done, scripts/10_cross_dataset.py builds the cross-dataset paper tables.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export TOONSENSE_DATASET="${1:-cmban}"
L="logs/${TOONSENSE_DATASET}"
mkdir -p "$L"

python -m pytest -q tests                                   | tee "$L/00_tests.log"
python -W ignore scripts/01_ocr.py                           | tee "$L/01_ocr.log"
python -W ignore scripts/02_extract_features.py              | tee "$L/02_features.log"
python -W ignore scripts/03_benchmark.py                     | tee "$L/03_benchmark.log"
python -W ignore scripts/04_robustness.py                    | tee "$L/04_robustness.log"
python -W ignore scripts/05_explain.py                       | tee "$L/05_explain.log"
# CMBan also has a fine label level. Error-analysis tables are not suffixed by granularity, so the fine level runs
# first and the primary (coarse) level overwrites it.
if [ "$TOONSENSE_DATASET" = "cmban" ]; then
  python -W ignore scripts/06_error_analysis.py --granularity fine | tee "$L/06_error_analysis_fine.log"
  python -W ignore scripts/07_report.py --granularity fine     | tee "$L/07_report_fine.log"
fi
python -W ignore scripts/06_error_analysis.py                | tee "$L/06_error_analysis.log"
python -W ignore scripts/07_report.py                        | tee "$L/07_report.log"
python -W ignore scripts/11_augmentation.py                  | tee "$L/11_augmentation.log"
