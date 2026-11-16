#!/usr/bin/env bash
# Regenerate tables/figures from results/ and compile report/final_report.pdf (needs tectonic).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python3 || command -v python)}
"$PY" -m src.report >/dev/null
"$PY" -m src.report_toyota >/dev/null
"$PY" -m src.figures
mkdir -p report/tables
cp results/ntu_table.tex results/natural_table.tex results/toyota_table.tex results/transfer_table.tex report/tables/
cd report && tectonic -X compile final_report.tex
