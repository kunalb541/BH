#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p outputs/data outputs/figures outputs/tables outputs/logs

# Run BH simulation package
python bh.py

# Compile manuscript (pdflatex + bibtex + two more passes)
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex

# Archive LaTeX auxiliary files
shopt -s nullglob
for f in *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg *.nav *.snm; do
  mv "$f" outputs/logs/ 2>/dev/null || true
done
shopt -u nullglob

# Move compiled PDF to root
mv paper.pdf ./paper.pdf 2>/dev/null || true

echo
echo "Done."
ls -1
echo "Figures:"; ls -1 outputs/figures 2>/dev/null || true
echo "Tables:"; ls -1 outputs/tables 2>/dev/null || true
