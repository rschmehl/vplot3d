#!/bin/bash
# converts Latex code in original document file.svg into outline font in new document file_tex.svg

SCRIPT_DIR=$(dirname "$0")

fname=$( basename -s _tex.svg $1 )

# export svg to pdf and pdf_tex files
inkscape --export-type=pdf --export-filename=$fname.pdf --export-latex --export-area-page $1

# compile with pdflatex
export TEXINPUTS=$SCRIPT_DIR//:
if test -f "$fname.tex"; then
  # use a custom Latex wrapper (e.g. to scale font)
  pdflatex --interaction=batchmode -jobname=tmp "\def\filename{$fname.pdf_tex}\input{$fname.tex}" 2>&1 > /dev/null
else
  # Standard Latex wrapper
  pdflatex --interaction=batchmode -jobname=tmp "\def\filename{$fname.pdf_tex}\input{$SCRIPT_DIR/template.tex}" 2>&1 > /dev/null
fi

# Convert generated pdf back to svg
inkscape tmp.pdf --pdf-poppler --export-type=svg --export-filename=tmp.svg &>/dev/null
scour -i tmp.svg -o $fname'.svg'

# clean up
rm -f tmp.* $fname.pdf $fname.pdf_tex

inkscape --export-type=png --export-filename=$fname'.png' $fname'.svg'
