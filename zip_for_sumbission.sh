#!/bin/bash

# Script for rendering the project report to a pdf and preparing a zip file for
# submission.

quarto render report.qmd --to pdf

mkdir HPDM098_assessment2

cp report.pdf HPDM098_assessment2/
cp -r team_portfolio HPDM098_assessment2/
cp -r technical_appendix HPDM098_assessment2/
cp binder/environment.yml HPDM098_assessment2/technical_appendix/

zip -r HPDM098_assessment2.zip HPDM098_assessment2 -x '.*' -x '__MACOSX'

mv HPDM098_assessment2.zip ../
rm -r HPDM098_assessment2
