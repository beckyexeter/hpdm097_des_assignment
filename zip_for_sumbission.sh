#!/bin/bash

quarto render report.qmd --to pdf

mkdir submission_directory

cp report.pdf submission_directory/
cp -r team_portfolio submission_directory/
cp -r technical_appendix submission_directory/
cp binder/environment.yml submission_directory/technical_appendix/

zip -r HPDM098_assessment2.zip submission_directory -x '.*' -x '__MACOSX'

