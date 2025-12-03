#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
echo '>>>>>>>>>>>>>Now start the example of HEAD.'
cd ./code/HEAD
bash run_example.sh
cd ../..
echo '>>>>>>>>>>>>>Now start the example of TED.'
cd ./code/TED
bash run_example.sh
cd ../..
echo '>>>>>>>>>>>>>Now start the example of ScreeningPipeline'
cd ./code/ScreeningPipeline
bash reproduce_ScreeningPipeline.sh