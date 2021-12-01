#!/bin/sh

. /jet/home/jaypat/.bashrc

set -x

conda activate pytorch

python play.py --model_weights goalGAIL1b_60.pt > log.txt

