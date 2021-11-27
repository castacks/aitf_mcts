#!/bin/sh

. /jet/home/jaypat/.bashrc

set -x

conda activate pytorch

python play.py > log.txt

