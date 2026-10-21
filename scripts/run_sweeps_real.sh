#!/usr/bin/env bash
# Model sweep on the real-image benchmark (UAIS_SET=real). Optional $1: file of stimulus ids.
set -u
cd "$(dirname "$0")/.."
export UAIS_SET=real
SUB=${1:+--subset $1}
run() { python3 scripts/02_run_vlm.py --model ${1%%:*} --prompt ${1##*:} --workers 3 --budget 1000 $SUB; }
( for mp in haiku:desc haiku:trans opus:desc; do run $mp; done ) > runs/real/chainA.log 2>&1 &
( for mp in sonnet:desc sonnet:trans; do run $mp; done ) > runs/real/chainB.log 2>&1 &
( for mp in opus:trans fable:desc; do run $mp; done ) > runs/real/chainC.log 2>&1 &
wait
