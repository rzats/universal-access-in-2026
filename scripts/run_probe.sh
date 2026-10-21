#!/usr/bin/env bash
# Near-word probe sweep: description prompt, all four models (UAIS_SET=probe).
set -u
cd "$(dirname "$0")/.."
export UAIS_SET=probe
mkdir -p runs/probe
( python3 scripts/02_run_vlm.py --model haiku --prompt desc --workers 3 --budget 1000; python3 scripts/02_run_vlm.py --model opus --prompt desc --workers 3 --budget 1000 ) > runs/probe/chainA.log 2>&1 &
( python3 scripts/02_run_vlm.py --model sonnet --prompt desc --workers 3 --budget 1000; python3 scripts/02_run_vlm.py --model fable --prompt desc --workers 3 --budget 1000 ) > runs/probe/chainB.log 2>&1 &
wait
