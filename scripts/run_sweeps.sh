#!/usr/bin/env bash
# Full model sweep as two parallel chains; every step is resumable, and no
# (model, prompt) pair appears in both chains.
set -u
cd "$(dirname "$0")/.."
python3 -c "import json;print('\n'.join(json.loads(l)['id'] for l in open('data/stimuli.jsonl') if json.loads(l)['rep']<2))" > runs/fable_subset.txt
run() { python3 scripts/02_run_vlm.py --model ${1%%:*} --prompt ${1##*:} --workers 4 --budget 85 ${2:-}; }
( for mp in haiku:desc haiku:trans opus:desc opus:trans; do run $mp; done ) > runs_chainA.log 2>&1 &
( for mp in sonnet:desc sonnet:trans; do run $mp; done; run fable:desc "--subset runs/fable_subset.txt" ) > runs_chainB.log 2>&1 &
wait
