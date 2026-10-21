#!/usr/bin/env bash
# Full model sweep, sequential per (model, prompt); every step is resumable.
set -u
cd "$(dirname "$0")/.."
for mp in haiku:desc sonnet:desc opus:desc haiku:trans sonnet:trans opus:trans; do
  python3 scripts/02_run_vlm.py --model ${mp%%:*} --prompt ${mp##*:} --workers 5 --budget 85
done
# the most expensive model on a two-replicate subset (162 images)
python3 -c "import json;print('\n'.join(json.loads(l)['id'] for l in open('data/stimuli.jsonl') if json.loads(l)['rep']<2))" > runs/fable_subset.txt
python3 scripts/02_run_vlm.py --model fable --prompt desc --workers 4 --subset runs/fable_subset.txt --budget 85
