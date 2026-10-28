#!/usr/bin/env bash
# Revision sweeps: complete the real-image benchmark, the near-word probe, and two repeat
# runs of the description prompt on one synthetic image per design cell. All resumable.
set -u
cd "$(dirname "$0")/.."
R() { python3 scripts/02_run_vlm.py --workers 3 --budget 1000 "$@"; }
(
  export UAIS_SET=real
  for mp in haiku:desc haiku:trans sonnet:desc sonnet:trans opus:desc opus:trans fable:desc; do R --model ${mp%%:*} --prompt ${mp##*:}; done
) > runs/rev_real.log 2>&1 &
(
  export UAIS_SET=probe
  for m in sonnet opus fable haiku; do R --model $m --prompt desc; done
  export UAIS_SET=synthetic
  for t in r2 r3; do for m in haiku sonnet opus fable; do R --model $m --prompt desc --tag $t --subset runs/repeat_subset.txt; done; done
) > runs/rev_probe_rep.log 2>&1 &
( UAIS_SET=real python3 scripts/03_ocr.py ) > runs/rev_ocr.log 2>&1 &
wait
