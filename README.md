# Detecting text-reading hallucinations in VLM descriptions of Cyrillic text for blind users

Manuscript for *Universal Access in the Information Society* (Springer). Built on the
official Springer Nature LaTeX template (`sn-jnl`, numbered references).

## Question

When a vision-language model describes a photograph of Ukrainian signage, packaging or a
document for a blind user, how often does it assert text that is not printed (or not
visible) there, and can those hallucinations be detected at run time?

## Design

* 567 synthetic, photograph-like stimuli: genre (signage / packaging / document) ×
  quality (clean / blur / severe) × framing (full / crop / finger-or-glare occlusion) ×
  name slot (real word / invented / near-word) × 7 replicates.
* Exact ground truth for every character, its slot category and its **visibility** in the
  final frame (visible / partial / hidden), so completions of unseen text can be separated
  from misreadings of visible text.
* Models: Claude Haiku 5.5, Sonnet 5.5, Opus 5.5 (all images), Fable 5.1 (162-image
  subset), queried through the Claude Code CLI with a blind-user *description* prompt and
  a constrained *transcription* prompt.
* Detectors: re-reading consistency, cross-model agreement, Tesseract 5 OCR, lexicon,
  hedging cues, and a cross-validated logistic combination.

## Pipeline

```
scripts/00_fetch_assets.sh   fonts, Tesseract AppImage + tessdata_best, Ukrainian word list
scripts/01_make_stimuli.py   images (data/images) + ground truth (data/stimuli.jsonl)
scripts/01b_retokenise.py    recompute GT tokens after a tokeniser change (no re-render)
scripts/02_run_vlm.py        resumable model sweep -> runs/vlm/*.jsonl (git-ignored)
scripts/run_sweeps.sh        the full sweep as parallel chains
scripts/03_ocr.py            Tesseract readings -> runs/ocr.jsonl
scripts/04_score.py          claim extraction, labelling, detector features -> runs/claims.jsonl
scripts/05_analyse.py        every reported number -> results/summary.json
scripts/06_figures.py        paper/figures/*.pdf
scripts/07_audit_sample.py   sample for the manual audit of the labeller (results/audit.json)
scripts/08_build_paper.py    assembles paper/universal-access-in-2026.tex from the template
                             and paper/body_*.tex, defines number macros from summary.json,
                             compiles with tectonic
src/textgen.py, src/render.py, src/textnorm.py, src/scoring.py
```

Every reported number is a LaTeX macro computed from `results/summary.json` (and
`results/audit.json`) by `scripts/08_build_paper.py`; tables are generated the same way.

## Reproducing

```bash
bash scripts/00_fetch_assets.sh
python3 scripts/01_make_stimuli.py --reps 7
scripts/run_sweeps.sh            # needs an authenticated `claude` CLI; about $83 at list prices (as run)
python3 scripts/03_ocr.py
python3 scripts/04_score.py && python3 scripts/05_analyse.py && python3 scripts/06_figures.py
python3 scripts/08_build_paper.py
```

Raw model responses are not committed (`runs/` is git-ignored); only response-free
aggregates are. Fonts are OFL/UFL (google/fonts); the word list is CC BY-SA 4.0
(hermitdave/FrequencyWords, OpenSubtitles 2018).
