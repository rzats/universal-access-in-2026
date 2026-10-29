# Detecting text-reading hallucinations in VLM descriptions of Cyrillic text for blind users

Manuscript for *Universal Access in the Information Society* (Springer), built on the
official Springer Nature LaTeX template (`sn-jnl`, numbered references).

## Question

When a vision-language model describes a photograph of Ukrainian signage, packaging or a
document for a blind user, how often does it assert text that is not printed (or not
visible) there, and can those hallucinations be detected at run time? All evaluated models
are **four Claude models of one vendor** (Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.1);
the findings are scoped to that model family.

## Studies

* **Study 1, synthetic factorial** (`data/`, `runs/`, `results/`): 567 rendered stimuli,
  genre × quality × framing × name-slot (real word / invented / near-word) × 7 replicates,
  with exact character-level visibility ground truth.
* **Study 2, real images** (`data/real/`, `runs/real/`, `results/real/`): 176 images.
  - Signage is **excluded**: 70 Wikimedia Commons sign photographs were collected and transcribed
    (`data/real/sign_transcriptions.jsonl`), but the transcription was produced with a Claude model
    (the agent that ran this project), i.e. a member of the evaluated family, so it is not an
    independent ground truth (`INCLUDE_SIGNAGE = False` in `scripts/11_make_real_stimuli.py`).
  - 108 Open Food Facts packaging photographs; ground truth = Google Cloud Vision OCR stored by
    OFF, kept only when it agrees with the contributor's independent transcription / product name.
  - 68 real Ukrainian documents (PDF text layer, checked against Tesseract); sources and licence
    notes in `data/raw_docs/manifest.jsonl` (PDFs not redistributed).
  - Licences of every image are recorded per stimulus (`attribution` field of `data/real/stimuli.jsonl`).
* **Near-word probe** (`data/probe/`): 64 pairs of real document pages, one word edited to a
  one-letter near-word, identical degradation.

Raw model responses and stimulus images derived from third-party material are not committed
(`runs/`, `data/*/images` are git-ignored); everything is regenerable from the scripts.

## Pipeline

```
scripts/00_fetch_assets.sh     fonts, Tesseract AppImage + tessdata_best, Ukrainian word list
scripts/01_make_stimuli.py     Study 1 images + ground truth          (01b_retokenise.py: re-tokenise GT)
scripts/10_harvest_off.py      Open Food Facts packaging photographs + OCR (rate-limited, resumable)
scripts/12_collect_docs.py     copies the collected document PDFs + manifest
scripts/11_make_real_stimuli.py Study 2 images + ground truth (UAIS_SET=real)
scripts/13_make_lure_probe.py  near-word probe pairs (UAIS_SET=probe)
scripts/02_run_vlm.py          resumable model runs through the Claude Code CLI (--tag for repeats)
scripts/run_sweeps.sh, run_sweeps_real.sh, run_probe.sh, run_revision.sh   the sweeps as run
scripts/03_ocr.py              Tesseract readings
scripts/04_score.py            claim extraction, labelling, detector features
scripts/05_analyse.py          rates, factors, detection  -> results[/real]/summary.json
scripts/14_probe_analysis.py   near-word probe            -> results/probe.json
scripts/15_audit_raters.py     automated two-rater audit (Claude Sonnet 5.5 + gemma3:12b via Ollama)
scripts/16_revision_stats.py   paired tests (McNemar, cluster bootstrap, Holm), stability,
                               cost/latency, synthetic-vs-real comparison -> results/revision.json
scripts/06_figures.py, 17_failure_figure.py   figures
scripts/08_build_paper.py      assembles paper/universal-access-in-2026.tex from the template and
                               paper/body_*.tex; every number is a macro computed from results/*.json
```

Select the dataset with `UAIS_SET=synthetic|real|probe`.

## Reproducing (order)

```bash
bash scripts/00_fetch_assets.sh
python3 scripts/01_make_stimuli.py --reps 7
python3 scripts/10_harvest_off.py 200; python3 scripts/12_collect_docs.py <dir with PDFs>
python3 scripts/11_make_real_stimuli.py; python3 scripts/13_make_lure_probe.py
scripts/run_sweeps.sh; scripts/run_sweeps_real.sh; scripts/run_probe.sh; scripts/run_revision.sh
for s in synthetic real; do UAIS_SET=$s python3 scripts/03_ocr.py; UAIS_SET=$s python3 scripts/04_score.py; \
  UAIS_SET=$s python3 scripts/05_analyse.py; UAIS_SET=$s python3 scripts/06_figures.py; done
UAIS_SET=probe python3 scripts/04_score.py; python3 scripts/14_probe_analysis.py
python3 scripts/15_audit_raters.py sample synthetic; python3 scripts/15_audit_raters.py sample real
python3 scripts/15_audit_raters.py claude; python3 scripts/15_audit_raters.py gemma; python3 scripts/15_audit_raters.py analyse
python3 scripts/16_revision_stats.py; python3 scripts/17_failure_figure.py; python3 scripts/08_build_paper.py
```

Fonts are OFL/UFL (google/fonts); the word list is CC BY-SA 4.0 (hermitdave/FrequencyWords,
OpenSubtitles 2018); Open Food Facts photographs CC BY-SA 3.0; Commons photographs under
the licences recorded per image.
