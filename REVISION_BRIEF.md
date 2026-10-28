# Revision brief: UAIS manuscript (paper #4)

Two of our other papers were just rejected. Reviewers rated the novelty good and rejected
on evaluation rigour. Revise this manuscript so that it survives the same reviewers. The
reviewer checklist in this prompt is the standard. Everything stays automated: there is no
human annotation, no human photography and no user study.

## 1. Fix the audit attribution (do first)
`results/audit.json` and Section 4.4 say the labeller audit was "judged by the first
author". Neither author performed it. Redo the audit as described in item 4 and describe
it as automated, naming the models. Search the whole manuscript and README for any other
work credited to the authors that they did not do.

## 2. Scope the claims to one vendor
The authors decided not to run non-Claude models in this revision. Do not add them.
Instead, make the single-vendor design explicit everywhere a reviewer would look for it:
- In the abstract, the contributions, the comparison claims and the conclusion, say "four
  Claude models" or "models of one vendor", not "VLMs" in general. Recast "capability is no
  remedy" as a finding within one model family.
- In the limitations, keep the point that the cross-model detector may be optimistic
  because the peers share a vendor. Name a multi-vendor replication as future work.

## 3. Repeated runs
Rerun the description prompt twice more for one image per design cell (81 images), for
every Claude model. Report run-to-run stability (image-level outcome flip rate, and the spread of
the token rate). Add sample-to-sample disagreement as a detector signal on that subset.

## 4. Automated labeller audit with two raters
Draw a stratified sample of at least 400 description tokens across all models and labels.
Two independent automated raters judge whether each label is right. Each rater sees the
image crop around the token, the printed line, the quoted span and the label. Use one
Claude model and one open model, `gemma3:12b`. The open model is served from the author's
Mac at `http://127.0.0.1:11434` (Ollama `/api/chat` with base64 images). Run
`curl -s 127.0.0.1:11434/api/tags` to check it is up. Send one request at a time, keep the
output resumable, and record the Ollama version and decoding settings. This is about 400
short calls, roughly 1–2 h. If the Mac stays unreachable, finish with the Claude rater
alone and list the second rater as open. Report each rater's labeller
precision per label with Wilson CIs, Cohen's kappa between the raters, and how
disagreements were resolved (deterministic rule, stated). Describe this as agreement
between automated raters, not as human validation. Put the rater prompt in the appendix.

## 5. Real images
A real-image benchmark is already half built (commit 0ad16f9 plus uncommitted changes:
Open Food Facts packaging, Wikimedia Commons signage, real documents, a near-word probe;
`results/real/summary.json` covers 138 images for the Claude models). Finish it:
- State exactly where each ground-truth transcription comes from. It must not come from
  any evaluated model. Drop items whose ground truth cannot be sourced independently.
- Record the licence of every image.
- Complete the runs of all four Claude models on the set (Opus and Fable cover 129 of 138
  images; transcription for Fable is not needed). Report the main RQ1/RQ2 numbers on real images next to the
  synthetic ones.
- Add a synthetic-vs-real comparison: token-length and slot-category distributions, plus
  hallucination rates side by side. Quantify how far the ї/є over-representation in the
  invented names departs from natural Ukrainian letter frequencies.
- Rewrite the "Synthetic stimuli" limitation to match.

## 6. Statistics for every comparison
Every model-vs-model or condition-vs-condition statement in the prose needs a paired
test on shared images: exact McNemar for image-level outcomes, and a paired cluster
bootstrap CI on the difference for token rates. Apply a Holm correction across the model
pairs.

## 7. Reproducibility appendix
Give every prompt verbatim: system, description, transcription and both audit raters.
Add model ids and versions, the CLI version, the Ollama version for the audit rater,
decoding settings and the run dates.

## 8. Cost and latency
Add a table per model: input and output tokens per image, list-price $ per image for the
Claude models, and seconds per image. All of them were measured through the same CLI path,
so say so and do not compare against the local audit rater. Back the
"re-reading costs little" recommendation with these numbers, or soften it.

## 9. Presentation
- Add a failure figure with 3–4 real cases from the runs, each showing the image crop,
  the printed text and the model's claim: a lexicalised near-word, a Ukrainian-letter
  loss, a completion of cropped text, and a digit error. Give each a one-line cause.
- Add an appendix table with the per-genre and per-condition breakdown, so nothing is
  only pooled.
- Reread every citation against the sentence it supports. Prefer published versions to
  arXiv preprints wherever they exist.
- Update the abstract, the contributions and the conclusion to the new results. Keep the
  title unless it now claims more than one vendor's models show.

## Done means
The PDF is rebuilt and checked page by page, the reviewer checklist is applied, and every
new number comes from committed code. The final summary lists anything still open, with
the reason.
