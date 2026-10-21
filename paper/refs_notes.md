# Reference candidates: verification notes

Every entry in refs_candidates.bib was checked against api.crossref.org/works/<DOI> (metadata taken from Crossref JSON) or export.arxiv.org/api/query?id_list=<id> on 2026-10-21. Format: key | how verified | what it supports.

## 1. BLV users and AI visual assistance
- gurari2018vizwiz | Crossref DOI 10.1109/CVPR.2018.00380 | VizWiz VQA dataset: real photos taken by blind people are often blurry, poorly framed or occluded, and text reading is a frequent request (supports the choice of blur, crop and occlusion degradations).
- gurari2020captioning | Crossref DOI 10.1007/978-3-030-58520-4_25 | VizWiz-Captions (ECCV 2020): captioning images taken by blind people; many contain text and have quality problems.
- bigham2010vizwiz | Crossref DOI 10.1145/1866029.1866080 | Original VizWiz (UIST 2010): blind users send photos with questions, and many of the questions are about reading labels and text.
- brady2013visual | Crossref DOI 10.1145/2470654.2481291 | Large-scale analysis of blind people's everyday visual questions; reading text and identifying products are dominant needs.
- macleod2017understanding | Crossref DOI 10.1145/3025453.3025814 | Blind users tend to trust incorrect computer-generated captions, even when they don't match the context (motivates detecting hallucinations).
- stangl2020person | Crossref DOI 10.1145/3313831.3376404 | What BLV people want in image descriptions, including transcription of visible text, varies by context.
- gonzalezpenuela2024investigating | Crossref DOI 10.1145/3613904.3642211 | Diary study of AI scene-description apps (e.g., Be My AI) used by BLV people: use cases, how well the descriptions met their needs, and trust.
- xie2025beyond | Crossref DOI 10.1145/3706598.3714210 | CHI 2025 study of visually impaired users interacting with LMM assistants (Be My AI) on smartphones, including errors and verification practices. NOTE: arXiv 2407.08882 ("Emerging Practices for Large Multimodal Model (LMM) Assistance for People with Visual Impairments", same authors, verified on arXiv) looks like an earlier version, so it was not added as a separate entry.
- alharbi2024misfitting | Crossref DOI 10.1145/3663548.3675659 | How blind people verify and contest AI errors (ASSETS 2024); supports the claim that BLV users have few ways to verify outputs independently.
- adnin2024king | Crossref DOI 10.1145/3663548.3675631 | How blind people use and understand generative AI tools, including how far they trust them (ASSETS 2024).
- hong2024understanding | Crossref DOI 10.1145/3663548.3675635 | Strategies and challenges blind users have when handling object-recognition errors (ASSETS 2024).
- granquist2021evaluation | Crossref DOI 10.1177/0145482X211027492 | Evaluation of text reading by Seeing AI and OrCam MyEye (J. Visual Impairment & Blindness); text-reading apps as an established BLV tool.
- khan2020insight | Crossref DOI 10.1007/s10209-020-00733-8 | UAIS review of smartphone-based assistive solutions for blind and visually impaired people, including text/OCR readers (UAIS journal context).

## 2. Hallucination in multimodal models
- bai2024hallucination | arXiv 2404.18930 | Survey of hallucination in MLLMs: taxonomy, benchmarks and causes, including language priors.
- liu2024survey | arXiv 2402.00253 | Survey of hallucination in large vision-language models.
- rohrbach2018object | Crossref DOI 10.18653/v1/D18-1437 | CHAIR metric; captioners hallucinate objects because of language priors.
- li2023evaluating | Crossref DOI 10.18653/v1/2023.emnlp-main.20 | POPE: object hallucination in LVLMs, measured by polling yes/no questions.
- guan2024hallusionbench | Crossref DOI 10.1109/CVPR52733.2024.01363 | HallusionBench: language hallucination and visual illusion in LVLMs (CVPR 2024).

## 3. Text reading / OCR in VLMs
- singh2019towards | Crossref DOI 10.1109/CVPR.2019.00851 | TextVQA: VQA that requires reading scene text.
- biten2019scene | Crossref DOI 10.1109/ICCV.2019.00439 | ST-VQA scene-text VQA benchmark.
- sidorov2020textcaps | Crossref DOI 10.1007/978-3-030-58536-5_44 | TextCaps: captions that must include text read from the image.
- liu2024ocrbench | Crossref DOI 10.1007/s11432-024-4235-6 | OCRBench: LMMs are weak at multilingual and handwritten text and lean on semantics/language priors over the visual text (e.g., non-semantic strings).
- tang2025mtvqa | Crossref DOI 10.18653/v1/2025.findings-acl.404 | MTVQA: multilingual text-centric VQA, including Russian (Cyrillic); MLLMs perform much worse on non-English, non-Latin scripts.
- he2025seeing | arXiv 2506.20168 | "Seeing is Believing?" (KIE-HVQA benchmark): OCR hallucination by MLLMs on degraded documents.
- gardella2026low | arXiv 2607.24077 | VLM-based OCR on historical documents can fabricate fluent text even when the character error rate is low, so CER alone misses hallucinations.
- cheng2024unveiling | Crossref DOI 10.1007/978-3-031-73202-7_11 | Typographic vulnerability of LVLMs (ECCV 2024): text in the image strongly steers model output.

## 4. Hallucination detection and uncertainty
- manakul2023selfcheckgpt | Crossref DOI 10.18653/v1/2023.emnlp-main.557 | SelfCheckGPT: sampling consistency as a black-box hallucination detector (basis for our re-reading consistency detector).
- farquhar2024detecting | Crossref DOI 10.1038/s41586-024-07421-0 | Semantic entropy over multiple samples detects confabulations (Nature 2024).
- kuhn2023semantic | arXiv 2302.09664 | Semantic uncertainty / semantic entropy (published at ICLR 2023; ICLR has no DOI, so the arXiv version is cited).
- lin2022teaching | arXiv 2205.14334 | Models can be trained to express calibrated uncertainty in words (published in TMLR 2022; cited via arXiv).
- xiong2023can | arXiv 2306.13063 | Verbalised confidence from LLMs is often overconfident; compares elicitation strategies (published at ICLR 2024; cited via arXiv).
- kadavath2022language | arXiv 2207.05221 | LLMs are fairly well calibrated about whether they know an answer (P(True)), which supports self-evaluation signals.
- yona2024can | Crossref DOI 10.18653/v1/2024.emnlp-main.443 | LLMs do not faithfully express their intrinsic uncertainty through verbal hedging (supports the expected weakness of the hedging detector).

## 5. Statistics
- wilson1927probable | Crossref DOI 10.1080/01621459.1927.10502953 | Wilson score interval for binomial proportions (hallucination rates).
- hanley1982meaning | Crossref DOI 10.1148/radiology.143.1.7063747 | How to interpret ROC AUC and its standard error (detector evaluation).
- delong1988comparing | Crossref DOI 10.2307/2531595 | Nonparametric test for comparing correlated AUCs (comparing detectors on the same items). NOTE: Crossref lists only the first page (837); pages set to 837--845 from the standard citation.
- cohen1960coefficient | Crossref DOI 10.1177/001316446002000104 | Cohen's kappa for inter-annotator / inter-model agreement.
- efron1979bootstrap | Crossref DOI 10.1214/aos/1176344552 | Bootstrap confidence intervals (e.g., for AUC and rate differences).

## 6. OCR engine
- smith2007overview | Crossref DOI 10.1109/ICDAR.2007.4376991 | Tesseract OCR engine overview (the OCR cross-check detector).
  (No peer-reviewed Ukrainian/Cyrillic Tesseract evaluation with a DOI could be verified, so none was included.)

## 7. Synthetic text data
- gupta2016synthetic | Crossref DOI 10.1109/CVPR.2016.254 | SynthText: rendering synthetic text into images gives exact ground truth (precedent for our synthetic stimuli).
- jaderberg2016reading | Crossref DOI 10.1007/s11263-015-0823-z | Reading text in the wild using CNNs trained purely on synthetic data (IJCV 116, 2016; online 2015). It is the journal version of the 2014 synthetic-data work; arXiv 1406.2227 could not be fetched because of arXiv rate limits, so it is not included.

## Not included (could not be verified or out of scope)
- "Kupferstein" BLV/LMM paper: nothing found in Crossref or arXiv.
- WCAG: no DOI, skipped as requested.
- OCRBench v2 (arXiv 2501.00321, verified) and the GPT-4 technical report (arXiv 2303.08774, verified) were left out to keep the list short; they can be added if needed.
