# EARAM Rationale Reliability Stress Test

[![Tests](https://github.com/EddieR314/earam-rationale-stress/actions/workflows/tests.yml/badge.svg)](https://github.com/EddieR314/earam-rationale-stress/actions/workflows/tests.yml)

![Within-label rationale shuffle results](docs/within_label_shuffle_results.png)

A reproducible reliability audit of **EARAM** (*From Predictions to Analyses: Rationale-Augmented Fake News Detection with Large Vision-Language Models*). It tests how an EARAM-style detector responds when LVLM analyses are absent, assigned to the wrong sample, incomplete, irrelevant, contradictory, unsupported, or given a flipped conclusion.

This repository does **not** claim to reproduce EARAM's published numbers. It contains a low-memory execution path for the released VLR architecture, a controlled rationale-intervention toolkit, and preliminary internal results on the 2,558 MR2 rows for which both released rationale files are available.

## Result in 30 seconds

| Archived fixed-checkpoint condition | Mean test Macro-F1 | Change from clean |
|---|---:|---:|
| Clean rationales | 0.9333 | — |
| Rationale pairs shuffled within the same label | 0.9517 | **+0.0184** |

**Follow-up finding.** Same-label shuffling breaks sample-level rationale correspondence while
preserving label-associated rationale signals. Across three frozen model checkpoints and three
fixed-point-free shuffle permutations, it did not reduce Macro-F1; the mean instead increased by
0.0184. Only 2 of 9 individual paired-bootstrap intervals were strictly above zero, so the increase
itself should not be treated as a general performance improvement.

**Interpretation.** This result overturns the simple interpretation of the earlier unrestricted
shuffle pilot: its drop cannot be attributed to sample-level semantic misalignment alone. A likely
confound is label-associated information retained or disrupted by different shuffle designs. This
does not establish that rationales are useless, nor does it reproduce EARAM's official result.

## Research question

> How robust is rationale-augmented fake-news detection when the generated rationales are unreliable, and can a lightweight pre-filter reduce the damage?

## Internal pilot protocol

On the 2,558 public MR2 training rows, we trained three low-memory EARAM-style models on
stratified 80/10/10 splits (seeds 13, 42, and 97), then evaluated frozen clean checkpoints after
moving both rationale channels together to a different sample of the same class. Every permutation
had zero fixed points. Clean test Macro-F1 was **0.9333** on average; within-label shuffling produced
**0.9517** across the 3 × 3 evaluation matrix (mean change **+0.0184**).

All nine result files, prediction files, paired-bootstrap outputs, runtime manifests, cache
manifests, and donor manifests are archived under `results/within-label/raw`. These are internal
EARAM-style results—not a reproduction of the paper's official MR2 result and not a claim about the
official model generally. The earlier unrestricted-shuffle pilot is retained in `RESULTS.md` as an
unarchived historical result and is not pooled with this rerun. User-specific local paths in the
archive are replaced with `<PROJECT_ROOT>` while linked hashes remain internally consistent.

Evidence and scope:

- [`docs/LUO_LAB_ONE_PAGE.md`](docs/LUO_LAB_ONE_PAGE.md): concise bilingual research brief.
- [`docs/WITHIN_LABEL_RESULTS.md`](docs/WITHIN_LABEL_RESULTS.md): exact 3 × 3 follow-up table and interpretation boundary.
- [`docs/within_label_results.csv`](docs/within_label_results.csv): machine-readable per-run metrics.
- [`docs/results_summary.csv`](docs/results_summary.csv): machine-readable aggregate results.
- [`docs/PILOT_PROVENANCE.md`](docs/PILOT_PROVENANCE.md): protocol, available evidence, and missing artifacts.
- [`RESULTS.md`](RESULTS.md): EARAM-style pilot plus the earlier text-only feasibility diagnostic.
- [`EARAM_REPRODUCTION_AUDIT.md`](EARAM_REPRODUCTION_AUDIT.md): public-code audit and reproduction boundary.

## What is implemented

- Adapter for EARAM's two line-aligned rationale files.
- Five deterministic perturbations:
  - `evidence_deletion`
  - `irrelevant`
  - `contradiction`
  - `unsupported_claim`
  - `conclusion_flip`
- Independent controls for corruption severity, target rate, and random seed.
- Transparent heuristic reliability baseline using relevance, evidence density, completeness, cross-rationale verdict consistency, and penalties.
- Export back to the exact two-file format expected by EARAM.
- Text-level stress-test summaries and classification metrics for EARAM predictions.
- Fixed-point-free cross-sample and within-label rationale-pair shuffle controls with donor manifests.
- Paired bootstrap confidence intervals for clean-versus-condition Macro-F1 differences.
- A reproducible command that reconstructs EARAM's 2,558/319 English binary MR2 subset.
- An 8 GB VRAM workflow that streams/selectively extracts MR2, caches CLIP-Large features in
  FP16, trains the released VLR with gradient accumulation, and evaluates corrupted conditions
  against a fixed clean checkpoint.
- A research protocol with hypotheses, controls, metrics, and claims boundaries in `RESEARCH_PLAN.md`.
- A three-seed diagnostic report in `RESULTS.md` and a public-code audit in
  `EARAM_REPRODUCTION_AUDIT.md`.

## Install and test

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
python -m unittest discover -s tests
```

The tests use only Python's standard library. `pytest` also works if installed.

## Quick demo

```bash
earam-stress perturb \
  --input examples/sample.jsonl \
  --output runs/unsupported.jsonl \
  --type unsupported_claim \
  --severity 0.7 \
  --target-rate 1.0 \
  --seed 42

earam-stress filter \
  --input runs/unsupported.jsonl \
  --output runs/unsupported.filtered.jsonl \
  --threshold 0.45 \
  --strategy peer

earam-stress summarize \
  --clean examples/sample.jsonl \
  --candidate runs/unsupported.filtered.jsonl
```

## Use with the official EARAM repository

### Reconstruct the MR2 subset

The public EARAM loader expects derived files that are not included in its repository. The exact
published-code row counts can be reconstructed from the official MR2 metadata by selecting the
longest all-ASCII caption suffix and excluding label `2` (unverified). EARAM's `test` subset comes
from MR2's validation split, not its test split.

```bash
earam-stress prepare-mr2 \
  --train-json /path/to/dataset_items_train.json \
  --validation-json /path/to/dataset_items_val.json \
  --archive /path/to/full-mr2.tar.gz \
  --output-dir runs/mr2-earam
```

This writes `dataset_merge/en_train.json` (2,558 rows), `dataset_merge/en_test.json`
(319 rows), aligned caption files, a source-row manifest, and only the 2,877 required images.

Reproduction caveat: the public EARAM repository contains both 2,558-line training rationale
files, but only `MR2_en_test_analysis_1.txt`; its data loader also requires an unpublished
`MR2_en_test_analysis_2.txt`. A faithful clean test run therefore requires obtaining that missing
file from the authors. Duplicating rationale 1 is allowed only as an explicitly labeled ablation,
not as a reproduction of the paper's result.

EARAM stores each LVLM analysis as one line in two aligned files, for example:

- `LVLMs_analysis/MR2_analyses/MR2_en_train_analysis_1.txt`
- `LVLMs_analysis/MR2_analyses/MR2_en_train_analysis_2.txt`

Import them:

```bash
earam-stress import-earam \
  --analysis-1 /path/to/EARAM/LVLMs_analysis/MR2_analyses/MR2_en_train_analysis_1.txt \
  --analysis-2 /path/to/EARAM/LVLMs_analysis/MR2_analyses/MR2_en_train_analysis_2.txt \
  --output runs/mr2.clean.jsonl
```

Create a corruption condition and filter it:

```bash
earam-stress perturb \
  --input runs/mr2.clean.jsonl \
  --output runs/mr2.contradiction.s05.jsonl \
  --type contradiction --severity 0.5 --target-rate 0.5 --seed 42

earam-stress filter \
  --input runs/mr2.contradiction.s05.jsonl \
  --output runs/mr2.contradiction.s05.filtered.jsonl \
  --threshold 0.4 --strategy peer
```

Export the transformed analyses:

```bash
earam-stress export-earam \
  --input runs/mr2.contradiction.s05.filtered.jsonl \
  --analysis-1 runs/earam_files/analysis_1.txt \
  --analysis-2 runs/earam_files/analysis_2.txt
```

Point the official EARAM data loader at the exported files, train with the same split and seed, and save one JSONL prediction file per condition:

```json
{"id":"0","label":1,"prediction":0}
{"id":"1","label":0,"prediction":0}
```

Evaluate it:

```bash
earam-stress evaluate-predictions --input runs/predictions.jsonl
```

Generate the full five-perturbation, four-rate matrix in one command:

```bash
earam-stress matrix \
  --input runs/mr2.clean.jsonl \
  --output-dir runs/matrix \
  --rates 0.25 0.50 0.75 1.00 \
  --severity 0.7 --seed 42 --threshold 0.4 --strategy peer
```

The `peer` filter replaces a rejected rationale with the other LVLM analysis when that analysis passes the threshold. If neither passes, it emits an empty line. Use `--strategy drop` for a strict ablation that always removes rejected rationales.

Tune the threshold on a held-out development split with known synthetic corruption labels:

```bash
earam-stress calibrate \
  --clean runs/mr2.clean.jsonl \
  --input runs/mr2.contradiction.s05.jsonl \
  --thresholds 0.25 0.35 0.45 0.55 0.65 0.75 \
  --strategy peer --output runs/calibration.json
```

Do not tune the threshold on the same test conditions used for final reporting.

## Low-cost diagnostic before a full EARAM run

When a CLIP-capable GPU is unavailable, run the optional text-only probe on a fixed stratified
20% holdout of EARAM's 2,558-row training subset:

```bash
pip install -e '.[probe]'
earam-stress run-text-probe \
  --dataset-json runs/mr2-earam/dataset_merge/en_train.json \
  --analysis-1 /path/to/EARAM/MR2_en_train_analysis_1.txt \
  --analysis-2 /path/to/EARAM/MR2_en_train_analysis_2.txt \
  --rates 0.25 0.50 0.75 1.00 \
  --severity 0.7 --seed 42 --threshold 0.65 \
  --output runs/text-probe.seed42.json
```

This is a feasibility diagnostic using TF-IDF and logistic regression. It does not use images,
CLIP, cross-attention, or EARAM's architecture, so its scores must never be reported as an EARAM
reproduction. Its purpose is to test whether controlled rationale corruption produces a measurable
signal before spending GPU time. The probe reports both the transparent heuristic filter and a
lightweight learned corruption detector. Because the latter is trained and tested with the same
synthetic corruption generators, it may recognize generator artifacts; it is not a factuality
detector and must be validated on human-written corruptions before making a broader claim.

## Self-contained 80/10/10 EARAM-style experiment

The public second MR2 test rationale is unavailable, so the primary continuation no longer depends
on that file. Generate aligned partitions from the 2,558 rows for which both rationales are public:

```bash
earam-stress make-splits \
  --dataset-json runs/mr2-earam/dataset_merge/en_train.json \
  --analysis-1 /path/to/EARAM/MR2_en_train_analysis_1.txt \
  --analysis-2 /path/to/EARAM/MR2_en_train_analysis_2.txt \
  --seeds 13 42 97 --output-dir runs/internal-splits

earam-stress validate-split --split-dir runs/internal-splits/seed13
```

Each seed directory contains disjoint, stratified train/validation/test JSON files and six aligned
rationale files. On a CLIP-capable GPU, clone the public EARAM repository and run:

```bash
pip install -e '.[gpu]'
python scripts/run_earam_portable.py \
  --earam-repo /path/to/EARAM \
  --split-dir runs/internal-splits/seed13 \
  --image-root /path/to/mr2-earam/dataset_merge \
  --seed 13 --output-dir runs/earam/seed13
```

Use `--dry-run` first to validate `model.py`, all six aligned files, and every image path without
importing PyTorch. The runner freezes CLIP, trains the released `VLR`, selects by validation
Macro-F1, and evaluates the internal test partition once. Repeat for seeds `42` and `97`.

These are **EARAM-style controlled robustness experiments**, not reproductions of the paper's
official MR2 score.

## Within-label shuffle control

Generate a deterministic rationale-pair condition with no self-assignments. The default
`within-label` mode preserves each recipient's class while breaking sample-level correspondence;
`cross-sample` reproduces the less controlled unrestricted shuffle.

```bash
earam-stress shuffle-rationales \
  --dataset-json runs/mr2-earam/dataset_merge/en_train.json \
  --analysis-1 /path/to/EARAM/MR2_en_train_analysis_1.txt \
  --analysis-2 /path/to/EARAM/MR2_en_train_analysis_2.txt \
  --mode within-label --seed 13 \
  --output-dir runs/rationales/within-label-seed13
```

Both rationale channels move together as one pair. The output directory contains the two aligned
analysis files plus `manifest.json`, which records the donor mapping, input SHA-256 hashes, seed,
label counts, and fixed-point count. Precompute a temporary feature cache from those files and
evaluate it with the corresponding frozen clean checkpoint. The cached trainer now writes a
`run_manifest.json` containing exact arguments, code revisions, runtime versions, input hashes,
checkpoint hash, and prediction/result hashes.

After evaluating a condition, compare its sample-level predictions with the corresponding clean
run. Pairing is by record ID, so a reordered JSONL file cannot silently invalidate the comparison.

```bash
earam-stress compare-predictions \
  --clean runs/earam/seed13/test_predictions.jsonl \
  --candidate runs/conditions/within-label-seed13/test_predictions.jsonl \
  --samples 2000 --seed 42 \
  --output runs/conditions/within-label-seed13/paired_bootstrap.json
```

After completing the 3 × 3 matrix, generate the exact per-run CSV and Markdown report directly
from the paired-bootstrap files:

```bash
python scripts/summarize_within_label_results.py \
  --conditions-root runs/conditions \
  --output-csv docs/within_label_results.csv \
  --output-markdown docs/WITHIN_LABEL_RESULTS.md
python scripts/generate_within_label_figure.py
```

The committed follow-up snapshot uses `--handoff-dir results/within-label/raw` to rebuild the same
outputs from the archived flat artifact bundle.

Validate hashes, donor mappings, record counts, and all nine recomputed bootstrap outputs with:

```bash
PYTHONPATH=src python scripts/validate_within_label_archive.py \
  --archive-dir results/within-label/raw
```

## Minimum experiment matrix

Use seeds `13, 42, 97`, target rates `0.25, 0.50, 0.75, 1.00`, and all five perturbations. Report:

1. Clean EARAM Macro-F1.
2. Macro-F1 under each corruption type/rate.
3. Macro-F1 after reliability filtering.
4. Robustness area under the corruption curve.
5. Three qualitative cases showing the first harmful rationale segment.

Do not claim the heuristic filter establishes factual correctness. It is a transparent baseline for selecting or rejecting suspicious rationales; factual verification requires external evidence or human annotation.

## Provenance

- EARAM paper: <https://doi.org/10.1145/3696410.3714532>
- Official EARAM code: <https://github.com/qingpingwan/EARAM>

The project is an independent research prototype and is not an official LUD release.

For an RTX 5060 Laptop with 8 GB VRAM, use the cached-feature workflow in
[`LOCAL_WINDOWS.md`](LOCAL_WINDOWS.md). It freezes CLIP-Large, stores its token features once in
FP16, and trains the released `VLR` at batch size one with gradient accumulation. This is a
low-memory execution adaptation, so report it separately from the paper's original batch-16 run.
