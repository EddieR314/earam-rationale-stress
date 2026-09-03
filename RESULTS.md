# Results

## EARAM-style fixed-checkpoint pilot

Three low-memory EARAM-style models were trained on stratified 80/10/10 splits of the 2,558 MR2
rows for which both released rationales are available. Model seeds were 13, 42, and 97. CLIP-Large
was frozen and its token features were cached in FP16 so that the released VLR architecture could
run on an RTX 5060 Laptop GPU with 8 GB VRAM.

| Condition | Mean test Macro-F1 | Change from clean |
|---|---:|---:|
| Clean rationales | 0.9193 | — |
| Incorrect verdict prepended to 50% of rationale fields | 0.9207 | +0.0014 |
| Both rationales removed | 0.9108 | −0.0086 |
| Rationale pairs shuffled across samples | 0.9054 | −0.0139 |

Every intervention was evaluated with the corresponding clean checkpoint fixed. The shuffled
mean covers three model seeds × three independent permutations. The result supports a narrow claim:
this implementation is sensitive to cross-sample rationale mismatch, and mismatch was more harmful
than absence in the pilot. It does not yet isolate semantic misalignment from label-associated
signals; within-label shuffling is the next required control.

The original per-run outputs were not retained in this repository. See
[`docs/PILOT_PROVENANCE.md`](docs/PILOT_PROVENANCE.md) for the audit status and required rerun
archive. These values are preliminary internal results, not official EARAM reproduction numbers.

## Earlier text-only feasibility diagnostic

## Status

This is a completed **text-only diagnostic**, not a reproduction of EARAM. It uses the exact
2,558-row English binary MR2 training subset inferred from the public EARAM release, then makes a
fixed stratified 80/20 split for each of three seeds (`13`, `42`, `97`). The classifier is a
class-balanced TF-IDF logistic regression over caption and two rationales.

## Reconstructed data

- MR2 train: longest all-ASCII suffix starts at key `6406`; removing label `2` gives 2,558 rows.
- MR2 validation: longest all-ASCII suffix starts at key `714`; removing label `2` gives 319 rows.
- All 2,877 selected images were found in the official 28.7 GB archive and extracted successfully.
- Labels are preserved as `0 = non-rumor`, `1 = rumor`; `2 = unverified` is excluded.

The two inferred counts exactly match the line counts in EARAM's released train/test rationale
files, and manual spot checks show the first released test rationale corresponds to MR2 validation
key `714`.

## Three-seed diagnostic

Mean and sample standard deviation across seeds:

| Condition | Target rate | Macro-F1 |
|---|---:|---:|
| Clean | 0% | 0.876 ± 0.006 |
| Evidence deletion | 25% | 0.874 ± 0.010 |
| Evidence deletion | 50% | 0.852 ± 0.014 |
| Evidence deletion | 75% | 0.827 ± 0.018 |
| Evidence deletion | 100% | 0.827 ± 0.014 |
| Irrelevant rationale | 25% | 0.788 ± 0.024 |
| Irrelevant rationale | 50% | 0.720 ± 0.031 |
| Irrelevant rationale | 75% | 0.618 ± 0.011 |
| Irrelevant rationale | 100% | 0.527 ± 0.029 |

Conclusion flip, unsupported-claim insertion, and appended contradiction changed Macro-F1 by less
than about 0.01 on average in this bag-of-ngrams probe. That is not evidence that EARAM is robust to
them; it shows that this diagnostic classifier is largely insensitive to those edits.

## Filter result: negative

The transparent heuristic filter at threshold `0.35` rejects about 0.45% of clean rationales on the
full subset. Across all 60 seed/perturbation/rate conditions, it changes Macro-F1 relative to the
unfiltered corruption by **-0.0053 ± 0.0108**.

The learned synthetic-corruption detector has a **4.52% ± 0.46%** false-positive rate on clean
held-out rationales. Its mean Macro-F1 change relative to unfiltered corruption is
**-0.0321 ± 0.0217**. It recognizes fixed generator artifacts but largely misses natural, irrelevant
rationales—the condition that damages the downstream classifier most.

Therefore neither filter is a successful method. They are retained as failed baselines. The useful
finding is that surface-form anomaly detection is not enough; reliability estimation must be
conditioned on the current image-caption pair and tested on non-template, preferably human-written,
errors.

## Reproduction blockers in the public EARAM release

1. `MR2_en_test_analysis_2.txt` is required by the data loader but absent from the repository.
2. `utils.py` hard-codes `cuda:3` and an author-local CLIP checkpoint path.
3. The released training code requires portability changes for commodity GPUs and reproducible
   validation-based checkpoint selection.

A faithful EARAM result must wait for the missing second test rationale and a CLIP-capable GPU. It
would be misleading to duplicate rationale 1 and call that a reproduction.

The implemented continuation avoids the missing file by creating internal stratified 80/10/10
splits from the 2,558 complete rows for seeds 13, 42, and 97. That independent EARAM-style GPU
experiment is reported at the top of this file; it remains distinct from the paper's official split.

## Next experiment after the text-only diagnostic

Using the already implemented fixed-checkpoint path, prioritize within-label rationale shuffling,
single-channel versus dual-channel corruption, and paired confidence intervals. Then add a
cross-modal reliability head that scores each rationale against both CLIP image tokens and caption
tokens, and compare it against the two failed text-only filters here.
