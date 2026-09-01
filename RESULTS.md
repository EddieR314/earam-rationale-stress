# Initial Results

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
3. The current environment has no CUDA device, PyTorch, or Transformers installation.

A faithful EARAM result must wait for the missing second test rationale and a CLIP-capable GPU. It
would be misleading to duplicate rationale 1 and call that a reproduction.

The implemented continuation avoids the missing file by creating internal stratified 80/10/10
splits from the 2,558 complete rows for seeds 13, 42, and 97. This enables an independent EARAM-style
experiment but does not remove the GPU requirement.

## Next experiment

On a GPU machine, freeze the official clean EARAM checkpoint and evaluate the same perturbation
matrix without retraining. Prioritize irrelevant-rationale replacement and evidence deletion. Add a
cross-modal reliability head that scores each rationale against both CLIP image tokens and caption
tokens, and compare it against the two failed text-only filters here.
