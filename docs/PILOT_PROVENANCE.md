# Pilot Result Provenance and Audit Status

## Scope

This document records exactly what can and cannot currently be audited for the preliminary
EARAM-style fixed-checkpoint experiment. It is intended to prevent aggregate pilot results from
being mistaken for an official EARAM reproduction or a fully archived benchmark result.

## Recorded protocol

- Data: 2,558 public English binary MR2 training rows with both released EARAM rationales.
- Split: stratified 80/10/10 internal train/validation/test partitions.
- Model seeds: 13, 42, and 97.
- Architecture: the released EARAM `VLR` model with frozen CLIP-Large token features.
- Hardware path: FP16 cached CLIP features on an RTX 5060 Laptop GPU with 8 GB VRAM.
- Model selection: best validation Macro-F1; internal test evaluated after selection.
- Intervention evaluation: the trained clean checkpoint is held fixed while rationale inputs change.
- Shuffle evaluation: three model seeds × three independent cross-sample shuffle permutations.

The portable runner defaults are 35 epochs, AdamW learning rate `2e-5`, weight decay `0.01`,
gradient accumulation `8`, and EARAM head-loss weights `(0.7, 0.15, 0.15)`. The current aggregate
snapshot does not preserve a machine-readable record proving that every pilot run used every
default unchanged, so these values describe the intended protocol rather than a complete run log.

## Public evidence in this snapshot

- Aggregate condition means: [`results_summary.csv`](results_summary.csv).
- Presentation figure generated from that CSV: [`rationale_reliability_results.png`](rationale_reliability_results.png).
- Training and fixed-checkpoint evaluation implementation: [`../scripts/train_cached_earam.py`](../scripts/train_cached_earam.py).
- Split generation and validation: [`../src/earam_stress/splits.py`](../src/earam_stress/splits.py).
- Controlled perturbation toolkit: [`../src/earam_stress/perturb.py`](../src/earam_stress/perturb.py).

## Artifacts not retained

The original per-run `result.json`, prediction JSONL files, training histories, checkpoint hashes,
package lockfile, and GPU/software metadata were not retained in this repository. The published
means therefore cannot yet be reconstructed solely from the repository.

No missing per-run values have been inferred from the means. The aggregate result should be cited
as a **preliminary internal pilot** until a clean rerun archives the complete artifacts.

## Required archive for the next run

For every model seed and intervention seed, retain:

1. command-line arguments and Git commit hashes for this project and the EARAM repository;
2. Python, PyTorch, Transformers, CUDA, and GPU versions;
3. split manifest and hashes of the two source rationale files;
4. clean checkpoint SHA-256 and selected epoch;
5. `result.json` and sample-level predictions with probabilities;
6. condition generator, rate, severity, target channel, and random seed;
7. paired bootstrap confidence intervals against the corresponding clean predictions.

## Decisive next control

Run **within-label rationale shuffling** before making a semantic-alignment claim. Unrestricted
cross-sample shuffling breaks sample correspondence but may also alter label-associated signals.
Within-label shuffling preserves the label distribution while breaking sample-level correspondence.

After that control, test a lightweight image-caption-rationale alignment gate on exactly the same
frozen checkpoints. A useful mitigation must improve the shuffled condition without materially
reducing clean-condition performance.
