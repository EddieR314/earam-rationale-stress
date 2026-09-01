# EARAM Public-Code Reproduction Audit

Audited repository: <https://github.com/qingpingwan/EARAM>

## Verified interfaces

- `data_loader/VLM_MR2_en_dataloader.py` expects `dataset_merge/en_train.json` and
  `dataset_merge/en_test.json` plus two line-aligned rationale files per split.
- The released training rationale files both contain 2,558 lines.
- The released first test rationale file contains 319 lines.
- `prepare-mr2` reconstructs those exact row counts from official MR2 metadata and writes a
  source-row manifest for alignment checks.

## Missing or non-portable pieces

- The second 319-line test rationale file is missing.
- The data root is hard-coded as `./data_3_MR2`.
- The device is hard-coded as `cuda:3`.
- CLIP is loaded from `/data4/zxf/hf/openai/clip-vit-large-patch14`.
- Training hyperparameters are module-level constants and no seed is set.
- The released script evaluates the test loader every epoch, so using it for model selection would
  leak test information unless a separate development split is introduced.

## Minimal faithful port

Before running on a GPU machine:

1. Obtain `MR2_en_test_analysis_2.txt` from the authors and verify it has 319 lines.
2. Make dataset root, checkpoint name/path, device, seed, epochs, and output path CLI arguments.
3. Freeze CLIP if matching the released implementation; record whether gradients are enabled.
4. Add a validation split for checkpoint selection and evaluate the 319-row test set once.
5. Save IDs, labels, predictions, seed, package versions, and checkpoint hash for every condition.
6. Run clean, corruption-only, and corruption-plus-filter conditions from the same checkpoint.

Until these conditions are met, results should be labeled as diagnostics or ablations rather than
paper reproduction.

## Adopted independent protocol

Because the missing test rationale cannot currently be obtained, this project uses only the 2,558
rows with two released rationales. `make-splits` creates stratified 80/10/10 partitions for seeds
13, 42, and 97. `scripts/run_earam_portable.py` uses validation Macro-F1 for checkpoint selection
and evaluates the internal test split once. This removes the missing-file dependency but changes the
evaluation population, so the result is labeled EARAM-style rather than official reproduction.
