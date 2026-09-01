# Windows local run: RTX 5060 Laptop 8 GB

This path is designed for an 8 GB NVIDIA GPU and about 19 GB of free system-disk space. It does
not run CLIP again during every training epoch. The FP16 feature cache is about 1.9 GB; Python,
PyTorch, CLIP-Large and experiment outputs need several additional GB.

## 1. Install tools

Install 64-bit Python 3.11 and Git for Windows. Open PowerShell in the project folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
git clone https://github.com/qingpingwan/EARAM.git vendor\EARAM
```

The setup uses the official PyTorch CUDA 12.8 wheel, which supports RTX 50-series GPUs.

## 2. Build the lite MR2 directory

Preferred option: stream the official archive but retain only the 2,558 images used by the
internal protocol:

```powershell
python scripts\build_mr2_lite_from_drive.py --output-dir data\mr2-lite
```

Expected stored image data is a small fraction of the 27 GB source. The network still transfers a
large stream, so use a stable connection and prevent the laptop from sleeping.

Google Drive sometimes reports a global download-quota error. In that case, use the official
[MR2 Baidu AI Studio page](https://aistudio.baidu.com/datasetdetail/230144), download its
`data.zip` (27,216.21 MB) to a drive with at least 30 GB free (for example `D:` or an external
disk), then selectively extract:

```powershell
python scripts\build_mr2_lite_from_drive.py `
  --archive D:\earam-data\data.zip `
  --output-dir data\mr2-lite
```

The script reads the archive in place and stores only the lite subset on `C:`. You may delete the
archive after `lite_report.json` confirms `records: 2558` and `images: 2558`.

## 3. Make three aligned splits

```powershell
earam-stress make-splits `
  --dataset-json data\mr2-lite\dataset_merge\en_train.json `
  --analysis-1 vendor\EARAM\LVLMs_analysis\MR2_analyses\MR2_en_train_analysis_1.txt `
  --analysis-2 vendor\EARAM\LVLMs_analysis\MR2_analyses\MR2_en_train_analysis_2.txt `
  --seeds 13 42 97 `
  --output-dir runs\internal-splits
```

## 4. Precompute CLIP-Large once

First test one sample:

```powershell
python scripts\precompute_clip_features.py `
  --dataset-json data\mr2-lite\dataset_merge\en_train.json `
  --analysis-1 vendor\EARAM\LVLMs_analysis\MR2_analyses\MR2_en_train_analysis_1.txt `
  --analysis-2 vendor\EARAM\LVLMs_analysis\MR2_analyses\MR2_en_train_analysis_2.txt `
  --image-root data\mr2-lite\dataset_merge `
  --output-dir cache\clip-large `
  --limit 1
```

If that succeeds, rerun without `--limit 1` and add `--resume`. The cache is shared by all seeds.

```powershell
python scripts\precompute_clip_features.py `
  --dataset-json data\mr2-lite\dataset_merge\en_train.json `
  --analysis-1 vendor\EARAM\LVLMs_analysis\MR2_analyses\MR2_en_train_analysis_1.txt `
  --analysis-2 vendor\EARAM\LVLMs_analysis\MR2_analyses\MR2_en_train_analysis_2.txt `
  --image-root data\mr2-lite\dataset_merge `
  --output-dir cache\clip-large `
  --resume
```

## 5. Train one seed

```powershell
python scripts\train_cached_earam.py `
  --earam-repo vendor\EARAM `
  --split-dir runs\internal-splits\seed13 `
  --feature-dir cache\clip-large `
  --seed 13 `
  --gradient-accumulation 8 `
  --output-dir runs\earam\seed13
```

Monitor memory in another PowerShell window with `nvidia-smi -l 2`. After seed 13 finishes, repeat
with seeds 42 and 97. These are internal EARAM-style experiments, not official-paper reproduction.

If a run is interrupted, rerun feature encoding with `--resume`. Training checkpoints currently
restart per seed, so let one seed finish before shutting down.

## 6. Evaluate one corrupted-rationale condition

Create and export one corrupted rationale pair, for example a 50% conclusion-flip condition:

```powershell
earam-stress import-earam `
  --analysis-1 vendor\EARAM\LVLMs_analysis\MR2_analyses\MR2_en_train_analysis_1.txt `
  --analysis-2 vendor\EARAM\LVLMs_analysis\MR2_analyses\MR2_en_train_analysis_2.txt `
  --captions data\mr2-lite\en_train_captions.txt `
  --output runs\rationales\clean.jsonl

earam-stress perturb `
  --input runs\rationales\clean.jsonl `
  --output runs\rationales\conclusion-flip-r050.jsonl `
  --type conclusion_flip --severity 0.7 --target-rate 0.5 --seed 13

earam-stress export-earam `
  --input runs\rationales\conclusion-flip-r050.jsonl `
  --analysis-1 runs\rationales\condition_analysis_1.txt `
  --analysis-2 runs\rationales\condition_analysis_2.txt
```

Precompute that condition into a temporary feature directory with the same command from step 4,
changing `--analysis-1`, `--analysis-2`, and `--output-dir cache\condition`. Then evaluate the
frozen clean checkpoint:

```powershell
python scripts\train_cached_earam.py `
  --earam-repo vendor\EARAM `
  --split-dir runs\internal-splits\seed13 `
  --feature-dir cache\condition `
  --seed 13 `
  --checkpoint-in runs\earam\seed13\best.pt `
  --output-dir runs\conditions\condition-name\seed13
```

This changes the rationale input while keeping the trained detector fixed. Save `result.json`,
delete `cache\condition`, and reuse that temporary directory for the next condition to avoid
filling the system disk. Keep `cache\clip-large`, because it is the clean baseline cache.
