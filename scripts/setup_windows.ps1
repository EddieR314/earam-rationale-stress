$ErrorActionPreference = "Stop"

Write-Host "Checking NVIDIA GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install 64-bit Python 3.11 first."
}

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -e ".[probe]" transformers pillow numpy

& .\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('gpu', torch.cuda.get_device_name(0)); print('cuda', torch.version.cuda); assert torch.cuda.is_available()"

Write-Host "Environment is ready. Activate it with: .\.venv\Scripts\Activate.ps1"
