from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict:
    target = Path(path)
    return {
        "path": str(target.resolve()),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def git_revision(path: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_record(torch_module=None) -> dict:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version
            for name in ("numpy", "torch", "transformers", "Pillow")
            if (version := _version(name)) is not None
        },
    }
    if torch_module is not None:
        cuda_available = bool(torch_module.cuda.is_available())
        result["cuda"] = {
            "available": cuda_available,
            "runtime": torch_module.version.cuda,
            "cudnn": torch_module.backends.cudnn.version(),
            "device": torch_module.cuda.get_device_name(0) if cuda_available else None,
        }
    return result
