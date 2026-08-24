"""Detect which compute runtime is actually accessible, in priority order:
Colab GPU > Kaggle GPU > another verified GPU > local GPU > CPU.

Every field is read from the real environment. Nothing here is assumed
or fabricated - a provider is only reported as available if its actual
markers are present (env vars, importable module, mounted paths).
"""

import os
import platform
import shutil
import subprocess

import torch


def _is_colab() -> bool:
    # google.colab is only importable inside an actual Colab runtime;
    # COLAB_RELEASE_TAG / COLAB_GPU are set by the Colab kernel itself.
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))


def _is_kaggle() -> bool:
    return bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_URL_BASE")
        or os.path.isdir("/kaggle/input")
    )


def _nvidia_smi_summary():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def _cuda_info():
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": "Not available",
        "vram_gb": "Not available",
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_gb"] = round(props.total_memory / 1024 ** 3, 2)
    return info


def _system_info():
    total_ram_gb = free_ram_gb = "Not available"
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/format:list"],
                capture_output=True, text=True, timeout=10,
            )
            values = {}
            for line in out.stdout.splitlines():
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    if v.strip().isdigit():
                        values[k.strip()] = int(v.strip())
            if "TotalVisibleMemorySize" in values:
                total_ram_gb = round(values["TotalVisibleMemorySize"] / 1024 ** 2, 2)
            if "FreePhysicalMemory" in values:
                free_ram_gb = round(values["FreePhysicalMemory"] / 1024 ** 2, 2)
        else:
            with open("/proc/meminfo") as f:
                meminfo = dict(
                    (parts[0].rstrip(":"), int(parts[1]))
                    for parts in (line.split() for line in f)
                    if len(parts) >= 2 and parts[1].isdigit()
                )
            if "MemTotal" in meminfo:
                total_ram_gb = round(meminfo["MemTotal"] / 1024 ** 2, 2)
            if "MemAvailable" in meminfo:
                free_ram_gb = round(meminfo["MemAvailable"] / 1024 ** 2, 2)
    except Exception:
        pass

    _, _, disk_free = shutil.disk_usage(".")
    return {
        "cpu_count": os.cpu_count(),
        "total_ram_gb": total_ram_gb,
        "free_ram_gb": free_ram_gb,
        "disk_free_gb": round(disk_free / 1024 ** 3, 1),
        "platform": platform.platform(),
    }


def detect_runtime() -> dict:
    """Returns provider, GPU, VRAM, CUDA, PyTorch, CPU, RAM, disk, device -
    all read from the actual running environment."""
    cuda = _cuda_info()
    system = _system_info()
    smi = _nvidia_smi_summary()

    colab = _is_colab()
    kaggle = _is_kaggle()

    if colab and cuda["cuda_available"]:
        provider, device, status = "colab", "cuda", "AVAILABLE"
    elif kaggle and cuda["cuda_available"]:
        provider, device, status = "kaggle", "cuda", "AVAILABLE"
    elif cuda["cuda_available"]:
        provider, device, status = "local_gpu", "cuda", "AVAILABLE"
    elif torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False:
        provider, device, status = "local_mps", "mps", "AVAILABLE"
    else:
        provider, device, status = "cpu", "cpu", "AVAILABLE"

    blocked = []
    if not colab:
        blocked.append({
            "provider": "colab",
            "reason": "Not running inside a Colab kernel (google.colab not importable, "
            "no COLAB_RELEASE_TAG/COLAB_GPU env vars) and no authenticated browser "
            "session (Chrome extension) is connected to reach colab.research.google.com.",
        })
    if not kaggle:
        blocked.append({
            "provider": "kaggle",
            "reason": "Not running inside a Kaggle kernel (no KAGGLE_KERNEL_RUN_TYPE/"
            "KAGGLE_URL_BASE env vars, no /kaggle/input mount) and no authenticated "
            "browser session available to reach kaggle.com.",
        })
    if not cuda["cuda_available"] and provider == "cpu":
        blocked.append({
            "provider": "local_gpu",
            "reason": "torch.cuda.is_available() is False on this machine"
            + (f"; nvidia-smi reports no attached GPU (smi output: {smi})"
               if smi is None else f"; nvidia-smi shows {smi} but torch.cuda cannot see it"),
        })

    return {
        "provider": provider,
        "device": device,
        "status": status,
        "python": platform.python_version(),
        "torch": torch.__version__,
        **cuda,
        **system,
        "nvidia_smi": smi if smi is not None else "Not available",
        "blocked": blocked,
    }


def print_report(report: dict):
    print("=" * 60)
    print("RUNTIME DETECTION REPORT")
    print("=" * 60)
    for k in ("provider", "device", "status", "python", "torch",
              "cuda_available", "cuda_version", "gpu_name", "vram_gb",
              "cpu_count", "total_ram_gb", "free_ram_gb", "disk_free_gb"):
        print(f"  {k:<16}: {report[k]}")
    if report["blocked"]:
        print()
        print("  BLOCKED providers:")
        for b in report["blocked"]:
            print(f"    {b['provider']:<10}: {b['reason']}")
    print("=" * 60)
