import platform
import re
from pathlib import Path


def _read_proc_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError as e:
        raise RuntimeError(f"Cannot read {path}: {e}") from e


def get_cpu_flags() -> set[str]:
    content = _read_proc_file("/proc/cpuinfo")
    for line in content.splitlines():
        if line.startswith("flags") or line.startswith("Features"):
            return set(line.split(":", 1)[1].split())
    return set()


def get_available_ram_gb() -> float:
    content = _read_proc_file("/proc/meminfo")
    for line in content.splitlines():
        if line.startswith("MemAvailable"):
            kb = int(re.search(r"\d+", line).group())
            return round(kb / (1024 ** 2), 2)
    return 0.0


def _classify_simd(flags: set[str]) -> str:
    if "avx512f" in flags:
        return "avx512"
    if "avx2" in flags:
        return "avx2"
    if "avx" in flags:
        return "avx"
    if "sse4_2" in flags:
        return "sse4"
    return "none"


def _ram_threshold_gb(model_id: str) -> float:
    """Minimum free RAM needed based on model size."""
    if "E4B" in model_id or "4B" in model_id:
        return 11.0
    if "26B" in model_id or "31B" in model_id:
        return 32.0
    return 6.0  # safe default for E2B and unknown models


def check_compatibility(model_id: str = "") -> dict:
    if platform.system() != "Linux":
        raise RuntimeError(
            f"CPU check requires Linux (/proc/cpuinfo). Detected: {platform.system()}"
        )

    flags = get_cpu_flags()
    simd = _classify_simd(flags)
    ram_gb = get_available_ram_gb()
    threshold = _ram_threshold_gb(model_id)
    ram_ok = ram_gb >= threshold

    warnings = []
    if simd == "none":
        warnings.append("No SSE4.2/AVX support detected — inference will be very slow or fail")
    elif simd in ("sse4", "avx"):
        warnings.append(f"SIMD level '{simd}' detected — AVX2 or AVX-512 recommended for better performance")
    if not ram_ok:
        warnings.append(
            f"Only {ram_gb:.1f} GB RAM available; {threshold:.1f} GB recommended for {model_id or 'this model'}"
        )

    return {
        "compatible": simd != "none" and ram_ok,
        "simd_level": simd,
        "available_ram_gb": ram_gb,
        "ram_threshold_gb": threshold,
        "ram_sufficient": ram_ok,
        "cpu_flags_count": len(flags),
        "warnings": warnings,
    }
