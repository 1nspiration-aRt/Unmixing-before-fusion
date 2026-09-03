"""
检查 PyTorch、CUDA、cuDNN 和 NVIDIA 驱动环境。

主要作用：定位 CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH 等 GPU 环境错误，
并执行一次最小 CUDA Conv2d 测试。脚本只读系统信息，不修改 Conda 环境、
驱动或项目文件。

运行方法（Python 3.10）：
    python check_cuda_env.py

运行环境：Python 3.10；若已安装 PyTorch，则额外检查 CUDA/cuDNN 和 GPU。
"""

from __future__ import annotations

import ctypes.util
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def print_item(name: str, value: object) -> None:
    """统一输出检查项目，避免依赖第三方日志库。"""

    print(f"{name}: {value}")


def command_output(command: list[str]) -> str | None:
    """执行只读诊断命令；命令不存在或执行失败时返回 None。"""

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def installed_versions() -> dict[str, str]:
    """读取当前解释器中的关键包版本。"""

    versions = {}
    for package in ("torch", "torchvision", "torchaudio", "nvidia-cudnn-cu11", "nvidia-cudnn-cu12"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "未安装"
    return versions


def find_cudnn_libraries() -> list[str]:
    """列出 PATH 中可能重复的 cuDNN DLL/动态库。"""

    patterns = ("cudnn*.dll", "*cudnn*.dll") if os.name == "nt" else ("*cudnn*.so*",)
    found: set[str] = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        path = Path(directory)
        if not path.is_dir():
            continue
        for pattern in patterns:
            found.update(str(item) for item in path.glob(pattern) if item.is_file())
    return sorted(found)


def test_cuda_conv(torch_module) -> tuple[bool, str]:
    """执行最小 CUDA 卷积，直接复现常见 cuDNN 加载问题。"""

    if not torch_module.cuda.is_available():
        return False, "CUDA 不可用，未执行 CUDA Conv2d"
    try:
        device = torch_module.device("cuda")
        model = torch_module.nn.Conv2d(3, 8, kernel_size=3).to(device)
        data = torch_module.randn(1, 3, 16, 16, device=device)
        with torch_module.no_grad():
            model(data)
    except Exception as exc:  # 诊断脚本需要保留完整错误文本。
        return False, f"{type(exc).__name__}: {exc}"
    return True, "CUDA Conv2d 测试通过"


def main() -> int:
    print("=== Python/CUDA 环境诊断 ===")
    print_item("Python", sys.version.replace("\n", " "))
    print_item("平台", platform.platform())
    print_item("解释器", sys.executable)

    print("\n=== 关键包 ===")
    for package, version in installed_versions().items():
        print_item(package, version)

    print("\n=== NVIDIA 驱动 ===")
    nvidia_smi = shutil.which("nvidia-smi")
    print_item("nvidia-smi 路径", nvidia_smi or "未找到")
    driver_info = (
        command_output([nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if nvidia_smi
        else None
    )
    print_item("GPU/驱动", driver_info or "无法读取")

    print("\n=== CUDA/cuDNN ===")
    print_item("ctypes cudnn", ctypes.util.find_library("cudnn") or "未找到")
    libraries = find_cudnn_libraries()
    print_item("PATH 中 cuDNN 动态库数量", len(libraries))
    for library in libraries:
        print(f"  {library}")

    try:
        import torch
    except Exception as exc:
        print_item("PyTorch 导入", f"失败：{type(exc).__name__}: {exc}")
        return 1

    print("\n=== PyTorch 运行时 ===")
    print_item("torch", torch.__version__)
    print_item("torch.version.cuda", torch.version.cuda or "无 CUDA 构建")
    print_item("cuDNN 版本", torch.backends.cudnn.version() or "不可用")
    cuda_available = bool(torch.cuda.is_available())
    print_item("torch.cuda.is_available", cuda_available)
    print_item("GPU 数量", torch.cuda.device_count())
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            print_item(f"GPU {index}", torch.cuda.get_device_name(index))

    passed, detail = test_cuda_conv(torch)
    print_item("CUDA Conv2d", detail)
    if cuda_available and not passed:
        print("\n结论：PyTorch 能发现 GPU，但 CUDA Conv2d 失败，优先检查 PyTorch/cuDNN DLL 与驱动版本混用。")
        return 2
    if not cuda_available:
        print("\n结论：当前环境不可用 CUDA。可先用 Unmixing.py --cuda 0 做 CPU smoke test。")
        return 0
    print("\n结论：CUDA 基础 Conv2d 测试通过，当前环境未复现 cuDNN 子库版本错误。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
