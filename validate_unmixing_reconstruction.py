"""
独立验证训练完成的 UnmixingAE 在 Chikusei tests 集上的 HSI 重构能力。

主要功能：
    1. 使用项目现有的 ``HSIDataset`` 读取 ``dataset/tests/*.mat``，严格复用
       Chikusei 的 59 波段裁剪、伪 RGB 波段选择、归一化和 CHW tensor 预处理；
    2. 从既有 checkpoint 加载完整的 UnmixingAE，执行 ``eval()`` 与
       ``torch.no_grad()`` 下的 encoder inference；
    3. 保存五通道 abundance，并检查形状、范围、NaN/Inf 以及逐像素 simplex
       约束；
    4. 直接使用 checkpoint 中 UnmixingAE 自带的 ``decoderlayer`` 重构完整 59
       波段 HSI，不创建新的 decoder，不重新训练参数；
    5. 逐样本计算 RMSE、MAE、SAM(degree)、PSNR 和 MRAE，并保存 CSV/JSON；
    6. 为确定数量的代表性 tests 样本生成 abundance maps、original/reconstructed
       pseudo-RGB、absolute-error map 和 spectral curves。

当前项目的数据契约：
    原始 MAT key 为 ``Y``，布局为 H x W x C；原始 Chikusei 128 波段输入在
    ``core.loaddata.HSIDataset`` 中选择 Python 索引 ``7:66``，得到 59 波段。
    Chikusei 的 encoder 输入使用归一化后的 59 波段中的 ``(52, 32, 12)``，
    顺序为 R、G、B。HSIDataset 返回的完整 HSI target 才是训练时的 reconstruction
    reference，布局为 C x H x W；本脚本保存时统一转为 H x W x C。

运行环境：
    Python 3、PyTorch、NumPy、SciPy、scikit-image、Matplotlib。
    CUDA 可用时可以使用 GPU；本脚本不会执行训练、optimizer.step() 或参数更新。

推荐运行命令：
    python validate_unmixing_reconstruction.py --input-dir dataset/tests --checkpoint experiments/unmixing/ckpts/UnmixingAE_Chikusei_latest.pth --output-dir "experiments/compare generate hsi/validation_results"  --device auto  --n-blocks 3

输入：
    --input-dir：由 Chikusei 分割脚本生成的 tests MAT 目录，默认 ``dataset/tests``；
    --checkpoint：已经训练完成的 UnmixingAE checkpoint。checkpoint 应为当前项目
                 的 ``{"epoch": ..., "model": state_dict}`` 格式；
    --n-blocks：必须与 checkpoint 的 UnmixingAE 结构一致，默认 3。

输出：
    默认写入 ``experiments/compare generate hsi/validation_results``：
        abundance/<sample>.mat
        reconstructed_hsi/<sample>.mat
        comparison/<sample>_original_rgb.png
        comparison/<sample>_reconstructed_rgb.png
        comparison/<sample>_absolute_error.png
        comparison/<sample>_rgb_pair.png
        spectral_curves/<sample>.png
        metrics.csv
        summary.json

说明：
    指标和保存的 HSI 均使用训练阶段的 per-sample global min-max 归一化尺度；
    不对 decoder 输出执行额外的 per-sample normalization 或 clipping，以避免
    掩盖 decoder 的真实输出范围。伪 RGB 和 absolute-error map 仅在写 PNG 时
    做显示拉伸。PSNR 沿用项目 ``core.metrics.compare_mpsnr`` 的定义，即对每个
    spectral band 计算 PSNR 后取均值，并在 CSV 中命名为 ``PSNR``。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from core.common import default_conv
from core.loaddata import (
    CHIKUSEI_BAND_SLICE,
    CHIKUSEI_RGB_BANDS,
    HSIDataset,
)
from core.metrics import compare_mpsnr, compare_rmse, compare_sam
from unmixingmodel.unmixingAE import UnmixingAE
from checkScripts.visualize_abundance import (
    DEFAULT_CMAP,
    DEFAULT_VMAX,
    DEFAULT_VMIN,
    colorize,
    save_visualizations,
    write_png,
)


DEFAULT_INPUT_DIR = Path("dataset/tests")
DEFAULT_OUTPUT_DIR = (
    Path("experiments") / "compare generate hsi" / "validation_results"
)
EXPECTED_HSI_CHANNELS = 59
EXPECTED_ABUNDANCE_CHANNELS = 5
EXPECTED_INPUT_CHANNELS = 3
DEFAULT_N_BLOCKS = 3
DEFAULT_NUM_VISUAL_SAMPLES = 4
DEFAULT_NUM_CURVE_PIXELS = 6
DEFAULT_SEED = 3000
DEFAULT_MRAE_EPS = 1e-6
DEFAULT_SIMPLEX_ATOL = 1e-5
DEFAULT_SIMPLEX_RTOL = 1e-5


def resolve_device(requested: str) -> torch.device:
    """将 auto/cpu/cuda 解析为实际设备，并拒绝不可用的显式 CUDA。"""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已请求 CUDA，但当前环境中 CUDA 不可用")
    return torch.device(requested)


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> tuple[dict[str, Any], int | None]:
    """读取项目 checkpoint，并去除可能存在的 DataParallel ``module.`` 前缀。

    当前训练代码保存的是 ``{"epoch": epoch, "model": state_dict}``。这里允许
    直接传入 state_dict 作为兼容性兜底，但始终要求最终结果是完整字典；模型
    加载时仍使用严格的 ``load_state_dict``，不会以 ``strict=False`` 静默补层。
    """

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        epoch = checkpoint.get("epoch")
    else:
        state_dict = checkpoint
        epoch = None

    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint 中没有有效的 model state_dict")

    normalized_state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    return normalized_state, int(epoch) if epoch is not None else None


def validate_checkpoint_contract(state_dict: dict[str, Any]) -> int:
    """检查 checkpoint 的输入/abundance/decoder 通道契约并返回输出波段数。"""

    required_keys = {
        "layerup1.weight",
        "layerup1.bias",
        "decoderlayer.weight",
    }
    missing = sorted(required_keys - set(state_dict))
    if missing:
        raise KeyError(f"checkpoint 缺少关键参数：{missing}")

    first_weight = state_dict["layerup1.weight"]
    decoder_weight = state_dict["decoderlayer.weight"]
    first_shape = tuple(getattr(first_weight, "shape", ()))
    decoder_shape = tuple(getattr(decoder_weight, "shape", ()))
    if len(first_shape) < 2 or first_shape[1] != EXPECTED_INPUT_CHANNELS:
        raise ValueError(
            "checkpoint 的 encoder 输入通道不是 3："
            f"实际 layerup1.weight shape={first_shape}"
        )
    if len(decoder_shape) != 4:
        raise ValueError("checkpoint 的 decoderlayer.weight 不是四维卷积权重")
    if decoder_shape[2:] != (1, 1):
        raise ValueError(
            "checkpoint 的 decoderlayer.weight 必须是 [bands, endmembers, 1, 1]，"
            f"实际 shape={decoder_shape}"
        )

    output_channels, abundance_channels = decoder_shape[:2]
    if abundance_channels != EXPECTED_ABUNDANCE_CHANNELS:
        raise ValueError(
            f"checkpoint 的 abundance 通道数应为 {EXPECTED_ABUNDANCE_CHANNELS}，"
            f"实际为 {abundance_channels}"
        )
    if output_channels != EXPECTED_HSI_CHANNELS:
        raise ValueError(
            f"当前 Chikusei 验证要求 decoder 输出 {EXPECTED_HSI_CHANNELS} 个波段，"
            f"checkpoint 实际输出 {output_channels} 个波段"
        )
    return int(output_channels)


def build_model(
    state_dict: dict[str, Any],
    n_blocks: int,
    device: torch.device,
) -> UnmixingAE:
    """按当前训练配置构造模型，并严格加载 checkpoint 权重。"""

    output_channels = validate_checkpoint_contract(state_dict)
    model = UnmixingAE(
        n_blocks=n_blocks,
        res_scale=0.1,
        input_channels=EXPECTED_INPUT_CHANNELS,
        output_channels=output_channels,
        conv=default_conv,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint 与当前 UnmixingAE 结构不匹配；请检查 --n-blocks、模型版本和输出波段数。"
        ) from exc
    return model.to(device)


def forward_with_cudnn_fallback(model: torch.nn.Module, inputs: torch.Tensor):
    """复用项目的 cuDNN 子库不匹配 fallback，仅处理指定错误。"""

    try:
        return model(inputs)
    except RuntimeError as exc:
        mismatch = "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH"
        cudnn_enabled = bool(getattr(torch.backends.cudnn, "enabled", False))
        if not inputs.is_cuda or not cudnn_enabled or mismatch not in str(exc):
            raise

        torch.backends.cudnn.enabled = False
        print(
            "WARNING: 检测到 cuDNN 子库版本不一致，已禁用 cuDNN 并使用原生 CUDA convolution 重试。"
        )
        return model(inputs)


def tensor_to_hwc(tensor: torch.Tensor, name: str) -> np.ndarray:
    """将单样本 C x H x W tensor 转为连续的 H x W x C float32 数组。"""

    if tensor.ndim != 3:
        raise ValueError(f"{name} 应为 CxHxW，实际 shape={tuple(tensor.shape)}")
    array = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.ascontiguousarray(array, dtype=np.float32)


def check_abundance(
    abundance: np.ndarray,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """检查 H x W x 5 abundance 的数值和逐像素 simplex 约束。"""

    if abundance.ndim != 3 or abundance.shape[2] != EXPECTED_ABUNDANCE_CHANNELS:
        raise ValueError(
            f"abundance 应为 HxW×{EXPECTED_ABUNDANCE_CHANNELS}，实际 shape={abundance.shape}"
        )
    if not np.isfinite(abundance).all():
        raise ValueError("abundance 包含 NaN 或 Inf，停止后续重构")

    pixel_sums = abundance.sum(axis=2, dtype=np.float32)
    deviation = np.abs(pixel_sums - 1.0)
    simplex_ok = bool(np.allclose(pixel_sums, 1.0, atol=atol, rtol=rtol))
    return {
        "abundance_shape": list(abundance.shape),
        "abundance_min": float(abundance.min()),
        "abundance_max": float(abundance.max()),
        "abundance_in_unit_interval": bool(
            np.all((abundance >= 0.0) & (abundance <= 1.0))
        ),
        "abundance_has_nan_or_inf": False,
        "abundance_sum_min": float(pixel_sums.min()),
        "abundance_sum_max": float(pixel_sums.max()),
        "abundance_sum_mean": float(pixel_sums.mean()),
        "abundance_sum_std": float(pixel_sums.std()),
        "abundance_sum_max_abs_error": float(deviation.max()),
        "abundance_simplex_ok": simplex_ok,
    }


def compute_mae(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """计算全体 HSI 元素的平均绝对误差。"""

    return float(np.mean(np.abs(reference.astype(np.float32) - reconstructed.astype(np.float32))))


def compute_mrae(reference: np.ndarray, reconstructed: np.ndarray, eps: float) -> float:
    """计算 MRAE，分母使用 abs(reference)+eps，避免零值除法。"""

    if eps <= 0:
        raise ValueError("MRAE epsilon 必须为正数")
    reference = reference.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    relative_error = np.abs(reference - reconstructed) / (np.abs(reference) + eps)
    return float(np.mean(relative_error))


def compute_metrics(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    mrae_eps: float,
) -> dict[str, float]:
    """计算项目验证所需的五项 HSI 重构指标。"""

    if reference.shape != reconstructed.shape:
        raise ValueError(
            f"reference 与 reconstructed shape 不一致："
            f"{reference.shape} vs {reconstructed.shape}"
        )
    if reference.ndim != 3 or reference.shape[2] != EXPECTED_HSI_CHANNELS:
        raise ValueError(
            f"HSI 应为 HxWx{EXPECTED_HSI_CHANNELS}，实际 shape={reference.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(reconstructed).all():
        raise ValueError("reference 或 reconstructed 包含 NaN/Inf")

    return {
        "RMSE": float(compare_rmse(reference, reconstructed)),
        "MAE": compute_mae(reference, reconstructed),
        "SAM_degree": float(compare_sam(reference, reconstructed)),
        "PSNR": float(compare_mpsnr(reference, reconstructed, data_range=1.0)),
        "MRAE": compute_mrae(reference, reconstructed, eps=mrae_eps),
    }


def display_rgb_pair(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    rgb_bands: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """用同一组 R/G/B 波段和共同显示范围生成两张 RGB uint8 图。"""

    if len(rgb_bands) != 3:
        raise ValueError(f"伪 RGB 必须包含 3 个波段，实际为 {rgb_bands}")
    reference_rgb = reference[:, :, list(rgb_bands)]
    reconstructed_rgb = reconstructed[:, :, list(rgb_bands)]
    finite_values = np.concatenate(
        [reference_rgb[np.isfinite(reference_rgb)], reconstructed_rgb[np.isfinite(reconstructed_rgb)]]
    )
    if finite_values.size == 0:
        raise ValueError("伪 RGB 中没有有限值")
    low = float(finite_values.min())
    high = float(finite_values.max())
    if high <= low:
        return np.zeros_like(reference_rgb, dtype=np.uint8), np.zeros_like(
            reconstructed_rgb, dtype=np.uint8
        )

    def to_uint8(image: np.ndarray) -> np.ndarray:
        normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
        return np.rint(normalized * 255.0).astype(np.uint8)

    return to_uint8(reference_rgb), to_uint8(reconstructed_rgb)


def build_rgb_pair(reference_rgb: np.ndarray, reconstructed_rgb: np.ndarray) -> np.ndarray:
    """将 original/reconstructed RGB 左右拼接为一张比较图。"""

    if reference_rgb.shape != reconstructed_rgb.shape:
        raise ValueError("两张 RGB 图的 shape 不一致")
    separator = np.full((reference_rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([reference_rgb, separator, reconstructed_rgb], axis=1)


def build_error_map(reference: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """将 59 个波段的绝对误差取均值，生成二维 absolute-error map。"""

    error = np.mean(np.abs(reference - reconstructed), axis=2).astype(np.float32)
    maximum = float(error.max())
    if maximum <= np.finfo(np.float32).eps:
        return np.zeros((*error.shape, 3), dtype=np.uint8)
    return colorize(error, cmap="viridis", vmin=0.0, vmax=maximum)


def select_pixel_positions(
    height: int,
    width: int,
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    """选择固定代表点并用固定随机种子补足像素位置。"""

    if count <= 0:
        raise ValueError("spectral curve 像素数必须为正数")
    if height <= 0 or width <= 0:
        raise ValueError(f"图像空间尺寸必须为正数，实际为 H={height}, W={width}")
    if count > height * width:
        raise ValueError(
            f"spectral curve 像素数 {count} 超过图像可用像素数 {height * width}"
        )
    fixed = [
        (height // 2, width // 2),
        (height // 4, width // 4),
        (height // 4, (3 * width) // 4),
        ((3 * height) // 4, width // 4),
        ((3 * height) // 4, (3 * width) // 4),
    ]
    positions: list[tuple[int, int]] = []
    for position in fixed:
        clipped = (
            min(max(position[0], 0), height - 1),
            min(max(position[1], 0), width - 1),
        )
        if clipped not in positions:
            positions.append(clipped)
        if len(positions) >= count:
            return positions[:count]

    rng = np.random.default_rng(seed)
    while len(positions) < count:
        candidate = (int(rng.integers(height)), int(rng.integers(width)))
        if candidate not in positions:
            positions.append(candidate)
    return positions


def save_spectral_curves(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Path,
    sample_seed: int,
    num_pixels: int,
) -> list[dict[str, int]]:
    """绘制若干固定/确定性随机像素的 original/reconstructed spectral curves。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, width, channels = reference.shape
    positions = select_pixel_positions(height, width, num_pixels, sample_seed)
    num_columns = 2
    num_rows = int(np.ceil(len(positions) / num_columns))
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(12, max(3.5 * num_rows, 4.0)),
        squeeze=False,
    )
    bands = np.arange(1, channels + 1)
    for axis, (row, column) in zip(axes.flat, positions):
        axis.plot(
            bands,
            reference[row, column, :],
            color="#1f77b4",
            linewidth=1.5,
            label="Original",
        )
        axis.plot(
            bands,
            reconstructed[row, column, :],
            color="#d62728",
            linewidth=1.3,
            linestyle="--",
            label="Reconstructed",
        )
        axis.set_title(f"pixel (y={row}, x={column})")
        axis.set_xlabel("Spectral band")
        axis.set_ylabel("HSI value")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    for axis in axes.flat[len(positions) :]:
        axis.axis("off")
    figure.suptitle("Original vs reconstructed spectral curves")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return [{"y": int(row), "x": int(column)} for row, column in positions]


def select_visual_indices(num_samples: int, count: int, seed: int) -> set[int]:
    """确定性选择代表性 validation sample，返回 DataLoader 索引集合。"""

    if count < 0:
        raise ValueError("代表性样本数不能为负数")
    if count == 0 or num_samples == 0:
        return set()
    if count >= num_samples:
        return set(range(num_samples))
    rng = np.random.default_rng(seed)
    return {int(index) for index in rng.choice(num_samples, size=count, replace=False)}


def finite_metric_summary(values: Iterable[float]) -> dict[str, Any]:
    """为一个指标计算 mean/std，并显式记录非有限值数量。"""

    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "mean": None,
        "std": None,
    }
    if finite.size:
        result["mean"] = float(finite.mean())
        result["std"] = float(finite.std())
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    """以标准 JSON 格式保存 summary，拒绝 NaN/Infinity。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def save_sample_outputs(
    sample_stem: str,
    abundance: np.ndarray,
    reference: np.ndarray,
    reconstructed: np.ndarray,
    output_dir: Path,
    sample_index: int,
    num_curve_pixels: int,
    rgb_bands: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    """保存单个样本的 MAT 与代表性可视化，并返回可追踪的输出信息。"""

    abundance_dir = output_dir / "abundance"
    reconstructed_dir = output_dir / "reconstructed_hsi"
    comparison_dir = output_dir / "comparison"
    curve_dir = output_dir / "spectral_curves"
    for directory in (abundance_dir, reconstructed_dir, comparison_dir, curve_dir):
        directory.mkdir(parents=True, exist_ok=True)

    abundance_path = abundance_dir / f"{sample_stem}.mat"
    reconstructed_path = reconstructed_dir / f"{sample_stem}.mat"
    sio.savemat(
        str(abundance_path),
        {"Abu": abundance.astype(np.float32)},
        do_compression=True,
    )
    # 同时保存完整 reference，避免重构结果脱离其逐样本比较对象。
    sio.savemat(
        str(reconstructed_path),
        {
            "HSI": reconstructed.astype(np.float32),
            "Reference": reference.astype(np.float32),
        },
        do_compression=True,
    )

    original_rgb, reconstructed_rgb = display_rgb_pair(
        reference,
        reconstructed,
        rgb_bands=rgb_bands,
    )
    original_rgb_path = comparison_dir / f"{sample_stem}_original_rgb.png"
    reconstructed_rgb_path = comparison_dir / f"{sample_stem}_reconstructed_rgb.png"
    pair_path = comparison_dir / f"{sample_stem}_rgb_pair.png"
    error_path = comparison_dir / f"{sample_stem}_absolute_error.png"
    write_png(original_rgb_path, original_rgb)
    write_png(reconstructed_rgb_path, reconstructed_rgb)
    write_png(pair_path, build_rgb_pair(original_rgb, reconstructed_rgb))
    write_png(error_path, build_error_map(reference, reconstructed))

    curve_path = curve_dir / f"{sample_stem}.png"
    curve_positions = save_spectral_curves(
        reference=reference,
        reconstructed=reconstructed,
        output_path=curve_path,
        sample_seed=seed + sample_index,
        num_pixels=num_curve_pixels,
    )
    # 复用现有 abundance 颜色配置，输出五张通道图和 all_channels.png。
    save_visualizations(
        abundance=abundance,
        mat_path=Path(f"{sample_stem}.mat"),
        output_dir=abundance_dir / "visualizations",
        cmap=DEFAULT_CMAP,
        vmin=DEFAULT_VMIN,
        vmax=DEFAULT_VMAX,
    )
    return {
        "abundance_path": str(abundance_path),
        "reconstructed_hsi_path": str(reconstructed_path),
        "original_rgb_path": str(original_rgb_path),
        "reconstructed_rgb_path": str(reconstructed_rgb_path),
        "absolute_error_path": str(error_path),
        "spectral_curve_path": str(curve_path),
        "spectral_curve_positions": curve_positions,
    }


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="在 Chikusei tests 集上验证 UnmixingAE 的 abundance 与 HSI reconstruction"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"tests HSI MAT 目录，默认：{DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="已经训练完成的 UnmixingAE checkpoint 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="推理设备；默认 auto",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=DEFAULT_N_BLOCKS,
        help=f"UnmixingAE SSPN block 数，必须匹配 checkpoint；默认：{DEFAULT_N_BLOCKS}",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker 数；默认 0，便于保持验证过程可追踪",
    )
    parser.add_argument(
        "--num-visual-samples",
        type=int,
        default=DEFAULT_NUM_VISUAL_SAMPLES,
        help=f"生成可视化的代表性样本数；默认：{DEFAULT_NUM_VISUAL_SAMPLES}",
    )
    parser.add_argument(
        "--num-curve-pixels",
        type=int,
        default=DEFAULT_NUM_CURVE_PIXELS,
        help=f"每个代表性样本绘制的像素光谱曲线数；默认：{DEFAULT_NUM_CURVE_PIXELS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"确定性选择代表性样本和像素的随机种子；默认：{DEFAULT_SEED}",
    )
    parser.add_argument(
        "--mrae-eps",
        type=float,
        default=DEFAULT_MRAE_EPS,
        help=f"MRAE 分母 epsilon；默认：{DEFAULT_MRAE_EPS}",
    )
    parser.add_argument(
        "--simplex-atol",
        type=float,
        default=DEFAULT_SIMPLEX_ATOL,
        help=f"abundance 和为 1 的绝对容差；默认：{DEFAULT_SIMPLEX_ATOL}",
    )
    parser.add_argument(
        "--simplex-rtol",
        type=float,
        default=DEFAULT_SIMPLEX_RTOL,
        help=f"abundance 和为 1 的相对容差；默认：{DEFAULT_SIMPLEX_RTOL}",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """执行完整 tests inference、reconstruction、metrics 与可视化流程。"""

    if args.n_blocks <= 0:
        raise ValueError("--n-blocks 必须为正整数")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能为负数")
    if args.num_visual_samples < 0:
        raise ValueError("--num-visual-samples 不能为负数")
    if args.num_curve_pixels <= 0:
        raise ValueError("--num-curve-pixels 必须为正整数")
    if args.seed < 0:
        raise ValueError("--seed 不能为负数")
    if args.mrae_eps <= 0:
        raise ValueError("--mrae-eps 必须为正数")
    if args.simplex_atol < 0 or args.simplex_rtol < 0:
        raise ValueError("simplex 容差不能为负数")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"tests 数据目录不存在：{args.input_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    state_dict, checkpoint_epoch = load_checkpoint_state(args.checkpoint, device)
    model = build_model(state_dict, n_blocks=args.n_blocks, device=device)
    model.eval()

    dataset = HSIDataset(
        image_dir=str(args.input_dir),
        augment=False,
        output_channels=EXPECTED_HSI_CHANNELS,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_indices = select_visual_indices(
        num_samples=len(dataset),
        count=args.num_visual_samples,
        seed=args.seed,
    )

    print(f"设备：{device}")
    print(f"tests 样本数：{len(dataset)}")
    print(f"checkpoint：{args.checkpoint}")
    if checkpoint_epoch is not None:
        print(f"checkpoint epoch：{checkpoint_epoch}")
    print(f"输出目录：{output_dir}")

    records: list[dict[str, Any]] = []
    visual_records: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for sample_index, (target_tensor, pseudo_rgb_tensor) in enumerate(loader):
            if target_tensor.ndim != 4 or pseudo_rgb_tensor.ndim != 4:
                raise ValueError(
                    "DataLoader 输出必须为 BCHW tensor："
                    f"target={tuple(target_tensor.shape)}, input={tuple(pseudo_rgb_tensor.shape)}"
                )
            if target_tensor.shape[1] != EXPECTED_HSI_CHANNELS:
                raise ValueError(
                    f"完整 HSI reference 通道数应为 {EXPECTED_HSI_CHANNELS}，"
                    f"实际为 {target_tensor.shape[1]}"
                )
            if pseudo_rgb_tensor.shape[1] != EXPECTED_INPUT_CHANNELS:
                raise ValueError(
                    f"encoder 输入通道数应为 {EXPECTED_INPUT_CHANNELS}，"
                    f"实际为 {pseudo_rgb_tensor.shape[1]}"
                )

            target_tensor = target_tensor.to(device, non_blocking=device.type == "cuda")
            pseudo_rgb_tensor = pseudo_rgb_tensor.to(
                device,
                non_blocking=device.type == "cuda",
            )
            abundance_tensor, reconstructed_tensor, decoder_weight = forward_with_cudnn_fallback(
                model,
                pseudo_rgb_tensor,
            )
            if abundance_tensor.shape[1] != EXPECTED_ABUNDANCE_CHANNELS:
                raise ValueError(
                    f"encoder abundance 通道数应为 {EXPECTED_ABUNDANCE_CHANNELS}，"
                    f"实际为 {abundance_tensor.shape[1]}"
                )
            if reconstructed_tensor.shape != target_tensor.shape:
                raise ValueError(
                    "reconstructed HSI 与 reference 的 BCHW shape 不一致："
                    f"{tuple(reconstructed_tensor.shape)} vs {tuple(target_tensor.shape)}"
                )
            if decoder_weight.shape[0] != EXPECTED_HSI_CHANNELS or decoder_weight.shape[1] != EXPECTED_ABUNDANCE_CHANNELS:
                raise ValueError(
                    "forward 返回的 decoder 权重形状不符合当前 Chikusei 契约："
                    f"{tuple(decoder_weight.shape)}"
                )

            abundance = tensor_to_hwc(abundance_tensor[0], "abundance")
            reference = tensor_to_hwc(target_tensor[0], "reference HSI")
            reconstructed = tensor_to_hwc(reconstructed_tensor[0], "reconstructed HSI")
            abundance_stats = check_abundance(
                abundance,
                atol=args.simplex_atol,
                rtol=args.simplex_rtol,
            )
            if not abundance_stats["abundance_simplex_ok"]:
                print(
                    f"WARNING: sample {sample_index} 的 abundance 逐像素和不满足容差；"
                    f"最大偏差={abundance_stats['abundance_sum_max_abs_error']:.6e}"
                )
            if not np.isfinite(reference).all() or not np.isfinite(reconstructed).all():
                raise ValueError(f"sample {sample_index} 的 reference/reconstructed 包含 NaN/Inf")

            sample_path = Path(dataset.image_files[sample_index])
            sample_stem = sample_path.stem
            metrics = compute_metrics(reference, reconstructed, mrae_eps=args.mrae_eps)
            record: dict[str, Any] = {
                "sample_index": sample_index,
                "sample": sample_stem,
                "source_mat": str(sample_path),
                "reference_shape": list(reference.shape),
                "reconstructed_shape": list(reconstructed.shape),
                "reconstructed_min": float(reconstructed.min()),
                "reconstructed_max": float(reconstructed.max()),
                "reconstructed_has_nan_or_inf": False,
                **abundance_stats,
                **metrics,
                "visualized": sample_index in visual_indices,
            }

            if sample_index in visual_indices:
                visual_records[sample_stem] = save_sample_outputs(
                    sample_stem=sample_stem,
                    abundance=abundance,
                    reference=reference,
                    reconstructed=reconstructed,
                    output_dir=output_dir,
                    sample_index=sample_index,
                    num_curve_pixels=args.num_curve_pixels,
                    rgb_bands=CHIKUSEI_RGB_BANDS,
                    seed=args.seed,
                )

            # 非代表性样本仍保存 abundance 和 reconstructed HSI，满足逐样本复核要求。
            if sample_index not in visual_indices:
                abundance_dir = output_dir / "abundance"
                reconstructed_dir = output_dir / "reconstructed_hsi"
                abundance_dir.mkdir(parents=True, exist_ok=True)
                reconstructed_dir.mkdir(parents=True, exist_ok=True)
                sio.savemat(
                    str(abundance_dir / f"{sample_stem}.mat"),
                    {"Abu": abundance.astype(np.float32)},
                    do_compression=True,
                )
                sio.savemat(
                    str(reconstructed_dir / f"{sample_stem}.mat"),
                    {
                        "HSI": reconstructed.astype(np.float32),
                        "Reference": reference.astype(np.float32),
                    },
                    do_compression=True,
                )

            records.append(record)
            print(
                f"[{sample_index + 1}/{len(dataset)}] {sample_stem}："
                f"RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}, "
                f"SAM={metrics['SAM_degree']:.4f} degree, PSNR={metrics['PSNR']:.4f}, "
                f"MRAE={metrics['MRAE']:.6f}"
            )

    if not records:
        raise RuntimeError("tests 数据集为空，未生成任何验证结果")

    fieldnames = list(records[0].keys())
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    metric_names = ("RMSE", "MAE", "SAM_degree", "PSNR", "MRAE")
    metric_summary = {
        name: finite_metric_summary(record[name] for record in records)
        for name in metric_names
    }
    simplex_failures = sum(
        not bool(record["abundance_simplex_ok"]) for record in records
    )
    unit_interval_failures = sum(
        not bool(record["abundance_in_unit_interval"]) for record in records
    )
    summary: dict[str, Any] = {
        "input": {
            "directory": str(args.input_dir.resolve()),
            "mat_key": "Y",
            "num_samples": len(records),
            "file_order": "sorted by HSIDataset",
        },
        "preprocessing": {
            "raw_hsi_layout": "HWC",
            "raw_or_selected_band_count": EXPECTED_HSI_CHANNELS,
            "chikusei_band_slice_python": [
                CHIKUSEI_BAND_SLICE.start,
                CHIKUSEI_BAND_SLICE.stop,
            ],
            "encoder_rgb_bands_within_59_bands": list(CHIKUSEI_RGB_BANDS),
            "encoder_rgb_order": ["R", "G", "B"],
            "normalization": "core.loaddata.datanorm per sample over the complete selected HSI",
            "augmentation": "disabled; mode=0",
            "reference_tensor_layout": "BCHW during inference, HWC when saved",
            "reference_scale": "same [0,1] training target scale produced by HSIDataset",
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "epoch": checkpoint_epoch,
            "n_blocks": args.n_blocks,
            "strict_state_dict_loading": True,
            "decoder_source": "checkpoint model.decoderlayer.weight through UnmixingAE.forward",
        },
        "inference": {
            "device": str(device),
            "model_eval": True,
            "torch_no_grad": True,
            "optimizer_or_parameter_update": False,
            "input_channels": EXPECTED_INPUT_CHANNELS,
            "abundance_channels": EXPECTED_ABUNDANCE_CHANNELS,
            "output_hsi_channels": EXPECTED_HSI_CHANNELS,
            "decoder_output_postprocessing": "none",
        },
        "abundance_checks": {
            "simplex_atol": args.simplex_atol,
            "simplex_rtol": args.simplex_rtol,
            "simplex_failure_samples": simplex_failures,
            "unit_interval_failure_samples": unit_interval_failures,
            "all_samples_finite": all(
                not record["abundance_has_nan_or_inf"] for record in records
            ),
        },
        "metrics": metric_summary,
        "visualization": {
            "num_visual_samples": len(visual_records),
            "visualized_samples": sorted(visual_records),
            "pseudo_rgb_bands": list(CHIKUSEI_RGB_BANDS),
            "absolute_error_definition": "mean absolute error across 59 spectral bands per pixel",
            "spectral_curve_seed": args.seed,
            "num_curve_pixels": args.num_curve_pixels,
            "outputs": visual_records,
        },
        "outputs": {
            "root": str(output_dir.resolve()),
            "metrics_csv": str(metrics_path.resolve()),
            "summary_json": str((output_dir / "summary.json").resolve()),
        },
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"metrics.csv：{metrics_path}")
    print(f"summary.json：{summary_path}")


def main() -> None:
    """解析参数并执行验证。"""

    parser = build_parser()
    try:
        run(parser.parse_args())
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
