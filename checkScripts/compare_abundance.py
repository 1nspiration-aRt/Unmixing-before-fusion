"""
比较两组五通道 abundance，并生成严格一致的视觉结果。

主要功能：读取 Reference 和 Inferred 两组 MAT 文件中的 ``Abu`` 数据，
统一为 H x W x 5 后逐通道比较。脚本直接复用 ``visualize_abundance.py``
中的 MAT 读取、颜色映射、数值 clipping、单通道保存、五通道拼图和 PNG
写入逻辑，不重新执行 HSI→RGB 或 unmixing 网络。

显示配置固定沿用 ``visualize_abundance.py``：默认使用其自定义的 ``jet``
颜色色阶，显示范围固定为 [0, 1]。Reference 和 Inferred 使用完全相同的
显示尺度，不执行任何按样本或按通道的 min-max normalization。

运行环境：Python 3、NumPy、SciPy。脚本不依赖 Matplotlib 或 OpenCV。

单个 MAT 文件比较：
    python3 compare_abundance.py \
        --reference path/to/reference.mat \
        --inferred path/to/inferred.mat \
        --output-dir experiments/abundance_compare

两个目录按同名 MAT 文件批量比较：
    python3 compare_abundance.py \
        --reference dataset/reference_abu \
        --inferred dataset/inferred_abu \
        --output-dir experiments/abundance_compare

也可以指定 MAT 键和布局；默认键为 ``Abu``，默认布局为 ``auto``：
    python3 compare_abundance.py \
        --reference path/to/reference.mat \
        --inferred path/to/inferred.mat \
        --key Abu \
        --layout auto \
        --output-dir experiments/abundance_compare

单个文件模式的输出结构为：
    output-dir/
    ├── comparison.png
    ├── Reference/
    │   ├── channel_1.png ... channel_5.png
    │   └── all_channels.png
    └── Inferred/
        ├── channel_1.png ... channel_5.png
        └── all_channels.png

批量模式会在 ``output-dir`` 下为每个同名 MAT 文件建立一个以文件 stem
命名的子目录。``comparison.png`` 为 2×5 布局：第一行是 Reference 的
Channel 1～5，第二行是 Inferred 的 Channel 1～5。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from visualize_abundance import (
    COLOR_STOPS,
    DEFAULT_CHANNELS,
    DEFAULT_CMAP,
    DEFAULT_KEY,
    DEFAULT_VMAX,
    DEFAULT_VMIN,
    colorize,
    list_mat_files,
    load_abundance,
    save_visualizations,
    write_png,
)


# 比较任务固定为五通道；该值直接沿用 visualize_abundance.py 的常量。
COMPARISON_CHANNELS = DEFAULT_CHANNELS


def resolve_input_pairs(
    reference_input: Path,
    inferred_input: Path,
) -> list[tuple[Path, Path]]:
    """解析单文件或双目录输入，并返回严格的一一对应 MAT 文件对。

    单文件模式要求两个输入都为 MAT 文件；批量模式要求两个输入都为目录，
    并按照不含路径的完整文件名进行配对。两目录的 MAT 文件名集合必须完全
    一致，避免静默跳过缺失样本。
    """

    reference_is_file = reference_input.is_file()
    inferred_is_file = inferred_input.is_file()
    reference_is_dir = reference_input.is_dir()
    inferred_is_dir = inferred_input.is_dir()

    if reference_is_file and inferred_is_file:
        # list_mat_files 负责复用现有的 .mat 后缀校验。
        reference_files = list_mat_files(reference_input)
        inferred_files = list_mat_files(inferred_input)
        return [(reference_files[0], inferred_files[0])]

    if reference_is_dir and inferred_is_dir:
        reference_files = list_mat_files(reference_input)
        inferred_files = list_mat_files(inferred_input)

        reference_by_name = {path.name: path for path in reference_files}
        inferred_by_name = {path.name: path for path in inferred_files}
        reference_names = set(reference_by_name)
        inferred_names = set(inferred_by_name)

        missing_inferred = sorted(reference_names - inferred_names)
        missing_reference = sorted(inferred_names - reference_names)
        if missing_inferred or missing_reference:
            details: list[str] = []
            if missing_inferred:
                details.append(
                    "Inferred 目录缺少：" + ", ".join(missing_inferred)
                )
            if missing_reference:
                details.append(
                    "Reference 目录缺少：" + ", ".join(missing_reference)
                )
            raise ValueError("两目录的 MAT 文件无法完整按同名配对；" + "；".join(details))

        return [
            (reference_by_name[name], inferred_by_name[name])
            for name in sorted(reference_names)
        ]

    if not reference_is_file and not reference_is_dir:
        raise FileNotFoundError(f"Reference 输入路径不存在：{reference_input}")
    if not inferred_is_file and not inferred_is_dir:
        raise FileNotFoundError(f"Inferred 输入路径不存在：{inferred_input}")

    raise ValueError(
        "Reference 和 Inferred 必须使用相同的输入类型：同时为单个 MAT 文件，"
        "或同时为包含 MAT 文件的目录"
    )


def validate_display_configuration() -> None:
    """确认比较脚本所依赖的显示常量仍来自现有可视化脚本。"""

    if COMPARISON_CHANNELS != 5:
        raise RuntimeError(
            "visualize_abundance.py 的 DEFAULT_CHANNELS 已不是 5，"
            "无法执行固定五通道比较"
        )
    if DEFAULT_CMAP not in COLOR_STOPS:
        raise RuntimeError(
            f"visualize_abundance.py 中不存在默认颜色配置：{DEFAULT_CMAP!r}"
        )
    if DEFAULT_VMAX <= DEFAULT_VMIN:
        raise RuntimeError(
            "visualize_abundance.py 的默认显示范围无效："
            f"[{DEFAULT_VMIN}, {DEFAULT_VMAX}]"
        )


def print_abundance_stats(
    label: str,
    mat_path: Path,
    abundance: np.ndarray,
) -> None:
    """打印一组 abundance 的形状、范围和非有限值状态。"""

    has_nan_or_inf = bool(not np.isfinite(abundance).all())
    print(
        f"{label}：{mat_path}；shape={abundance.shape}；"
        f"min={abundance.min():.6f}；max={abundance.max():.6f}；"
        f"是否存在 NaN/Inf：{'是' if has_nan_or_inf else '否'}"
    )


def validate_spatial_shape(
    reference: np.ndarray,
    inferred: np.ndarray,
    reference_path: Path,
    inferred_path: Path,
) -> None:
    """检查两组 abundance 的空间尺寸和通道数，禁止任何空间变换。"""

    if reference.shape[2] != COMPARISON_CHANNELS:
        raise ValueError(
            f"Reference {reference_path} 的通道数错误："
            f"实际形状为 {reference.shape}，预期为 5 通道"
        )
    if inferred.shape[2] != COMPARISON_CHANNELS:
        raise ValueError(
            f"Inferred {inferred_path} 的通道数错误："
            f"实际形状为 {inferred.shape}，预期为 5 通道"
        )
    if reference.shape[:2] != inferred.shape[:2]:
        raise ValueError(
            "Reference 和 Inferred 的空间尺寸不一致："
            f"Reference={reference.shape[:2]}，Inferred={inferred.shape[:2]}；"
            "禁止 resize、crop 或 interpolate"
        )


def build_channel_images(abundance: np.ndarray) -> list[np.ndarray]:
    """按 Channel 1→5 顺序调用现有 colorize() 生成 RGB 图像。"""

    # 这里严格使用原数组的通道索引，不做任何排序、匹配或重新排列。
    return [
        colorize(
            abundance[:, :, channel_index],
            cmap=DEFAULT_CMAP,
            vmin=DEFAULT_VMIN,
            vmax=DEFAULT_VMAX,
        )
        for channel_index in range(COMPARISON_CHANNELS)
    ]


def join_channel_row(channel_images: list[np.ndarray]) -> np.ndarray:
    """使用与现有 all_channels.png 相同的白色分隔条横向拼接五个通道。"""

    if len(channel_images) != COMPARISON_CHANNELS:
        raise ValueError(
            f"比较图必须包含 {COMPARISON_CHANNELS} 个通道，"
            f"实际收到 {len(channel_images)} 个"
        )

    height = channel_images[0].shape[0]
    separator = np.full(
        (height, 4, 3),
        fill_value=255,
        dtype=np.uint8,
    )
    row_parts: list[np.ndarray] = []
    for channel_index, channel_image in enumerate(channel_images):
        if channel_image.shape != channel_images[0].shape:
            raise ValueError("同一行的通道图像尺寸不一致，无法生成比较图")
        if channel_index:
            row_parts.append(separator)
        row_parts.append(channel_image)
    return np.concatenate(row_parts, axis=1)


def build_comparison_image(
    reference: np.ndarray,
    inferred: np.ndarray,
) -> np.ndarray:
    """生成第一行 Reference、第二行 Inferred 的 2×5 RGB 比较图。"""

    reference_row = join_channel_row(build_channel_images(reference))
    inferred_row = join_channel_row(build_channel_images(inferred))
    if reference_row.shape[1] != inferred_row.shape[1]:
        raise ValueError("Reference 和 Inferred 的比较图宽度不一致")

    # 采用与现有横向分隔条一致的 4 像素白色分隔条区分两行。
    row_separator = np.full(
        (4, reference_row.shape[1], 3),
        fill_value=255,
        dtype=np.uint8,
    )
    return np.concatenate(
        [reference_row, row_separator, inferred_row],
        axis=0,
    )


def save_pair_visualizations(
    reference: np.ndarray,
    inferred: np.ndarray,
    pair_output_dir: Path,
) -> None:
    """保存两组单通道图、各自拼图以及最终 2×5 比较图。"""

    # save_visualizations() 按 mat_path.stem 建立子目录。使用固定标签可以让
    # 输出目录明确区分 Reference 和 Inferred，同时完整复用其保存实现。
    save_visualizations(
        abundance=reference,
        mat_path=Path("Reference.mat"),
        output_dir=pair_output_dir,
        cmap=DEFAULT_CMAP,
        vmin=DEFAULT_VMIN,
        vmax=DEFAULT_VMAX,
    )
    save_visualizations(
        abundance=inferred,
        mat_path=Path("Inferred.mat"),
        output_dir=pair_output_dir,
        cmap=DEFAULT_CMAP,
        vmin=DEFAULT_VMIN,
        vmax=DEFAULT_VMAX,
    )

    comparison_path = pair_output_dir / "comparison.png"
    write_png(comparison_path, build_comparison_image(reference, inferred))
    print(f"comparison.png 输出路径：{comparison_path}")
    print(f"Reference 图像输出目录：{pair_output_dir / 'Reference'}")
    print(f"Inferred 图像输出目录：{pair_output_dir / 'Inferred'}")


def process_pair(
    reference_path: Path,
    inferred_path: Path,
    pair_output_dir: Path,
    key: str,
    layout: str,
) -> None:
    """读取、检查并处理一个 Reference/Inferred MAT 文件对。"""

    reference = load_abundance(
        mat_path=reference_path,
        key=key,
        expected_channels=COMPARISON_CHANNELS,
        layout=layout,
    )
    inferred = load_abundance(
        mat_path=inferred_path,
        key=key,
        expected_channels=COMPARISON_CHANNELS,
        layout=layout,
    )

    # load_abundance() 已经统一输出为 HWC 并拒绝 NaN/Inf；这里仍显式打印
    # 状态，满足比较运行时的可追溯性要求。
    print_abundance_stats("Reference abundance", reference_path, reference)
    print_abundance_stats("Inferred abundance", inferred_path, inferred)
    validate_spatial_shape(reference, inferred, reference_path, inferred_path)

    pair_output_dir.mkdir(parents=True, exist_ok=True)
    save_pair_visualizations(reference, inferred, pair_output_dir)


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="严格使用 visualize_abundance.py 配置比较两组五通道 abundance"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="原始 HSI 得到的 Reference MAT 文件，或 MAT 文件目录",
    )
    parser.add_argument(
        "--inferred",
        type=Path,
        required=True,
        help="伪 RGB 经 unmixing 网络得到的 Inferred MAT 文件，或 MAT 文件目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="比较结果输出目录",
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help=f"MAT 数据键名，默认：{DEFAULT_KEY}",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "hwc", "chw"),
        default="auto",
        help="输入数组布局；默认 auto，输出统一为 HWC",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """解析输入配对并生成全部比较结果。"""

    validate_display_configuration()
    pairs = resolve_input_pairs(args.reference, args.inferred)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 单文件模式直接把 comparison.png 放在 output-dir 根目录；批量模式为
    # 每个同名 MAT 文件建立独立子目录，避免不同样本互相覆盖。
    batch_mode = args.reference.is_dir() and args.inferred.is_dir()
    output_dirs: set[Path] = set()
    for index, (reference_path, inferred_path) in enumerate(pairs, start=1):
        pair_output_dir = (
            args.output_dir / reference_path.stem if batch_mode else args.output_dir
        )
        if pair_output_dir in output_dirs:
            raise ValueError(
                f"多个 MAT 文件映射到同一输出目录，无法安全保存：{pair_output_dir}"
            )
        output_dirs.add(pair_output_dir)

        print(f"[{index}/{len(pairs)}] 开始比较：{reference_path.name}")
        process_pair(
            reference_path=reference_path,
            inferred_path=inferred_path,
            pair_output_dir=pair_output_dir,
            key=args.key,
            layout=args.layout,
        )


def main() -> None:
    """解析参数并执行比较。"""

    parser = build_parser()
    try:
        run(parser.parse_args())
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
