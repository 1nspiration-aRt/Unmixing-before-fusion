"""
可视化五通道丰度图。

主要功能：读取一个 MAT 文件或目录中的多个 MAT 文件，提取指定键对应的
五通道丰度图，分别保存每个通道的伪彩色图，并额外保存一张五通道拼图。
默认按照当前 Step 1 推理输出约定读取 ``Abu``，数据布局为 H x W x 5，
数值显示范围固定为 [0, 1]。

运行环境：Python 3、NumPy、SciPy。脚本直接写入 PNG，不依赖 Matplotlib
或桌面显示环境。

运行示例：
    python3 visualize_abundance.py \
        --input dataset/inferred_abu \
        --output-dir experiments/abundance_vis

也可以直接可视化单个文件，或指定其他 MAT 键：
    python3 visualize_abundance.py \
        --input "ground truth.mat" \
        --key Abu \
        --layout auto \
        --output-dir experiments/ground_truth_abundance_vis
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import zlib

import numpy as np


DEFAULT_KEY = "Abu"
DEFAULT_CHANNELS = 5
DEFAULT_CMAP = "viridis"
DEFAULT_VMIN = 0.0
DEFAULT_VMAX = 1.0

# 使用少量固定色阶插值生成伪彩色图，避免为简单可视化额外依赖绘图库。
COLOR_STOPS = {
    "viridis": np.asarray(
        [
            (68, 1, 84),
            (59, 82, 139),
            (33, 145, 140),
            (94, 201, 98),
            (253, 231, 37),
        ],
        dtype=np.float32,
    ),
    "magma": np.asarray(
        [
            (0, 0, 4),
            (81, 18, 124),
            (182, 54, 121),
            (251, 140, 60),
            (252, 253, 191),
        ],
        dtype=np.float32,
    ),
    "gray": np.asarray(
        [(0, 0, 0), (255, 255, 255)],
        dtype=np.float32,
    ),
}


def list_mat_files(input_path: Path) -> list[Path]:
    """解析单个 MAT 文件或目录，并按文件名排序返回输入文件。"""

    if input_path.is_file():
        if input_path.suffix.lower() != ".mat":
            raise ValueError(f"输入文件必须是 .mat 文件：{input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    mat_files = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".mat"
    )
    if not mat_files:
        raise RuntimeError(f"输入目录中没有 .mat 文件：{input_path}")
    return mat_files


def load_abundance(
    mat_path: Path,
    key: str,
    expected_channels: int,
    layout: str,
) -> np.ndarray:
    """读取并统一为 H x W x C 的丰度数组。"""

    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "读取 MAT 文件需要 SciPy，请在运行环境中安装 scipy。"
        ) from exc

    try:
        mat_data = loadmat(str(mat_path))
    except NotImplementedError as exc:
        raise RuntimeError(
            f"无法读取 {mat_path}：该文件可能是 MATLAB v7.3 格式，"
            "当前脚本需要先转换为普通 MAT 格式。"
        ) from exc

    if key not in mat_data:
        available_keys = sorted(
            name for name in mat_data if not name.startswith("__")
        )
        raise KeyError(
            f"{mat_path} 中缺少键 {key!r}；可用数据键：{available_keys}"
        )

    abundance = np.asarray(mat_data[key], dtype=np.float32)
    if abundance.ndim != 3:
        raise ValueError(
            f"{mat_path} 的 {key!r} 必须是三维数组，实际形状为 {abundance.shape}"
        )

    if layout == "hwc":
        if abundance.shape[2] != expected_channels:
            raise ValueError(
                f"{mat_path} 的 {key!r} 按 HWC 解释时通道数错误："
                f"实际形状 {abundance.shape}，预期最后一维为 {expected_channels}"
            )
    elif layout == "chw":
        if abundance.shape[0] != expected_channels:
            raise ValueError(
                f"{mat_path} 的 {key!r} 按 CHW 解释时通道数错误："
                f"实际形状 {abundance.shape}，预期第一维为 {expected_channels}"
            )
        abundance = abundance.transpose(1, 2, 0)
    else:
        # auto 模式优先识别当前工程使用的 HWC；若只有第一维等于通道数，
        # 则按常见深度学习张量布局 CHW 转换。两端同时匹配时拒绝猜测。
        first_matches = abundance.shape[0] == expected_channels
        last_matches = abundance.shape[2] == expected_channels
        if first_matches and last_matches:
            raise ValueError(
                f"{mat_path} 的形状 {abundance.shape} 在 auto 模式下存在布局歧义，"
                "请显式指定 --layout hwc 或 --layout chw"
            )
        if last_matches:
            pass
        elif first_matches:
            abundance = abundance.transpose(1, 2, 0)
        else:
            raise ValueError(
                f"{mat_path} 的 {key!r} 无法识别为 {expected_channels} 通道丰度图："
                f"实际形状为 {abundance.shape}"
            )

    if not np.isfinite(abundance).all():
        raise ValueError(f"{mat_path} 的 {key!r} 包含 NaN 或 Inf")
    return np.ascontiguousarray(abundance, dtype=np.float32)


def colorize(values: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    """按照固定范围把二维丰度图转换为 RGB uint8 图像。"""

    normalized = np.clip((values.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    stops = COLOR_STOPS[cmap]
    positions = normalized * (len(stops) - 1)
    lower_index = np.floor(positions).astype(np.int32)
    upper_index = np.minimum(lower_index + 1, len(stops) - 1)
    weight = (positions - lower_index)[..., np.newaxis]
    rgb = stops[lower_index] * (1.0 - weight) + stops[upper_index] * weight
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def write_png(path: Path, image: np.ndarray) -> None:
    """使用 Python 标准库把 H x W x 3 uint8 数组写成 RGB PNG。"""

    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"PNG 图像必须是 HxWx3，实际形状为 {image.shape}")
    image = np.ascontiguousarray(image)
    height, width, _ = image.shape

    def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
        chunk = chunk_type + payload
        checksum = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + chunk + struct.pack(">I", checksum)

    # 每一行前的过滤器字节设为 0，适用于直接写入未过滤的 RGB 数据。
    raw_rows = b"".join(b"\x00" + row.tobytes() for row in image)
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
    )
    png += png_chunk(b"IDAT", zlib.compress(raw_rows, level=6))
    png += png_chunk(b"IEND", b"")
    path.write_bytes(png)


def save_visualizations(
    abundance: np.ndarray,
    mat_path: Path,
    output_dir: Path,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """保存单通道图和五通道拼图。"""

    sample_dir = output_dir / mat_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    outside_range = (abundance < vmin) | (abundance > vmax)
    outside_count = int(outside_range.sum())
    if outside_count:
        print(
            f"WARNING: {mat_path.name} 有 {outside_count} 个值超出显示范围 "
            f"[{vmin}, {vmax}]，图像中将被色标截断；原始 MAT 不会被修改。"
        )

    channel_count = abundance.shape[2]
    channel_images: list[np.ndarray] = []
    for channel_index in range(channel_count):
        channel_image = colorize(
            abundance[:, :, channel_index],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        channel_images.append(channel_image)
        channel_path = sample_dir / f"channel_{channel_index + 1}.png"
        write_png(channel_path, channel_image)

    # 各通道按 1→5 的顺序横向拼接，单通道文件名负责明确通道编号。
    separator = np.full(
        (abundance.shape[0], 4, 3),
        fill_value=255,
        dtype=np.uint8,
    )
    contact_sheet_parts: list[np.ndarray] = []
    for channel_index, channel_image in enumerate(channel_images):
        if channel_index:
            contact_sheet_parts.append(separator)
        contact_sheet_parts.append(channel_image)
    contact_sheet = np.concatenate(contact_sheet_parts, axis=1)
    write_png(sample_dir / "all_channels.png", contact_sheet)


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="提取并可视化 MAT 文件中的五通道丰度图"
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        required=True,
        help="单个 MAT 文件，或包含 MAT 文件的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="可视化结果输出目录",
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help=f"MAT 数据键名，默认：{DEFAULT_KEY}",
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help=f"期望的丰度通道数，默认：{DEFAULT_CHANNELS}",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "hwc", "chw"),
        default="auto",
        help="输入数组布局；默认 auto，输出统一为 HWC",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        choices=tuple(COLOR_STOPS),
        help=f"伪彩色方案，默认：{DEFAULT_CMAP}",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=DEFAULT_VMIN,
        help=f"显示下限，默认：{DEFAULT_VMIN}",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=DEFAULT_VMAX,
        help=f"显示上限，默认：{DEFAULT_VMAX}",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """读取输入文件并生成全部可视化结果。"""

    if args.expected_channels <= 0:
        raise ValueError("--expected-channels 必须为正整数")
    if args.vmax <= args.vmin:
        raise ValueError("--vmax 必须大于 --vmin")
    if args.cmap not in COLOR_STOPS:
        raise ValueError(f"不存在的伪彩色方案：{args.cmap}")

    mat_files = list_mat_files(args.input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, mat_path in enumerate(mat_files, start=1):
        abundance = load_abundance(
            mat_path=mat_path,
            key=args.key,
            expected_channels=args.expected_channels,
            layout=args.layout,
        )
        save_visualizations(
            abundance=abundance,
            mat_path=mat_path,
            output_dir=args.output_dir,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        print(
            f"[{index}/{len(mat_files)}] {mat_path.name} -> "
            f"{args.output_dir / mat_path.stem}；形状：{abundance.shape}；"
            f"范围：[{abundance.min():.6f}, {abundance.max():.6f}]"
        )


def main() -> None:
    """解析参数并执行可视化。"""

    parser = build_parser()
    try:
        run(parser.parse_args())
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
