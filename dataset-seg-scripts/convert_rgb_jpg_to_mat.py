"""
将一个外部 RGB 类别目录中的 JPG/JPEG 图像转换为 Unmixing 推理所需的 MAT 文件。

主要处理：BGR 转 RGB、缩放到固定空间尺寸、转换为 [0, 1] 范围的
float32，并以 H x W x 3 布局保存到 MAT 变量 ``Y``。

运行环境：Python 3.10，依赖 NumPy、OpenCV 和 SciPy。

使用示例：
    python dataset-seg-scripts/convert_rgb_jpg_to_mat.py \
        --input-dir "D:\\RGBDataset\\forest" \
        --output-dir ./dataset/train \
        --size 256

默认不会覆盖已有同名 MAT 文件；确认需要覆盖时添加 ``--overwrite``。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import scipy.io as sio


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}
MAT_KEY = "Y"
DEFAULT_SIZE = 256


def list_jpg_files(input_dir: Path) -> list[Path]:
    """按文件名排序并返回当前目录中的 JPG/JPEG 文件，不递归子目录。"""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    image_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_files:
        raise RuntimeError(f"输入目录中没有 JPG/JPEG 文件：{input_dir}")
    return image_files


def validate_output_names(image_files: Iterable[Path], output_dir: Path, overwrite: bool) -> None:
    """在写入前检查文件名冲突，避免只转换部分数据后才失败。"""

    output_names: set[str] = set()
    duplicate_names: list[str] = []
    existing_files: list[Path] = []

    for image_path in image_files:
        output_name = f"{image_path.stem}.mat"
        normalized_name = output_name.casefold()
        if normalized_name in output_names:
            duplicate_names.append(output_name)
        output_names.add(normalized_name)

        output_path = output_dir / output_name
        if output_path.exists() and not overwrite:
            existing_files.append(output_path)

    if duplicate_names:
        names = ", ".join(sorted(set(duplicate_names)))
        raise RuntimeError(f"JPG/JPEG 文件转换后会产生同名 MAT 文件：{names}")
    if existing_files:
        preview = ", ".join(str(path) for path in existing_files[:3])
        raise FileExistsError(
            f"输出目录中已有 {len(existing_files)} 个同名 MAT 文件，例如：{preview}。"
            "如需覆盖，请添加 --overwrite。"
        )


def load_and_convert_image(image_path: Path, size: int) -> np.ndarray:
    """读取图像并返回 size x size x 3、RGB、float32、[0, 1] 数组。"""

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV 无法读取图像：{image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    if (height, width) != (size, size):
        interpolation = cv2.INTER_AREA if height >= size and width >= size else cv2.INTER_CUBIC
        rgb = cv2.resize(rgb, (size, size), interpolation=interpolation)

    rgb_float = rgb.astype(np.float32) / np.float32(255.0)
    if rgb_float.shape != (size, size, 3):
        raise RuntimeError(
            f"转换后的图像形状错误：{rgb_float.shape}，预期 {(size, size, 3)}"
        )
    if not np.isfinite(rgb_float).all():
        raise ValueError(f"转换后的图像包含 NaN 或 Inf：{image_path}")
    return np.ascontiguousarray(rgb_float, dtype=np.float32)


def convert_directory(input_dir: Path, output_dir: Path, size: int, overwrite: bool) -> int:
    """转换单个目录中的全部 JPG/JPEG，并返回成功转换的文件数。"""

    image_files = list_jpg_files(input_dir)
    validate_output_names(image_files, output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_files, start=1):
        rgb_float = load_and_convert_image(image_path, size)
        output_path = output_dir / f"{image_path.stem}.mat"
        sio.savemat(str(output_path), {MAT_KEY: rgb_float}, do_compression=True)
        if index == 1 or index % 100 == 0 or index == len(image_files):
            print(f"[{index}/{len(image_files)}] {image_path.name} -> {output_path.name}")

    return len(image_files)


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="将单个类别目录中的 JPG/JPEG 转换为 Unmixing 所需的 float32 MAT 文件"
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="JPG/JPEG 输入目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="MAT 输出目录")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="输出边长，默认 256")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有同名 MAT 文件")
    return parser


def main() -> None:
    """解析参数、执行转换并打印汇总信息。"""

    args = build_parser().parse_args()
    if args.size <= 0:
        raise ValueError("--size 必须为正整数")

    count = convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=args.size,
        overwrite=args.overwrite,
    )
    print(
        f"转换完成：{count} 个文件；输出形状：{args.size}x{args.size}x3；"
        f"数据类型：float32；MAT 变量名：{MAT_KEY}"
    )


if __name__ == "__main__":
    main()
