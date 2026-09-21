"""将分类目录下的 HSRS-SC TIFF 转为现有 UnmixingAE 可验证的 59 波段 MAT。

环境：Python 3.9+，numpy、scipy、rasterio。
安装：python -m pip install numpy scipy rasterio
用法（在项目根目录执行，将引号中的路径换成自己的数据集根目录）：
    python dataset-seg-scripts/prepare_hsrs_sc.py --input-dir "D:/Data/HSRS-SC"
    python dataset-seg-scripts/prepare_hsrs_sc.py --input-dir "/data/HSRS-SC" --output-dir ./dataset/hsrs_sc_tests
也可修改下方 INPUT_DIR，随后直接运行本脚本。

输入：递归查找 .tif/.tiff（忽略后缀大小写），要求每幅为 48 波段、256x256。
假定 TIFF 保留 HSRS-SC 原始波段顺序；三通道预览图不属于有效输入。
输出：平铺保存 Y:256x256x59、float32，附带类别、来源和波长信息。
光谱处理：48 个真实中心波长 -> Chikusei [7:66] 的 59 个中心波长，线性插值；
不进行空间裁剪/缩放，不做归一化（后续 HSIDataset 复用训练时的逐样本归一化）。
读取 TIFF 声明的 scale/offset；有 NoData/NaN/Inf 时停止并报告具体文件。

默认处理全部样本；尚未核实论文使用的 700 幅名单及插值算法，不声称严格复现。
默认输出 dataset/tests；为防止旧 Chikusei 测试样本混入，拒绝非空输出目录。
失败时已完成的 MAT 会保留；重跑请选择另一空目录或自行处理已有输出。

波长依据：
1. HSRS-SC 原始数据论文，表3，DOI:10.11834/jig.200835：
   https://www.cjig.cn/zh/article/doi/10.11834/jig.200835/?viewType=HTML
2. Chikusei ENVI 原始头文件（公开镜像；本项目此前读取的头文件记录）：
   https://huggingface.co/datasets/danaroth/chikusei/blob/main/HyperspecVNIR_Chikusei_20140729.hdr
   以下目标值逐项取自原始表 [7:66]，单位由 um 转换为 nm，未用 linspace 近似。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


# 只需在此粘贴本地/远程数据集根目录；命令行 --input-dir 优先。
INPUT_DIR = None  # 例如 Path(r"D:/Data/HSRS-SC")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "dataset" / "tests"

# 表3按原始第1--48波段排列；不是在380--1050之间均匀生成的波长。
HSRS_WAVELENGTHS_NM = np.array([
    382.5, 396.9, 411.3, 425.7, 440.0, 454.4, 468.7, 483.1,
    497.4, 511.8, 526.1, 540.4, 554.7, 569.0, 583.3, 597.6,
    611.9, 626.2, 640.5, 654.8, 669.1, 683.4, 697.7, 712.0,
    726.3, 740.6, 754.9, 769.1, 783.4, 797.7, 812.0, 826.3,
    840.6, 854.9, 869.2, 883.5, 897.8, 912.1, 926.4, 940.7,
    955.1, 969.4, 983.7, 998.1, 1012.4, 1026.8, 1041.1, 1055.5,
], dtype=np.float64)

CHIKUSEI_WAVELENGTHS_NM = np.array([
    398.71, 403.87, 409.03, 414.19, 419.36, 424.52, 429.68,
    434.84, 440.00, 445.16, 450.32, 455.48, 460.64, 465.80,
    470.96, 476.12, 481.29, 486.45, 491.61, 496.77, 501.93,
    507.09, 512.25, 517.41, 522.57, 527.73, 532.89, 538.06,
    543.21, 548.38, 553.54, 558.70, 563.86, 569.02, 574.18,
    579.34, 584.50, 589.66, 594.83, 599.99, 605.14, 610.31,
    615.47, 620.63, 625.79, 630.95, 636.11, 641.27, 646.43,
    651.59, 656.75, 661.92, 667.07, 672.24, 677.40, 682.56,
    687.72, 692.88, 698.04,
], dtype=np.float64)


def find_tiffs(root: Path) -> list[Path]:
    """按相对路径排序递归查找样本，确定跨次运行一致的编号。"""
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{root}")
    paths = sorted(
        (path for path in root.rglob("*")
         if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise ValueError(f"未找到 .tif/.tiff 文件：{root}")
    return paths


def read_hsi(path: Path):
    """由 rasterio 统一按 CHW 读取多波段 TIFF，再转换为 HWC。

    rasterio 处理 TIFF 的波段存储布局，避免把第一页或 RGB 预览误当光谱。
    有无效像素时停止，因为现有验证器不支持有效像素掩码；填零会污染指标。
    """
    import rasterio

    with rasterio.open(path) as source:
        shape = (source.height, source.width, source.count)
        if shape != (256, 256, 48):
            raise ValueError(
                f"{path}: 预期 HxWxC=(256, 256, 48)，实际 {shape}。"
                "请确认输入为原始高光谱 TIFF，而非 RGB 预览图。"
            )
        pixels = source.read(masked=True, out_dtype="float32")
        if np.ma.getmaskarray(pixels).any():
            raise ValueError(f"{path}: 含 TIFF 标记的无效像素；停止，未填零或删除样本")
        # 默认 scale=1、offset=0；若文件带辐射缩放元数据则使用其逐波段值。
        scales = np.asarray(source.scales, dtype=np.float32)
        offsets = np.asarray(source.offsets, dtype=np.float32)
        cube = np.asarray(pixels, dtype=np.float32) * scales[:, None, None]
        cube += offsets[:, None, None]
        if not np.isfinite(cube).all():
            raise ValueError(f"{path}: 像素或 scale/offset 包含 NaN/Inf")
    return np.moveaxis(cube, 0, -1), scales, offsets


def align_spectrum(cube: np.ndarray) -> np.ndarray:
    """按波长作逐像素分段线性插值，保持原高宽与辐射值尺度。

    目标698.04 nm略高于HSRS第23波段697.7 nm，须保留第24波段712 nm
    作为插值右端点；先截掉所有700 nm以外波段会错误地引入外推。
    插值提高采样密度，不创造新的实测光谱分辨率。
    """
    if cube.shape != (256, 256, 48) or not np.isfinite(cube).all():
        raise ValueError(f"插值输入必须为有限值256x256x48，实际 {cube.shape}")
    source = HSRS_WAVELENGTHS_NM
    target = CHIKUSEI_WAVELENGTHS_NM
    if target[0] < source[0] or target[-1] > source[-1]:
        raise ValueError("目标波长超出源波长范围，不允许外推")
    right = np.searchsorted(source, target, side="right")
    left = right - 1
    weight = ((target - source[left]) / (source[right] - source[left])).astype(np.float32)
    aligned = cube[..., left] * (1.0 - weight) + cube[..., right] * weight
    aligned = np.ascontiguousarray(aligned, dtype=np.float32)
    if not np.isfinite(aligned).all():
        raise ValueError("光谱插值结果含 NaN/Inf")
    return aligned


def run(input_dir: Path, output_dir: Path) -> None:
    """逐文件转换到独立、平铺的 MAT 目录，供现有非递归验证加载器使用。"""
    from scipy.io import savemat

    root = input_dir.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    files = find_tiffs(root)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"输出必须是空目录或尚不存在：{output}。"
            "可用 --output-dir 指定新目录，避免混入旧数据；未删除任何文件。"
        )
    counts = Counter(path.relative_to(root).parts[0]
                     if len(path.relative_to(root).parts) > 1 else "unclassified"
                     for path in files)
    print(f"发现 {len(files)} 幅 TIFF；按全部样本处理（不是作者700幅名单）")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print("源波段顺序使用HSRS-SC论文表3；目标398.71--698.04 nm，共59波段")
    print(f"输出：{output}")

    for index, path in enumerate(files):
        # 先成功读取当前样本，再创建输出目录和文件；出错立即停在该样本。
        cube, scales, offsets = read_hsi(path)
        aligned = align_spectrum(cube)
        relative = path.relative_to(root)
        label = relative.parts[0] if len(relative.parts) > 1 else "unclassified"
        output.mkdir(parents=True, exist_ok=True)
        destination = output / f"hsrs_{index:05d}.mat"
        try:
            # 独占创建，防止运行期间意外覆盖同名文件。
            with destination.open("xb") as stream:
                savemat(stream, {
                    "Y": aligned,
                    "source_relative_path": relative.as_posix(),
                    "scene_class": label,
                    "wavelength_nm": CHIKUSEI_WAVELENGTHS_NM,
                    "source_wavelength_nm": HSRS_WAVELENGTHS_NM,
                    "tiff_scales": scales,
                    "tiff_offsets": offsets,
                    "spectral_resampling": "linear; no extrapolation",
                }, do_compression=False)
        except OSError as exc:
            raise OSError(f"写入失败：{destination}；请检查磁盘空间及该文件是否完整") from exc
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(files):
            print(f"[{index + 1}/{len(files)}] {relative} -> {destination.name}", flush=True)
    print(f"完成：{len(files)} 幅 Y:256x256x59 float32，可通过验证脚本 --input-dir 指向 {output}")


def main() -> None:
    """命令行入口；未给路径时提示用户粘贴数据集目录。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="HSRS-SC根目录，内部可含各类别子目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="输出MAT目录，默认项目dataset/tests")
    args = parser.parse_args()
    try:
        root = args.input_dir
        if root is None:
            entered = input("请粘贴HSRS-SC数据集根目录：").strip().strip('"').strip("'")
            if not entered:
                raise ValueError("输入路径不能为空")
            root = Path(entered)
        run(root, args.output_dir)
    except ImportError as exc:
        parser.error(f"缺少依赖 {exc.name}；请运行 python -m pip install numpy scipy rasterio")
    except (OSError, ValueError, EOFError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
