"""
将原始 Chikusei HSI 切成 patch，并随机划分为训练、验证和测试集。

处理顺序：读取 HSI -> 选择 59 个波段 -> 生成全部 patch -> 随机打乱 ->
按比例分配到 trains/evals/tests -> 保存 MAT 文件和 manifest。

运行命令：
    python3 dataset-seg-scripts/split_chikusei.py
    python3 dataset-seg-scripts/split_chikusei.py --input /path/to/Chikusei.mat --dry-run

依赖：Python 3、NumPy、SciPy；读取 MATLAB v7.3 文件还需要 h5py。输出
patch 使用 H x W x C 布局，MAT 键为 Y；脚本保留原始数值，归一化由
core/loaddata.py 在训练时执行。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.io as sio
except ImportError:  # 未安装 SciPy 时仍允许查看 --help。
    sio = None

try:
    import h5py
except ImportError:  # 非 v7.3 文件不需要 h5py。
    h5py = None


# 用户可直接修改默认路径，也可通过 --input 和 --output 临时覆盖。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "Chikusei.mat"
OUTPUT_ROOT = PROJECT_ROOT / "dataset"

MAT_KEY = "Y"
RAW_SPECTRAL_CHANNELS = 128
CHIKUSEI_BAND_START = 7
CHIKUSEI_BAND_END = 66  # Python [7:66] 对应 MATLAB 第 8--66 波段，共 59 个。
PATCH_SIZE = 256
STRIDE = 256
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_SEED = 3000
DROP_INCOMPLETE_EDGE = True
OVERWRITE = False

SPLIT_NAMES = ("trains", "evals", "tests")
SPLIT_PREFIXES = {"trains": "train", "evals": "eval", "tests": "test"}
OUTPUT_CHANNELS = CHIKUSEI_BAND_END - CHIKUSEI_BAND_START


@dataclass(frozen=True)
class Patch:
    """一个 patch 在原始 HSI 中的空间范围。"""

    y0: int
    y1: int
    x0: int
    x1: int
    padded_bottom: int = 0
    padded_right: int = 0


def require_scipy():
    """返回 SciPy MAT 接口；依赖缺失时给出明确错误。"""

    if sio is None:
        raise RuntimeError("处理 MAT 文件需要 SciPy：python3 -m pip install scipy")
    return sio


def load_v73_variable(input_path: Path, mat_key: str) -> np.ndarray:
    """使用 HDF5 读取 MATLAB v7.3 数值数组并恢复 MATLAB 维度顺序。"""

    if h5py is None:
        raise RuntimeError(
            "读取 MATLAB v7.3 文件需要 h5py：python3 -m pip install h5py"
        )
    try:
        with h5py.File(str(input_path), "r") as mat_file:
            if mat_key not in mat_file:
                keys = [key for key in mat_file.keys() if not key.startswith("#")]
                raise KeyError(f"MAT 文件中没有变量 {mat_key!r}；可用变量：{keys}")
            dataset = mat_file[mat_key]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"MAT 变量 {mat_key!r} 不是数值数组")
            # MATLAB v7.3 在 HDF5 中按逆序保存数组维度。
            return np.asarray(dataset).transpose()
    except (KeyError, ValueError):
        raise
    except OSError as exc:
        raise RuntimeError(f"无法以 MATLAB v7.3/HDF5 格式读取：{input_path}") from exc


def parse_band_axis(value: str) -> Optional[int]:
    """解析光谱轴；auto 表示自动查找长度为 128 的维度。"""

    if value.lower() == "auto":
        return None
    try:
        axis = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("band-axis 必须是 auto、0、1 或 2") from exc
    if axis not in (0, 1, 2):
        raise argparse.ArgumentTypeError("band-axis 必须是 auto、0、1 或 2")
    return axis


def parse_ratios(value: str) -> Tuple[float, float, float]:
    """解析 trains,evals,tests 比例。"""

    try:
        ratios = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratios 格式示例：0.8,0.1,0.1") from exc
    if len(ratios) != 3 or any(not math.isfinite(item) or item <= 0 for item in ratios):
        raise argparse.ArgumentTypeError("ratios 必须包含三个正数")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise argparse.ArgumentTypeError("三个比例之和必须为 1")
    return ratios  # type: ignore[return-value]


def load_hsi(input_path: Path, mat_key: str, band_axis: Optional[int]) -> Tuple[np.ndarray, List[int]]:
    """读取 MAT，将布局统一为 H x W x 59，并尽量避免全 128 波段复制。"""

    mat_io = require_scipy()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入数据集不存在：{input_path}")
    try:
        mat_data = mat_io.loadmat(str(input_path))
        if mat_key not in mat_data:
            keys = [key for key in mat_data if not key.startswith("__")]
            raise KeyError(f"MAT 文件中没有变量 {mat_key!r}；可用变量：{keys}")
        cube = np.asarray(mat_data[mat_key])
    except NotImplementedError:
        cube = load_v73_variable(input_path, mat_key)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"无法读取 MAT 文件：{input_path}") from exc
    raw_shape = list(cube.shape)
    if cube.ndim != 3 or np.iscomplexobj(cube):
        raise ValueError(f"HSI 必须是三维实数数组，实际形状：{cube.shape}")

    if band_axis is None:
        candidates = [axis for axis, size in enumerate(cube.shape) if size == RAW_SPECTRAL_CHANNELS]
        if len(candidates) != 1:
            raise ValueError(f"无法从形状 {cube.shape} 唯一识别 128 波段轴，请设置 --band-axis")
        band_axis = candidates[0]
    if cube.shape[band_axis] != RAW_SPECTRAL_CHANNELS:
        raise ValueError(f"光谱轴长度应为 128，实际为 {cube.shape[band_axis]}")

    cube_hwc = np.moveaxis(cube, band_axis, -1)
    selected = cube_hwc[:, :, CHIKUSEI_BAND_START:CHIKUSEI_BAND_END]
    try:
        hsi = np.ascontiguousarray(selected, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("HSI 必须是数值数组") from exc
    if hsi.shape[2] != OUTPUT_CHANNELS or not np.isfinite(hsi).all():
        raise ValueError(f"选择后的 HSI 应为有限值 H x W x {OUTPUT_CHANNELS} 数组")
    return hsi, raw_shape


def patch_starts(length: int, patch_size: int, stride: int, include_edge: bool) -> List[int]:
    """生成单个空间轴上的 patch 起点。"""

    if include_edge:
        return list(range(0, length, stride))
    if length < patch_size:
        return []
    return list(range(0, length - patch_size + 1, stride))


def build_patches(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
    include_edge: bool,
) -> List[Patch]:
    """先为整幅图像生成全部 patch 坐标。"""

    patches: List[Patch] = []
    for y0 in patch_starts(height, patch_size, stride, include_edge):
        for x0 in patch_starts(width, patch_size, stride, include_edge):
            y1 = min(y0 + patch_size, height)
            x1 = min(x0 + patch_size, width)
            patches.append(
                Patch(
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    padded_bottom=patch_size - (y1 - y0),
                    padded_right=patch_size - (x1 - x0),
                )
            )
    if len(patches) < len(SPLIT_NAMES):
        raise ValueError(f"只生成了 {len(patches)} 个 patch，无法划分三个非空数据集")
    return patches


def allocate_counts(total: int, ratios: Sequence[float]) -> List[int]:
    """按照最大余数法分配数量，并保证三个集合均非空。"""

    raw = np.asarray(ratios, dtype=np.float64) * total
    counts = np.floor(raw).astype(int)
    counts = np.maximum(counts, 1)

    while int(counts.sum()) > total:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        counts[max(candidates, key=lambda i: counts[i] - raw[i])] -= 1
    while int(counts.sum()) < total:
        counts[max(range(len(ratios)), key=lambda i: raw[i] - counts[i])] += 1
    return [int(count) for count in counts]


def split_patches(
    patches: Sequence[Patch],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[Patch]]:
    """随机打乱全部 patch，并按比例分配到三个集合。"""

    shuffled = list(patches)
    np.random.default_rng(seed).shuffle(shuffled)
    train_count, eval_count, _ = allocate_counts(len(shuffled), ratios)
    train_end = train_count
    eval_end = train_end + eval_count
    return {
        "trains": shuffled[:train_end],
        "evals": shuffled[train_end:eval_end],
        "tests": shuffled[eval_end:],
    }


def extract_patch(hsi: np.ndarray, info: Patch, patch_size: int) -> np.ndarray:
    """按坐标取出 patch，并按需填充右侧和下侧边缘。"""

    patch = hsi[info.y0:info.y1, info.x0:info.x1, :]
    if info.padded_bottom or info.padded_right:
        mode = "reflect" if patch.shape[0] > 1 and patch.shape[1] > 1 else "edge"
        patch = np.pad(
            patch,
            ((0, info.padded_bottom), (0, info.padded_right), (0, 0)),
            mode=mode,
        )
    expected = (patch_size, patch_size, OUTPUT_CHANNELS)
    if patch.shape != expected:
        raise RuntimeError(f"patch 形状错误：{patch.shape}，预期：{expected}")
    return np.ascontiguousarray(patch, dtype=np.float32)


def prepare_output_dirs(output_root: Path, overwrite: bool) -> Dict[str, Path]:
    """创建输出目录，并防止意外混入旧 MAT 文件。"""

    output_dirs = {name: output_root / name for name in SPLIT_NAMES}
    old_files = [
        path
        for directory in output_dirs.values()
        if directory.is_dir()
        for path in directory.glob("*.mat")
        if path.is_file()
    ]
    if old_files and not overwrite:
        raise FileExistsError(f"输出目录已有 {len(old_files)} 个 MAT 文件；重新生成请使用 --overwrite")
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in old_files:
            path.unlink()
    return output_dirs


def save_patch(path: Path, patch: np.ndarray) -> None:
    """将一个 patch 保存为包含 Y 的 MAT 文件。"""

    require_scipy().savemat(str(path), {MAT_KEY: patch})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="切分并随机划分 Chikusei HSI patch")
    parser.add_argument("--input", type=Path, default=None, help="原始 MAT 路径")
    parser.add_argument("--output", type=Path, default=None, help="输出根目录")
    parser.add_argument("--mat-key", default=MAT_KEY, help=f"MAT 数据键，默认 {MAT_KEY!r}")
    parser.add_argument("--band-axis", type=parse_band_axis, default=None, help="光谱轴：auto、0、1 或 2")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE, help=f"patch 边长，默认 {PATCH_SIZE}")
    parser.add_argument("--stride", type=int, default=STRIDE, help=f"滑窗步长，默认 {STRIDE}")
    parser.add_argument("--ratios", type=parse_ratios, default=SPLIT_RATIOS, help="训练、验证、测试比例")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"随机种子，默认 {RANDOM_SEED}")
    parser.add_argument("--include-edge", action="store_true", default=not DROP_INCOMPLETE_EDGE, help="填充并保留边缘残缺 patch")
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE, help="覆盖已有 MAT patch")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    return parser


def run(args: argparse.Namespace) -> Dict[str, object]:
    """执行 patch 生成、随机划分和保存。"""

    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("patch-size 和 stride 必须为正数")
    if args.stride > args.patch_size:
        raise ValueError("stride 不能大于 patch-size，否则会遗漏图像内部区域")
    if args.seed < 0:
        raise ValueError("seed 不能为负数")

    input_path = args.input or DATASET_PATH
    output_root = args.output or OUTPUT_ROOT
    hsi, raw_shape = load_hsi(input_path, args.mat_key, args.band_axis)
    height, width, channels = hsi.shape

    all_patches = build_patches(height, width, args.patch_size, args.stride, args.include_edge)
    split_data = split_patches(all_patches, args.ratios, args.seed)
    counts = {name: len(split_data[name]) for name in SPLIT_NAMES}

    output_dirs = None if args.dry_run else prepare_output_dirs(output_root, args.overwrite)
    samples = []
    for split_name in SPLIT_NAMES:
        prefix = SPLIT_PREFIXES[split_name]
        for index, info in enumerate(split_data[split_name]):
            filename = f"{prefix}_{index:05d}_y{info.y0:04d}_x{info.x0:04d}.mat"
            samples.append({"split": split_name, "filename": filename, **asdict(info)})
            if output_dirs is not None:
                patch = extract_patch(hsi, info, args.patch_size)
                save_patch(output_dirs[split_name] / filename, patch)

    manifest = {
        "source": str(input_path.resolve()),
        "mat_key": args.mat_key,
        "original_shape": raw_shape,
        "output_shape": [height, width, channels],
        "band_slice": [CHIKUSEI_BAND_START, CHIKUSEI_BAND_END],
        "patch_size": args.patch_size,
        "stride": args.stride,
        "edge_mode": "reflect" if args.include_edge else "drop",
        "split_strategy": "shuffle_patches_then_split",
        "ratios": list(args.ratios),
        "seed": args.seed,
        "patch_counts": counts,
        "samples": samples,
    }
    if output_dirs is not None:
        manifest_path = output_root / "chikusei_split_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        print(f"切分完成，manifest：{manifest_path}")
    else:
        print("dry-run 完成，未写入文件")

    print(f"输入形状：{tuple(raw_shape)}；输出 HSI：{hsi.shape}；patch 总数：{len(all_patches)}")
    for split_name in SPLIT_NAMES:
        print(f"{split_name}: {counts[split_name]}")
    return manifest


def main() -> None:
    parser = build_parser()
    try:
        run(parser.parse_args())
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
