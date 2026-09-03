"""
切分原始 Chikusei HSI 数据集，生成 Step 1 所需的训练、验证和测试样本。

功能：
1. 从 MATLAB .mat 文件读取原始 HSI 数据；
2. 将数据统一为 H x W x C，并从 Chikusei 的 128 个波段裁剪为当前项目使用的 59 个波段；
3. 先按空间区域划分 trains/evals/tests，再在区域内部切分为固定大小的 patch，避免空间数据泄漏；
4. 将每个 patch 保存为包含 ``Y`` 变量的 .mat 文件，并保存切分配置和样本索引。

使用方法：
1. 优先修改下方“用户配置区”的 DATASET_PATH 和 OUTPUT_ROOT；
2. 在项目根目录执行：
   python dataset-seg-scripts/split_chikusei.py
3. 也可以使用命令行参数临时覆盖脚本配置，例如：
   python dataset-seg-scripts/split_chikusei.py --input /path/to/Chikusei.mat --dry-run

运行环境：Python 3、NumPy、SciPy。
脚本只负责切片和保存原始数值，不执行归一化；归一化由 core/loaddata.py 在训练时完成。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.io as sio
except ImportError:  # 允许在未安装 SciPy 时查看 --help
    sio = None


# ============================ 用户配置区 ============================
# 可以直接修改下面两个路径。命令行 --input 和 --output 会临时覆盖它们。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "Chikusei.mat"
OUTPUT_ROOT = PROJECT_ROOT / "dataset"

MAT_KEY = "Y"
RAW_SPECTRAL_CHANNELS = 128
CHIKUSEI_BAND_START = 7
CHIKUSEI_BAND_END = 66  # Python 切片右边界不包含 66，共 59 个波段
PATCH_SIZE = 256
REGION_SIZE = 512
STRIDE = 256
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # trains, evals, tests
RANDOM_SEED = 3000
DROP_INCOMPLETE_EDGE = True
OVERWRITE = False
# ====================================================================


SPLIT_NAMES = ("trains", "evals", "tests")
SPLIT_PREFIXES = {"trains": "train", "evals": "eval", "tests": "test"}
EXPECTED_OUTPUT_CHANNELS = CHIKUSEI_BAND_END - CHIKUSEI_BAND_START


def require_scipy():
    """检查 MAT 文件读写依赖是否可用。"""

    if sio is None:
        raise RuntimeError(
            "处理 MAT 文件需要 SciPy，请先安装依赖：python -m pip install scipy"
        )
    return sio


@dataclass(frozen=True)
class Region:
    """原始图像中的一个不重叠空间区域。"""

    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


@dataclass(frozen=True)
class PatchInfo:
    """一个输出 patch 的来源坐标和数据集归属。"""

    split: str
    filename: str
    region_row: int
    region_col: int
    y0: int
    y1: int
    x0: int
    x1: int
    source_height: int
    source_width: int
    padded_bottom: int
    padded_right: int


def parse_band_axis(value: str) -> Optional[int]:
    """解析光谱维参数；auto 表示根据长度 128 自动识别。"""

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
    """解析 trains,evals,tests 三个比例，并验证其和为 1。"""

    try:
        ratios = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratios 格式应为 trains,evals,tests，例如 0.8,0.1,0.1") from exc

    if len(ratios) != 3 or any(not math.isfinite(item) or item <= 0 for item in ratios):
        raise argparse.ArgumentTypeError("三个数据集比例必须是正数")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise argparse.ArgumentTypeError("三个数据集比例之和必须为 1")
    return ratios  # type: ignore[return-value]


def load_mat_cube(input_path: Path, mat_key: str) -> np.ndarray:
    """从 MATLAB 文件读取数据立方体，并给出可定位的错误信息。"""

    mat_io = require_scipy()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入数据集不存在：{input_path}")

    try:
        mat_data = mat_io.loadmat(str(input_path))
    except NotImplementedError as exc:
        raise RuntimeError(
            "当前 MAT 文件可能是 MATLAB v7.3 格式，scipy.io.loadmat 不支持该格式；"
            "请先转换为 v5 MAT，或安装 h5py 后扩展读取逻辑。"
        ) from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"无法读取 MAT 文件：{input_path}") from exc

    if mat_key not in mat_data:
        available_keys = [key for key in mat_data if not key.startswith("__")]
        raise KeyError(
            f"MAT 文件中找不到变量 {mat_key!r}：{input_path}；"
            f"可用变量为 {available_keys}，请通过 --mat-key 修改。"
        )

    cube = np.asarray(mat_data[mat_key])
    if cube.ndim != 3:
        raise ValueError(
            f"变量 {mat_key!r} 必须是三维 HSI 数据，实际形状为 {cube.shape}"
        )
    if np.iscomplexobj(cube):
        raise ValueError("HSI 数据不能是复数数组")
    try:
        finite = np.isfinite(cube).all()
    except TypeError as exc:
        raise ValueError("HSI 数据必须是数值数组") from exc
    if not finite:
        raise ValueError("HSI 数据包含 NaN 或 Inf，请先清理原始数据")
    return cube


def normalize_cube_layout(
    cube: np.ndarray,
    band_axis: Optional[int],
    raw_spectral_channels: int = RAW_SPECTRAL_CHANNELS,
) -> np.ndarray:
    """将输入统一为 H x W x C，并验证光谱维长度。"""

    if band_axis is None:
        candidate_axes = [
            axis for axis, size in enumerate(cube.shape) if size == raw_spectral_channels
        ]
        if len(candidate_axes) != 1:
            raise ValueError(
                f"无法唯一确定 128 个光谱通道所在的维度，数据形状为 {cube.shape}；"
                "请通过 --band-axis 指定 0、1 或 2。"
            )
        band_axis = candidate_axes[0]

    if cube.shape[band_axis] != raw_spectral_channels:
        raise ValueError(
            f"指定的光谱维 {band_axis} 长度为 {cube.shape[band_axis]}，"
            f"预期为 {raw_spectral_channels}。"
        )

    cube_hwc = np.moveaxis(cube, band_axis, -1)
    if cube_hwc.shape[0] < 1 or cube_hwc.shape[1] < 1:
        raise ValueError(f"空间尺寸无效：{cube_hwc.shape}")
    return np.ascontiguousarray(cube_hwc, dtype=np.float32)


def select_chikusei_bands(
    cube_hwc: np.ndarray,
    band_start: int = CHIKUSEI_BAND_START,
    band_end: int = CHIKUSEI_BAND_END,
) -> np.ndarray:
    """选择当前网络使用的 59 个 Chikusei 光谱通道。"""

    if band_start < 0 or band_end <= band_start:
        raise ValueError(f"非法波段范围：[{band_start}, {band_end})")
    if band_end > cube_hwc.shape[2]:
        raise ValueError(
            f"波段范围 [{band_start}, {band_end}) 超出输入通道数 {cube_hwc.shape[2]}"
        )
    if band_end - band_start != EXPECTED_OUTPUT_CHANNELS:
        raise ValueError(
            f"当前网络要求 {EXPECTED_OUTPUT_CHANNELS} 个输出波段，"
            f"但选择了 {band_end - band_start} 个"
        )

    hsi = cube_hwc[:, :, band_start:band_end]
    return np.ascontiguousarray(hsi, dtype=np.float32)


def build_spatial_regions(height: int, width: int, region_size: int) -> List[Region]:
    """按固定网格构建不重叠空间区域。"""

    regions: List[Region] = []
    for row, y0 in enumerate(range(0, height, region_size)):
        for col, x0 in enumerate(range(0, width, region_size)):
            regions.append(
                Region(
                    row=row,
                    col=col,
                    y0=y0,
                    y1=min(y0 + region_size, height),
                    x0=x0,
                    x1=min(x0 + region_size, width),
                )
            )
    return regions


def assign_regions(
    regions: Sequence[Region],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[Region]]:
    """以固定随机种子将空间区域分配到三个数据集。"""

    if len(regions) < len(SPLIT_NAMES):
        raise ValueError("空间区域数量少于 3，无法同时生成训练、验证和测试集")

    shuffled = list(regions)
    np.random.default_rng(seed).shuffle(shuffled)

    # 先按比例取整，再保证每个 split 至少一个区域，并用最大余数补足总数。
    # 这样可以处理训练比例接近 1 的极端自定义配置，同时保持区域总数不变。
    raw_counts = np.asarray(ratios, dtype=np.float64) * len(shuffled)
    counts = np.floor(raw_counts).astype(int)
    counts = np.maximum(counts, 1)

    while int(counts.sum()) > len(shuffled):
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError("空间区域数量不足以按当前比例生成三个非空数据集")
        remove_index = max(candidates, key=lambda index: (counts[index], -index))
        counts[remove_index] -= 1

    fractional_parts = raw_counts - np.floor(raw_counts)
    while int(counts.sum()) < len(shuffled):
        add_index = max(
            range(len(SPLIT_NAMES)),
            key=lambda index: (fractional_parts[index], index),
        )
        counts[add_index] += 1
        fractional_parts[add_index] = 0.0

    train_count, eval_count, test_count = (int(count) for count in counts)

    train_end = train_count
    eval_end = train_end + eval_count
    return {
        "trains": shuffled[:train_end],
        "evals": shuffled[train_end:eval_end],
        "tests": shuffled[eval_end:],
    }


def iter_region_patches(
    hsi: np.ndarray,
    region: Region,
    patch_size: int,
    stride: int,
    pad_edges: bool,
) -> Iterable[Tuple[np.ndarray, PatchInfo]]:
    """在单个空间区域内部生成 patch，不允许跨越区域边界。"""

    if pad_edges:
        y_starts = range(region.y0, region.y1, stride)
        x_starts = range(region.x0, region.x1, stride)
    else:
        if region.height < patch_size or region.width < patch_size:
            return
        y_starts = range(region.y0, region.y1 - patch_size + 1, stride)
        x_starts = range(region.x0, region.x1 - patch_size + 1, stride)

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + patch_size, region.y1)
            x1 = min(x0 + patch_size, region.x1)
            patch = hsi[y0:y1, x0:x1, :]
            source_height, source_width = patch.shape[:2]
            padded_bottom = patch_size - source_height
            padded_right = patch_size - source_width

            if padded_bottom or padded_right:
                if not pad_edges:
                    continue
                # 原始 Chikusei 的空间尺寸远大于 1；edge 模式同时覆盖极小的测试数组。
                pad_mode = "reflect" if source_height > 1 and source_width > 1 else "edge"
                patch = np.pad(
                    patch,
                    ((0, padded_bottom), (0, padded_right), (0, 0)),
                    mode=pad_mode,
                )

            if patch.shape != (patch_size, patch_size, hsi.shape[2]):
                raise RuntimeError(f"生成了错误的 patch 形状：{patch.shape}")

            yield patch.astype(np.float32, copy=False), PatchInfo(
                split="",
                filename="",
                region_row=region.row,
                region_col=region.col,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                source_height=source_height,
                source_width=source_width,
                padded_bottom=padded_bottom,
                padded_right=padded_right,
            )


def existing_mat_files(directory: Path) -> List[Path]:
    """返回指定输出目录下的 MAT 文件。"""

    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".mat")


def prepare_output_dirs(output_root: Path, overwrite: bool) -> Dict[str, Path]:
    """创建输出目录，并防止无意覆盖已有切片。"""

    output_dirs = {name: output_root / name for name in SPLIT_NAMES}
    old_mat_files = [path for directory in output_dirs.values() for path in existing_mat_files(directory)]
    if old_mat_files and not overwrite:
        raise FileExistsError(
            f"输出目录中已存在 {len(old_mat_files)} 个 .mat 文件：{output_root}；"
            "如需重新生成，请确认路径后使用 --overwrite。"
        )

    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    if overwrite:
        # 只清理明确指定输出目录下的 .mat 文件，不触碰其他文件或目录。
        for path in old_mat_files:
            path.unlink()
    return output_dirs


def save_patch(output_path: Path, patch: np.ndarray) -> None:
    """保存一个 H x W x 59 的 HSI patch。"""

    mat_io = require_scipy()
    if patch.dtype != np.float32:
        patch = patch.astype(np.float32)
    if patch.ndim != 3 or patch.shape[2] != EXPECTED_OUTPUT_CHANNELS:
        raise ValueError(f"待保存 patch 形状错误：{patch.shape}")
    if not np.isfinite(patch).all():
        raise ValueError(f"待保存 patch 包含 NaN 或 Inf：{output_path}")
    mat_io.savemat(str(output_path), {MAT_KEY: np.ascontiguousarray(patch)})


def verify_saved_patch(path: Path, patch_size: int) -> None:
    """重新读取一个输出文件，验证保存格式与网络输入约定一致。"""

    mat_io = require_scipy()
    data = mat_io.loadmat(str(path))
    if MAT_KEY not in data:
        raise RuntimeError(f"输出文件缺少变量 {MAT_KEY!r}：{path}")
    saved = np.asarray(data[MAT_KEY])
    expected_shape = (patch_size, patch_size, EXPECTED_OUTPUT_CHANNELS)
    if saved.shape != expected_shape:
        raise RuntimeError(f"输出文件形状错误：{path}，实际 {saved.shape}，预期 {expected_shape}")
    if saved.dtype != np.float32:
        raise RuntimeError(f"输出文件数据类型错误：{path}，实际 {saved.dtype}，预期 float32")
    if not np.isfinite(saved).all():
        raise RuntimeError(f"输出文件包含 NaN 或 Inf：{path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="切分原始 Chikusei HSI 数据集")
    parser.add_argument("--input", type=Path, default=None, help="原始 Chikusei MAT 文件路径，默认使用脚本内 DATASET_PATH")
    parser.add_argument("--output", type=Path, default=None, help="输出根目录，默认使用脚本内 OUTPUT_ROOT")
    parser.add_argument("--mat-key", default=MAT_KEY, help=f"MAT 中的数据变量名，默认 {MAT_KEY!r}")
    parser.add_argument("--band-axis", type=parse_band_axis, default=None, help="光谱维：auto、0、1 或 2，默认 auto")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE, help=f"patch 边长，默认 {PATCH_SIZE}")
    parser.add_argument("--region-size", type=int, default=REGION_SIZE, help=f"空间切分区域边长，默认 {REGION_SIZE}")
    parser.add_argument("--stride", type=int, default=STRIDE, help=f"区域内部滑窗步长，默认 {STRIDE}")
    parser.add_argument("--ratios", type=parse_ratios, default=SPLIT_RATIOS, help="trains,evals,tests 比例，默认 0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"随机种子，默认 {RANDOM_SEED}")
    parser.add_argument(
        "--include-edge",
        action="store_true",
        default=not DROP_INCOMPLETE_EDGE,
        help="使用反射填充保留不完整边缘，默认由 DROP_INCOMPLETE_EDGE 决定",
    )
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE, help="清理输出 split 目录中的旧 MAT 文件后重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据并统计预计样本数，不写输出文件")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """验证切片参数，尽早阻止无效配置。"""

    if args.patch_size <= 0:
        raise ValueError("patch-size 必须为正数")
    if args.region_size < args.patch_size:
        raise ValueError("region-size 不能小于 patch-size")
    if args.stride <= 0 or args.stride > args.patch_size:
        raise ValueError("stride 必须为正数且不能大于 patch-size")
    if args.seed < 0:
        raise ValueError("seed 不能为负数")


def run(args: argparse.Namespace) -> Dict[str, object]:
    """执行读取、区域划分、patch 生成和保存。"""

    validate_arguments(args)
    input_path = args.input if args.input is not None else DATASET_PATH
    output_root = args.output if args.output is not None else OUTPUT_ROOT

    raw_cube = load_mat_cube(input_path, args.mat_key)
    raw_shape = list(raw_cube.shape)
    cube_hwc = normalize_cube_layout(raw_cube, args.band_axis)
    normalized_shape = list(cube_hwc.shape)
    del raw_cube
    hsi = select_chikusei_bands(cube_hwc)
    del cube_hwc
    height, width, channels = hsi.shape
    regions = build_spatial_regions(height, width, args.region_size)
    split_regions = assign_regions(regions, args.ratios, args.seed)

    split_region_ids = {
        name: {(region.row, region.col) for region in split_regions[name]}
        for name in SPLIT_NAMES
    }
    if sum(len(item) for item in split_region_ids.values()) != len(regions):
        raise RuntimeError("空间区域分配数量不一致")
    if len(set.union(*split_region_ids.values())) != len(regions):
        raise RuntimeError("空间区域重复归属多个 split")

    print(f"输入：{input_path}")
    print(f"原始数据形状：{tuple(raw_shape)}，统一为 H×W×C：{tuple(normalized_shape)}")
    print(f"输出 HSI 形状：{hsi.shape}，波段范围：[{CHIKUSEI_BAND_START}, {CHIKUSEI_BAND_END})")
    print(f"空间区域：{len(regions)} 个，区域尺寸上限：{args.region_size}×{args.region_size}")
    print(f"切片尺寸：{args.patch_size}×{args.patch_size}，步长：{args.stride}")

    output_dirs: Optional[Dict[str, Path]] = None
    if not args.dry_run:
        output_dirs = prepare_output_dirs(output_root, args.overwrite)

    patch_infos: Dict[str, List[PatchInfo]] = {name: [] for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        prefix = SPLIT_PREFIXES[split_name]
        for region in split_regions[split_name]:
            for patch, info in iter_region_patches(
                hsi=hsi,
                region=region,
                patch_size=args.patch_size,
                stride=args.stride,
                pad_edges=args.include_edge,
            ):
                filename = (
                    f"{prefix}_r{info.region_row:03d}_c{info.region_col:03d}_"
                    f"y{info.y0:04d}_x{info.x0:04d}.mat"
                )
                info = PatchInfo(
                    split=split_name,
                    filename=filename,
                    region_row=info.region_row,
                    region_col=info.region_col,
                    y0=info.y0,
                    y1=info.y1,
                    x0=info.x0,
                    x1=info.x1,
                    source_height=info.source_height,
                    source_width=info.source_width,
                    padded_bottom=info.padded_bottom,
                    padded_right=info.padded_right,
                )
                patch_infos[split_name].append(info)
                if output_dirs is not None:
                    save_patch(output_dirs[split_name] / filename, patch)

    counts = {name: len(patch_infos[name]) for name in SPLIT_NAMES}
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"至少一个 split 没有生成 patch：{counts}")

    manifest = {
        "source": str(input_path.resolve()),
        "mat_key": args.mat_key,
        "original_shape": raw_shape,
        "normalized_shape": normalized_shape,
        "output_shape": [height, width, channels],
        "band_slice": [CHIKUSEI_BAND_START, CHIKUSEI_BAND_END],
        "patch_size": args.patch_size,
        "region_size": args.region_size,
        "stride": args.stride,
        "edge_mode": "reflect" if args.include_edge else "drop",
        "ratios": list(args.ratios),
        "seed": args.seed,
        "region_counts": {name: len(split_regions[name]) for name in SPLIT_NAMES},
        "patch_counts": counts,
        "samples": [asdict(info) for name in SPLIT_NAMES for info in patch_infos[name]],
    }

    if output_dirs is not None:
        manifest_path = output_root / "chikusei_split_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

        for split_name in SPLIT_NAMES:
            first_file = output_dirs[split_name] / patch_infos[split_name][0].filename
            verify_saved_patch(first_file, args.patch_size)
        print(f"切分完成，manifest：{manifest_path}")
    else:
        print("dry-run 完成，未写入任何文件。")

    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: regions={len(split_regions[split_name])}, "
            f"patches={counts[split_name]}"
        )
    return manifest


def main() -> None:
    """命令行入口。"""

    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
