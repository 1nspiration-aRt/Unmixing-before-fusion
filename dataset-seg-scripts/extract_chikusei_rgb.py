"""查看 Chikusei MAT 元数据，并从 59 波段中的指定索引生成伪 RGB 预览。

环境：Python 3.10+，numpy、scipy、h5py、Pillow。
用法：
    python dataset-seg-scripts/extract_chikusei_rgb.py --input /path/Chikusei.mat --inspect
    python dataset-seg-scripts/extract_chikusei_rgb.py --input /path/Chikusei.mat --key Y --output preview.png
    python dataset-seg-scripts/extract_chikusei_rgb.py --input /path/patch.mat --bands 7 17 27 --output preview.png

输入为 H×W×128 或 H×W×59，其他布局通过 --band-axis 指定光谱轴。
--bands 始终是裁剪后 59 波段内的 Python 零基索引，顺序为 R、G、B。
原始 128 波段对应先取 [7:66]；默认 RGB 对应原始索引 14、24、34。
兼容 MATLAB v7.3；该格式仅读取选中的三个波段，避免展开完整数据立方体。
PNG 使用三个通道共用的 min/max 显示拉伸，不代表物理真彩色，也不用于训练。
--inspect 仅打印变量形状、类型及小型数值变量，便于查找波长表；不生成文件。
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from PIL import Image


def inspect_mat(path):
    """不展开大数组；显示小型数值变量以帮助识别真实波长元数据。"""
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as handle:
            def show(name, obj):
                if name.startswith('#') or not isinstance(obj, h5py.Dataset):
                    return
                # MATLAB v7.3 的存储轴序与 MATLAB 逻辑轴序相反。
                print(f'{name}: MATLAB shape={obj.shape[::-1]}, dtype={obj.dtype}')
                if obj.size <= 256 and obj.dtype.kind in 'iuf':
                    print(np.asarray(obj).transpose().reshape(-1).tolist())
                for key, value in obj.attrs.items():
                    print(f'  attribute {key}: {value}')
            handle.visititems(show)
    else:
        for name, shape, kind in sio.whosmat(path):
            print(f'{name}: shape={shape}, type={kind}')
            if np.prod(shape) <= 256 and kind in {
                'double', 'single', 'int8', 'uint8', 'int16', 'uint16',
                'int32', 'uint32', 'int64', 'uint64',
            }:
                value = sio.loadmat(path, variable_names=[name])[name]
                print(np.asarray(value).reshape(-1).tolist())


def resolve_indices(shape, band_axis, bands):
    """将 59 波段相对索引转换为实际输入数组的通道索引。"""
    if len(shape) != 3:
        raise ValueError(f'HSI 必须是三维数组，实际 shape={shape}')
    if band_axis is None:
        axes = [axis for axis, size in enumerate(shape) if size in (59, 128)]
        if len(axes) != 1:
            raise ValueError(f'无法唯一识别光谱轴：{shape}，请指定 --band-axis')
        band_axis = axes[0]
    channels = shape[band_axis]
    if channels not in (59, 128):
        raise ValueError(f'光谱轴应为 59 或 128 通道，实际为 {channels}')
    offset = 7 if channels == 128 else 0
    indices = [band + offset for band in bands]
    print(f'输入逻辑形状={shape}，光谱轴={band_axis}')
    print(f'R/G/B 在59波段内的零基索引={bands}；输入文件索引={indices}')
    return band_axis, indices


def read_rgb(path, key, band_axis, bands):
    """读取三个二维波段并按 RGB 顺序拼接；不进行颜色或物理标定。"""
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as handle:
            if key not in handle:
                raise KeyError(f'找不到变量 {key!r}，请先运行 --inspect')
            cube = handle[key]
            if not isinstance(cube, h5py.Dataset):
                raise ValueError(f'{key!r} 必须是数值数组')
            if cube.dtype.kind not in 'iuf':
                raise ValueError(f'不支持的数据类型：{cube.dtype}')
            axis, indices = resolve_indices(cube.shape[::-1], band_axis, bands)
            storage_axis = 2 - axis
            planes = []
            for index in indices:
                selection = [slice(None)] * 3
                selection[storage_axis] = index
                planes.append(np.asarray(cube[tuple(selection)], dtype=np.float32).T)
            return np.stack(planes, axis=-1)
    data = sio.loadmat(path, variable_names=[key])
    if key not in data:
        raise KeyError(f'找不到变量 {key!r}，请先运行 --inspect')
    cube = np.asarray(data[key])
    if cube.dtype.kind not in 'iuf':
        raise ValueError(f'不支持的数据类型：{cube.dtype}')
    axis, indices = resolve_indices(cube.shape, band_axis, bands)
    selected = np.asarray(np.take(cube, indices, axis=axis), dtype=np.float32)
    return np.moveaxis(selected, axis, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='原始128波段或裁剪后59波段 MAT')
    parser.add_argument('--key', default='Y', help='HSI变量名；用 --inspect 查看')
    parser.add_argument('--bands', type=int, nargs=3, default=[7, 17, 27], metavar=('R', 'G', 'B'))
    parser.add_argument('--band-axis', type=int, choices=(0, 1, 2), help='MAT逻辑布局的光谱轴，默认自动识别')
    parser.add_argument('--output', type=Path, help='输出 PNG 路径')
    parser.add_argument('--inspect', action='store_true', help='只查看元数据，不生成图片')
    args = parser.parse_args()
    try:
        if not args.input.is_file():
            raise FileNotFoundError(f'文件不存在：{args.input}')
        if args.inspect:
            inspect_mat(args.input)
            return
        if args.output is None or args.output.suffix.lower() != '.png':
            raise ValueError('请通过 --output 指定 PNG 文件路径')
        if args.output.exists():
            raise FileExistsError(f'输出已存在，请使用其他文件名：{args.output}')
        if any(band < 0 or band >= 59 for band in args.bands):
            raise ValueError('--bands 必须是 0 到 58 之间的三个整数')
        rgb = read_rgb(args.input, args.key, args.band_axis, args.bands)
        if not np.isfinite(rgb).all():
            raise ValueError('选中波段含 NaN/Inf，停止生成，避免掩盖数据问题')
        for index, name in enumerate(('R', 'G', 'B')):
            plane = rgb[:, :, index]
            print(f'{name}: min={plane.min():.8g}, max={plane.max():.8g}, mean={plane.mean():.8g}')
        # 仅用于PNG显示；所有通道共享尺度，不独立拉伸各通道。
        low, high = float(rgb.min()), float(rgb.max())
        if high <= low:
            raise ValueError('三个波段整体为常量，无法进行显示拉伸')
        preview = np.rint(np.clip((rgb - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(args.output)
        print(f'已保存：{args.output.resolve()}，尺寸={preview.shape}')
        print(f'仅显示拉伸：共同 min={low:.8g}, max={high:.8g}；不是物理真彩色。')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
