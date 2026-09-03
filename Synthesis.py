"""
使用扩散模型生成的丰度图和解混模型端元权重合成 HSI。

主要作用：从 checkpoint 读取 decoderlayer.weight，其形状自动决定端元数和
输出波段数；对每个 ``SR`` 丰度 MAT 文件执行 1x1 线性解码，并保存 HSI MAT
和伪彩色预览图。代码直接使用训练所得端元权重，不再改变其数值范围。

运行示例：
    python3 Synthesis.py \
        --input-dir experiments/ddpm/CHIKUSEI/mat_results \
        --checkpoint experiments/unmixing/ckpts/UnmixingAE_Chikusei_latest.pth

运行环境：Python 3、PyTorch、NumPy、SciPy、OpenCV；默认自动选择 CUDA，
CUDA 不可用时使用 CPU。输入 MAT 的键必须为 ``SR``。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core import utils
from core.loaddata import HSSampledata


class DecoderAE(nn.Module):
    """以无偏置 1x1 卷积实现端元与丰度的线性混合。"""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.decoderlayer = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, abundance):
        return self.decoderlayer(abundance)


def resolve_device(requested):
    """解析 auto/cpu/cuda 设备设置。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已请求 CUDA，但当前环境中 CUDA 不可用")
    return torch.device(requested)


def load_decoder_weight(checkpoint_path, device):
    """从普通或 DataParallel checkpoint 中读取端元矩阵。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint 中没有有效的模型 state_dict")
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    key = "decoderlayer.weight"
    if key not in state_dict:
        raise KeyError(f"checkpoint 中缺少 {key!r}")
    weight = state_dict[key]
    if weight.ndim != 4 or weight.shape[2:] != (1, 1):
        raise ValueError(f"端元权重形状应为 [bands, endmembers, 1, 1]，实际为 {tuple(weight.shape)}")
    return weight.detach().to(device=device, dtype=torch.float32)


def normalize_preview(image):
    """将伪彩色图稳定地转换为 uint8。"""
    image = np.asarray(image, dtype=np.float32)
    value_range = float(image.max() - image.min())
    if value_range <= np.finfo(np.float32).eps:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - image.min()) / value_range
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def hsi_to_bgr(hsi, rgb_bands=None):
    """从 H x W x C HSI 选择 R/G/B 波段并转换为 OpenCV BGR。"""
    channels = hsi.shape[2]
    if rgb_bands is None:
        rgb_bands = (53, 33, 14) if channels >= 54 else tuple(
            np.linspace(channels - 1, 0, 3, dtype=int)
        )
    if len(rgb_bands) != 3 or min(rgb_bands) < 0 or max(rgb_bands) >= channels:
        raise ValueError(f"RGB 预览波段 {rgb_bands} 超出输出通道数 {channels}")
    red, green, blue = (hsi[:, :, index] for index in rgb_bands)
    return normalize_preview(cv2.merge([blue, green, red]))


def build_parser():
    parser = argparse.ArgumentParser(description="使用端元和生成丰度合成 HSI")
    parser.add_argument("--input-dir", type=Path, required=True, help="包含 SR MAT 文件的目录")
    parser.add_argument("--checkpoint", type=Path, required=True, help="解混模型 checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/Synthesis/HSI"))
    parser.add_argument("--preview-dir", type=Path, default=Path("experiments/Synthesis/RGB"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rgb-bands", type=int, nargs=3, default=None, metavar=("R", "G", "B"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=utils.DEFAULT_SEED)
    return parser


def run(args):
    """执行全部丰度文件的 HSI 合成。"""
    if args.num_workers < 0:
        raise ValueError("num-workers 不能为负数")
    utils.set_random_seed(args.seed)
    device = resolve_device(args.device)
    decoder_weight = load_decoder_weight(args.checkpoint, device)
    output_channels, input_channels = decoder_weight.shape[:2]

    model = DecoderAE(input_channels, output_channels).to(device)
    with torch.no_grad():
        model.decoderlayer.weight.copy_(decoder_weight)
    model.eval()

    dataset = HSSampledata(image_dir=str(args.input_dir), augment=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    endmembers = decoder_weight.cpu().numpy()

    print(f"设备：{device}；端元数：{input_channels}；输出波段数：{output_channels}")
    with torch.no_grad():
        for index, abundance in enumerate(loader):
            if abundance.ndim != 4 or abundance.shape[1] != input_channels:
                raise ValueError(
                    f"丰度应为 Bx{input_channels}xHxW，实际为 {tuple(abundance.shape)}"
                )
            abundance = ((abundance.to(device) + 1.0) / 2.0).clamp(0.0, 1.0)
            hsi = model(abundance).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            abundance_hwc = abundance.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            filename = f"{index:04d}"
            sio.savemat(
                str(args.output_dir / f"{filename}.mat"),
                {"HSI": hsi, "Abu": abundance_hwc, "End": endmembers},
            )
            preview = hsi_to_bgr(hsi, args.rgb_bands)
            preview_path = args.preview_dir / f"{filename}.jpg"
            if not cv2.imwrite(str(preview_path), preview):
                raise OSError(f"无法保存预览图：{preview_path}")


def main():
    parser = build_parser()
    try:
        run(parser.parse_args())
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
