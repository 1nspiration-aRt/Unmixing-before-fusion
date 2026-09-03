import numpy as np
import cv2
import scipy.io as sio

from core.metrics import compare_mpsnr, compare_mssim, compare_sam



def tensor2img(tensor):
    min_max=(-1, 1)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = tensor.numpy().transpose(1, 2, 0)
    return tensor


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    value_range = float(x.max() - x.min())
    if value_range <= np.finfo(np.float32).eps:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - x.min()) / value_range
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def tensor2rgb(tensor, rgb_bands=None):
    min_max=(-1, 1)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = tensor.numpy().transpose(1, 2, 0)
    channels = tensor.shape[2]
    if rgb_bands is None:
        rgb_bands = tuple(np.linspace(channels - 1, 0, 3, dtype=int))
    if len(rgb_bands) != 3 or min(rgb_bands) < 0 or max(rgb_bands) >= channels:
        raise ValueError(f"rgb_bands {rgb_bands} exceed {channels} channels")
    red, green, blue = (tensor[:, :, index] for index in rgb_bands)
    return norm(cv2.merge([blue, green, red]))

def tensor2rgb_band8(tensor):
    min_max=(-1, 1)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = tensor.numpy().transpose(1, 2, 0)
    if tensor.shape[2] < 3:
        raise ValueError("At least three channels are required for RGB preview")
    red, green, blue = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
    return norm(cv2.merge([blue, green, red]))



def calculate_psnr(x_true, x_pred, data_range=1.0):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    return compare_mpsnr(x_true, x_pred, data_range=data_range)


def calculate_sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    return compare_sam(x_true, x_pred)



def calculate_ssim(x_true, x_pred, data_range=1.0):
    """

    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    return compare_mssim(x_true, x_pred, data_range=data_range)

def save_img(img, img_path, mode='RGB'):
    cv2_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if mode == 'RGB' else img
    if not cv2.imwrite(img_path, cv2_image):
        raise OSError(f"Failed to write image: {img_path}")
    
def save_mat(mat, mat_path):
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    sio.savemat(mat_path,{'SR':mat})


def save_mat_gt(gt, mat, mat_path):
    sio.savemat(mat_path,{'GT':gt, 'SR':mat})
