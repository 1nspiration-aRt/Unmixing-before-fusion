"""扩散模型结果的图像和 MAT 保存接口。"""

import cv2
import scipy.io as sio

from core.HSImetrics import tensor2img, tensor2rgb, tensor2rgb_band8


def save_img(img, img_path, mode="RGB"):
    """保存 OpenCV BGR 图像，并在写入失败时给出错误。"""
    if not cv2.imwrite(img_path, img):
        raise OSError(f"Failed to save image: {img_path}")


def save_mat(mat, mat_path):
    """以 SR 键保存生成结果。"""
    sio.savemat(mat_path, {"SR": mat})


def save_mat_gt(gt, mat, mat_path):
    """以 GT 和 SR 键保存参考与生成结果。"""
    sio.savemat(mat_path, {"GT": gt, "SR": mat})
