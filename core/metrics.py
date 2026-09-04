# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   metrics.py
@Time    :   2019/12/4 17:35
@Desc    :
"""
import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def compare_ergas(x_true, x_pred, ratio):
    """
    Calculate ERGAS, ERGAS offers a global indication of the quality of fused image.The ideal value is 0.
    :param x_true:
    :param x_pred:
    :param ratio: 上采样系数
    :return:
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    sum_ergas = 0
    eps = np.finfo(np.float32).eps
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / max(np.mean(vec_x) ** 2, eps)
        sum_ergas += tmp
    return (100 / ratio) * np.sqrt(sum_ergas / x_true.shape[0])


def compare_sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    if x_true.shape != x_pred.shape or x_true.ndim != 3:
        raise ValueError("x_true and x_pred must have the same HxWxC shape")
    dot = np.sum(x_true * x_pred, axis=2)
    true_norm = np.linalg.norm(x_true, axis=2)
    pred_norm = np.linalg.norm(x_pred, axis=2)
    denominator = true_norm * pred_norm
    cosine = np.zeros_like(dot, dtype=np.float32)
    valid = denominator > np.finfo(np.float32).eps
    cosine[valid] = dot[valid] / denominator[valid]
    cosine[(true_norm <= np.finfo(np.float32).eps) &
           (pred_norm <= np.finfo(np.float32).eps)] = 1.0
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.mean(angles) * 180.0 / np.pi)


def compare_corr(x_true, x_pred):
    """
    Calculate the cross correlation between x_pred and x_true.
    求对应波段的相关系数，然后取均值
    CC is a spatial measure.
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    x_true = x_true - np.mean(x_true, axis=1).reshape(-1, 1)
    x_pred = x_pred - np.mean(x_pred, axis=1).reshape(-1, 1)
    numerator = np.sum(x_true * x_pred, axis=1).reshape(-1, 1)
    denominator = np.sqrt(
        np.sum(x_true * x_true, axis=1) * np.sum(x_pred * x_pred, axis=1)
    ).reshape(-1, 1)
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float32).eps,
    )
    return float(correlation.mean())


def img_2d_mat(x_true, x_pred):
    """
    # 将三维的多光谱图像转为2位矩阵
    :param x_true: (H, W, C)
    :param x_pred: (H, W, C)
    :return: a matrix which shape is (C, H * W)
    """
    if x_true.shape != x_pred.shape or x_true.ndim != 3:
        raise ValueError("x_true and x_pred must have the same HxWxC shape")
    h, w, c = x_true.shape
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    x_mat = np.zeros((c, h * w), dtype=np.float32)
    y_mat = np.zeros((c, h * w), dtype=np.float32)
    for i in range(c):
        x_mat[i] = x_true[:, :, i].reshape((1, -1))
        y_mat[i] = x_pred[:, :, i].reshape((1, -1))
    return x_mat, y_mat


def compare_rmse(x_true, x_pred):
    """
    Calculate Root mean squared error
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.shape[0] * x_true.shape[1] * x_true.shape[2]))


def compare_mpsnr(x_true, x_pred, data_range):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [compare_psnr(x_true[:, :, k], x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]

    return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, data_range, multidimension=False):
    """
    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [compare_ssim(x_true[:, :, i], x_pred[:, :, i], data_range=data_range)
            for i in range(x_true.shape[2])]

    return np.mean(mssim)


def compare_sid(x_true, x_pred):
    """
    SID is an information theoretic measure for spectral similarity and discriminability.
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    N = x_true.shape[2]
    err = np.zeros(N)
    for i in range(N):
        err[i] = abs(np.sum(x_pred[:, :, i] * np.log10((x_pred[:, :, i] + 1e-3) / (x_true[:, :, i] + 1e-3))) +
                     np.sum(x_true[:, :, i] * np.log10((x_true[:, :, i] + 1e-3) / (x_pred[:, :, i] + 1e-3))))
    return np.mean(err / (x_true.shape[1] * x_true.shape[0]))


def compare_appsa(x_true, x_pred):
    """

    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    nom = np.sum(x_true * x_pred, axis=2)
    denom = np.linalg.norm(x_true, axis=2) * np.linalg.norm(x_pred, axis=2)

    cos = np.where((nom / (denom + 1e-3)) > 1, 1, (nom / (denom + 1e-3)))
    appsa = np.arccos(cos)
    return np.sum(appsa) / (x_true.shape[1] * x_true.shape[0])


def compare_mare(x_true, x_pred):
    """

    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    diff = x_true - x_pred
    abs_diff = np.abs(diff)
    relative_abs_diff = np.divide(abs_diff, x_true + 1)  # added epsilon to avoid division by zero.
    return np.mean(relative_abs_diff)


def img_qi(img1, img2, block_size=8):
    N = block_size ** 2
    sum2_filter = np.ones((block_size, block_size))

    img1_sq = img1 * img1
    img2_sq = img2 * img2
    img12 = img1 * img2

    img1_sum = convolve2d(img1, np.rot90(sum2_filter), mode='valid')
    img2_sum = convolve2d(img2, np.rot90(sum2_filter), mode='valid')
    img1_sq_sum = convolve2d(img1_sq, np.rot90(sum2_filter), mode='valid')
    img2_sq_sum = convolve2d(img2_sq, np.rot90(sum2_filter), mode='valid')
    img12_sum = convolve2d(img12, np.rot90(sum2_filter), mode='valid')

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul
    quality_map = np.ones(denominator.shape)
    index = (denominator1 == 0) & (img12_sq_sum_mul != 0)
    quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]
    index = (denominator != 0)
    quality_map[index] = numerator[index] / denominator[index]
    return quality_map.mean()


def compare_qave(x_true, x_pred, block_size=8):
    n_bands = x_true.shape[2]
    q_orig = np.zeros(n_bands)
    for idim in range(n_bands):
        q_orig[idim] = img_qi(x_true[:, :, idim], x_pred[:, :, idim], block_size)
    return q_orig.mean()


def quality_assessment(x_true, x_pred, data_range, ratio, multi_dimension=False, block_size=8):
    """

    :param multi_dimension:
    :param ratio:
    :param data_range:
    :param x_true:
    :param x_pred:
    :param block_size
    :return:
    """
    result = {'MPSNR': compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range),
              'MSSIM': compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range,
                                     multidimension=multi_dimension),
              'ERGAS': compare_ergas(x_true=x_true, x_pred=x_pred, ratio=ratio),
              'SAM': compare_sam(x_true=x_true, x_pred=x_pred),
              # 'SID': compare_sid(x_true=x_true, x_pred=x_pred),
              'CrossCorrelation': compare_corr(x_true=x_true, x_pred=x_pred),
              'RMSE': compare_rmse(x_true=x_true, x_pred=x_pred),
              # 'APPSA': compare_appsa(x_true=x_true, x_pred=x_pred),
              # 'MARE': compare_mare(x_true=x_true, x_pred=x_pred),
              # "QAVE": compare_qave(x_true=x_true, x_pred=x_pred, block_size=block_size)
              }
    # NumPy reductions can return np.float32/np.float64, which json.dump
    # cannot serialize reliably.  Keep the public metric contract JSON-safe.
    return {name: float(value) for name, value in result.items()}

# from scipy import io as sio
# im_out = np.array(sio.loadmat('/home/zhwzhong/PycharmProject/HyperSR/SOAT/HyperSR/SRindices/Chikuse_EDSRViDeCNN_Blocks=9_Feats=256_Loss_H_Real_1_1_X2X2_N5new_BS32_Epo60_epoch_60_Fri_Sep_20_21:38:44_2019.mat')['output'])
# im_gt = np.array(sio.loadmat('/home/zhwzhong/PycharmProject/HyperSR/SOAT/HyperSR/SRindices/Chikusei_test.mat')['gt'])
#
# sum_rmse, sum_sam, sum_psnr, sum_ssim, sum_ergas = [], [], [], [], []
# for i in range(im_gt.shape[0]):
#     print(im_out[i].shape)
#     score = quality_assessment(x_pred=im_out[i], x_true=im_gt[i], data_range=1, ratio=4, multi_dimension=False, block_size=8)
#     sum_rmse.append(score['RMSE'])
#     sum_psnr.append(score['MPSNR'])
#     sum_ssim.append(score['MSSIM'])
#     sum_sam.append(score['SAM'])
#     sum_ergas.append(score['ERGAS'])
#
# print(np.mean(sum_rmse), np.mean(sum_psnr), np.mean(sum_ssim), np.mean(sum_sam))
