import scipy.io as sio
import numpy as np
import torch
import cv2
import os
import random


DEFAULT_SEED = 3000


def set_random_seed(seed=DEFAULT_SEED):
    """固定 Python、NumPy 和 PyTorch 随机状态，尽量保证实验可复现。"""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

def data_augmentation(label, mode=0):
    if mode == 0:
        # original
        return label
    elif mode == 1:
        # flip up and down
        return np.flipud(label)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(label)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(label))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(label, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(label, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(label, k=3))


# rescale every channel to between 0 and 1
def channel_scale(img):
    eps = 1e-5
    max_list = np.max((np.max(img, axis=0)), axis=0)
    min_list = np.min((np.min(img, axis=0)), axis=0)
    output = (img - min_list) / (max_list - min_list + eps)
    return output


# up sample before feeding into network
def upsample(img, ratio):
    [h, w, _] = img.shape
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    new_h, new_w = int(round(ratio * h)), int(round(ratio * w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def bicubic_downsample(img, ratio):
    [h, w, _] = img.shape
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    new_h, new_w = int(round(ratio * h)), int(round(ratio * w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def wald_downsample(data, ratio):
    [h, w, c] = data.shape
    out = []
    for i in range(c):
        dst = cv2.GaussianBlur(data[:, :, i], (7, 7), 0)
        dst = dst[0:h:ratio, 0:w:ratio, np.newaxis]
        out.append(dst)
    out = np.concatenate(out, axis=2)
    return out


def save_result(result_dir, out):
    out = out.numpy().transpose((0, 2, 3, 1))
    sio.savemat(result_dir, {'output': out})


def sam_loss(y, ref):
    if y.shape != ref.shape or y.ndim != 4:
        raise ValueError("y and ref must have the same BxCxHxW shape")
    cosine = torch.nn.functional.cosine_similarity(y, ref, dim=1, eps=1e-8)
    cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cosine).mean()


def extract_RGB(y):
    # take 4-2-1 band (R-G-B) for WV-3
    R = torch.unsqueeze(torch.mean(y[:, 4:8, :, :], 1), 1)
    G = torch.unsqueeze(torch.mean(y[:, 2:4, :, :], 1), 1)
    B = torch.unsqueeze(torch.mean(y[:, 0:2, :, :], 1), 1)
    y_RGB = torch.cat((R, G, B), 1)
    return y_RGB


def extract_edge(data):
    N = data.shape[0]
    out = np.zeros_like(data)
    for i in range(N):
        if len(data.shape) == 3:
            out[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            out[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return out


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


def add_channel(rgb):
    # initialize other channels using the average of RGB from VGG
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape Bx3xHxW")
    R = torch.unsqueeze(rgb[:, 0, :, :], 1)
    G = torch.unsqueeze(rgb[:, 1, :, :], 1)
    B = torch.unsqueeze(rgb[:, 2, :, :], 1)
    all_channel = torch.cat((B, B, G, G, R, R, R, R), 1)
    return all_channel


# from LapSRN
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

from datetime import datetime
def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')
