import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
from core import utils


# Chikusei's 59-band setting keeps bands 8--66 in MATLAB indexing, i.e.
# 7:66 in Python indexing. These three bands form the encoder input.
CHIKUSEI_BAND_SLICE = slice(7, 66)
DEFAULT_RGB_BANDS = (7, 17, 27)

def is_mat_file(filename):
    return filename.lower().endswith(".mat")

def datanorm(input):
    input = np.asarray(input, dtype=np.float32)
    input_min = np.nanmin(input)
    input_max = np.nanmax(input)
    denominator = input_max - input_min
    if not np.isfinite(input_min) or not np.isfinite(input_max):
        raise ValueError("Input contains no finite values")
    if denominator <= 1e-12:
        return np.zeros_like(input, dtype=np.float32)
    return (input - input_min) / denominator


def _list_mat_files(image_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {image_dir}")
    image_files = [
        os.path.join(image_dir, filename)
        for filename in sorted(os.listdir(image_dir))
        if is_mat_file(filename)
    ]
    if not image_files:
        raise RuntimeError(f"No .mat files found in dataset directory: {image_dir}")
    return image_files


def _to_tensor(image):
    return torch.from_numpy(image.copy()).permute(2, 0, 1)



class HSIDataset(data.Dataset):
    """Load HSI targets and derive the three-channel encoder input.

    Each file must contain ``Y`` in H x W x C layout. Chikusei files may
    contain either the original 128 bands or the preprocessed 59 bands.
    The returned tuple is ``(hsi_target, pseudo_rgb)`` in C x H x W layout.
    """

    def __init__(
        self,
        image_dir,
        augment=None,
        use_3D=False,
        output_channels=59,
        rgb_bands=DEFAULT_RGB_BANDS
    ):
        self.image_files = _list_mat_files(image_dir)
        self.augment = augment
        self.use_3Dconv = use_3D
        self.output_channels = output_channels
        self.rgb_bands = tuple(rgb_bands)
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        if "Y" not in data:
            raise KeyError(f"Missing key 'Y' in {load_dir}")

        hsi = np.asarray(data["Y"], dtype=np.float32)
        if hsi.ndim != 3:
            raise ValueError(f"HSI data must be HxWxC, got {hsi.shape} in {load_dir}")

        if hsi.shape[2] == 128 and self.output_channels == 59:
            hsi = hsi[:, :, CHIKUSEI_BAND_SLICE]
        if hsi.shape[2] != self.output_channels:
            raise ValueError(
                f"Expected {self.output_channels} HSI bands, got {hsi.shape[2]} in {load_dir}"
            )
        if max(self.rgb_bands) >= hsi.shape[2]:
            raise ValueError(
                f"RGB band indices {self.rgb_bands} exceed HSI channels {hsi.shape[2]}"
            )

        hsi = datanorm(hsi)
        pseudo_rgb = datanorm(hsi[:, :, self.rgb_bands])
        hsi = utils.data_augmentation(hsi, mode=aug_num)
        pseudo_rgb = utils.data_augmentation(pseudo_rgb, mode=aug_num)

        return _to_tensor(hsi), _to_tensor(pseudo_rgb)

    def __len__(self):
        return len(self.image_files)*self.factor


class RGBDataset(data.Dataset):
    """Load external RGB ``.mat`` files for abundance inference."""

    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_files = _list_mat_files(image_dir)
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)

        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        if "Y" not in data:
            raise KeyError(f"Missing key 'Y' in {load_dir}")

        rgb = np.asarray(data["Y"], dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB data must be HxWx3, got {rgb.shape} in {load_dir}")

        rgb = datanorm(rgb)
        rgb = utils.data_augmentation(rgb, mode=aug_num)
        return _to_tensor(rgb)

    def __len__(self):
        return len(self.image_files)*self.factor


def data_transform(input,min_max=(-1, 1)):
    input = input * (min_max[1] - min_max[0]) + min_max[0]
    return input

class AbuDataset(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_files = _list_mat_files(image_dir)
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1
            
        self.length = len(self.image_files)*self.factor
        
    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        gt = np.array(data['Abu'][...], dtype=np.float32)
        gt = data_transform(gt)

        if self.use_3Dconv:
            gt = gt[np.newaxis, :, :, :]
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        return {'Abu': gt}

    def __len__(self):
        return len(self.image_files)*self.factor
    
class HSSampledata(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_files = _list_mat_files(image_dir)
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1
    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        gt = np.array(data['SR'][...], dtype=np.float32)

        if self.use_3Dconv:
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)

        else:
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)


        return gt

    def __len__(self):
        return len(self.image_files)*self.factor
