import numpy as np
import scipy.io as sio

data = sio.loadmat("../dataset/inferred_abu/0006.mat")
abu = np.asarray(data["Abu"], dtype=np.float32)

print("shape:", abu.shape)
print("min:", abu.min(axis=(0, 1)))
print("max:", abu.max(axis=(0, 1)))
print("mean:", abu.mean(axis=(0, 1)))
print("std:", abu.std(axis=(0, 1)))

sum_map = abu.sum(axis=2)
print("sum-to-one mean error:", np.abs(sum_map - 1).mean())
print("sum-to-one max error:", np.abs(sum_map - 1).max())

argmax_ratio = np.bincount(
    abu.argmax(axis=2).reshape(-1),
    minlength=abu.shape[2],
) / abu.shape[0] / abu.shape[1]
print("argmax ratio:", argmax_ratio)