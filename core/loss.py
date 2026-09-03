import torch


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class reconstruction_SADloss(torch.nn.Module):
    def __init__(self):
        super(reconstruction_SADloss, self).__init__()

    def forward(self, x, y):
        cosine = torch.cosine_similarity(x, y, dim=1)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        abundance_loss = torch.acos(cosine)
        abundance_loss = torch.mean(abundance_loss)
        return abundance_loss

class TVLossEndmembers(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossEndmembers, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        if x.ndim != 4 or x.shape[0] < 2:
            raise ValueError("Endmember weights must have shape [bands, endmembers, 1, 1]")
        spectral_diff = x[1:, :, :, :] - x[:-1, :, :, :]
        return self.TVLoss_weight * 2 * spectral_diff.pow(2).mean()

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c)

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
