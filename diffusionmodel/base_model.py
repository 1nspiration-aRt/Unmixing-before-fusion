import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        requested_cuda = bool(opt['gpu_ids'])
        use_cuda = requested_cuda and torch.cuda.is_available()
        if requested_cuda and not use_cuda:
            raise RuntimeError(
                'CUDA GPU was requested but PyTorch cannot access it. '
                'Check CUDA_VISIBLE_DEVICES, the GPU index, and the PyTorch CUDA build.'
            )
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.channels_last = bool(opt['train']['channels_last']) and use_cuda
        self.non_blocking = bool(opt['train']['pin_memory']) and use_cuda
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        """将张量传至运行设备，并为卷积输入应用可选的 channels-last 布局。"""
        def move_tensor(item):
            item = item.to(self.device, non_blocking=self.non_blocking)
            if (
                self.channels_last
                and isinstance(item, torch.Tensor)
                and item.ndim == 4
                and item.is_floating_point()
            ):
                item = item.contiguous(memory_format=torch.channels_last)
            return item

        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = move_tensor(item)
        elif isinstance(x, list):
            x = [move_tensor(item) if item is not None else None for item in x]
        else:
            x = move_tensor(x)
        return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
