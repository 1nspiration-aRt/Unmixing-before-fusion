import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        use_cuda = bool(opt['gpu_ids']) and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
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
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            x = [item.to(self.device) if item is not None else None for item in x]
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
