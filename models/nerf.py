import torch
from torch import nn
from math import pi
import numpy as np

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels 
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        
        # self.B = np.random.randn(N_freqs,2)
        
        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        
        # for freq_id in range(self.N_freqs):
        #     for func_id in range(len(self.funcs)):
        #         func = self.funcs[func_id]
        #         scale_cur = 2.3*self.B[freq_id, func_id]
        #         out += [func(2*pi*scale_cur*x)]
        
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
 
        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, D=8, W=256,
                 in_channels_xy=82,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xy: number of input channels for xy (2+2*20*2=82 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xy = in_channels_xy
        self.skips = skips

        # xy encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xy, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xy, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"uvxy_encoding_{i+1}", layer)
            
        self.uvxy_encoding_final1 = nn.Linear(W, W)

        self.uvxy_encoding_final2 = nn.Sequential(
                                nn.Linear(W, W//2),
                                nn.ReLU(True))
        # output layers
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 1),
                        nn.Sigmoid())

    def forward(self, x):
        """
        Inputs:
            x: (B, self.in_channels_xy) the embedded vector of position
            
        Outputs:
            out: (B, 1), gray value
        """
        input_uvxy = x

        uvxy_ = input_uvxy
        for i in range(self.D):
            if i in self.skips:
                uvxy_ = torch.cat([input_uvxy, uvxy_], -1)
            uvxy_ = getattr(self, f"uvxy_encoding_{i+1}")(uvxy_)

        uvxy_encoding_final1 = self.uvxy_encoding_final1(uvxy_)
        uvxy_encoding_final2 = self.uvxy_encoding_final2(uvxy_encoding_final1)
        out = self.rgb(uvxy_encoding_final2)

        return out
        