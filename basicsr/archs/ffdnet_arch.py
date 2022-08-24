import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY, pixel_unshuffle


@ARCH_REGISTRY.register()
class FFDNet(nn.Module):
    """FFDNet network structure.

    Paper: FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch
                  and https://github.com/MKFMIKU/d2sm

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16):
        super(FFDNet, self).__init__()

        self.m_head = nn.Sequential(
            nn.Conv2d(num_in_ch * 2 * 2 + 1, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.m_body = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(num_feat, self.args.n_feats, 3, 1, 1),
                nn.ReLU(inplace=True),
            ) for _ in range(num_block)])
        self.m_tail = nn.Conv2d(num_feat, num_out_ch * 2 * 2, 3, 1, 1)

    def forward(self, x, sigma):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = pixel_unshuffle(x, scale=2)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)

        x = self.m_tail(self.m_body(self.m_head(x)))
        x = F.pixel_shuffle(x, upscale_factor=self.sf)

        x = x[..., :h, :w]
        return x
