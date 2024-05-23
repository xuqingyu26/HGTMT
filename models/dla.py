## TransCenter: Transformers with Dense Representations for Multiple-Object Tracking
## Copyright Inria
## Year 2022
## Contact : yihong.xu@inria.fr
##
## TransCenter is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## TransCenter is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program, TransCenter.  If not, see <http://www.gnu.org/licenses/> and the LICENSE file.
##
##
## TransCenter has code derived from
## (1) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
## (2) 2020 Philipp Bergmann, Tim Meinhardt. (GNU General Public License v3.0 Licence: https://github.com/phil-bergmann/tracking_wo_bnw)
## (3) 2020 Facebook. (Apache License Version 2.0: https://github.com/facebookresearch/detr/)
## (4) 2020 Xingyi Zhou.(MIT License: https://github.com/xingyizhou/CenterTrack)
## (5) 2021 Wenhai Wang. (Apache License Version 2.0: https://github.com/whai362/PVT/blob/v2/LICENSE)
##
## TransCenter uses packages from
## (1) 2019 Charles Shang. (BSD 3-Clause Licence: https://github.com/CharlesShang/DCNv2)
## (2) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
import torch.nn as nn
# from dcn_v2 import DCN
from mmcv.ops import DeformConv2dPack as DCN

import math
import numpy as np
import torch.nn.functional as F
import torch
BN_MOMENTUM = 0.1
class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        # self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=1)
    @torch.cuda.amp.autocast()
    def forward(self, x):
        x_out = self.conv(x)
        # print("1", x.dtype)
        x_out = self.actf(x_out.type(x.dtype))
        # print("2", x.dtype)
        return x_out

DLA_NODE = {
    'dcn': (DeformConv, DeformConv)
}


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUpV3_bis(nn.Module):
    # bilinear upsampling version of IDA
    def __init__(self, o, channels, node_type=(DeformConv, DeformConv)):
        super(IDAUpV3_bis, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # no params

        for i in range(0, len(channels)):
            c = channels[i]
            if i == 0:
                node = node_type[1](c, o)
            else:
                node = node_type[1](c, c)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(endp-1, startp, -1):
            # print(f"layers[{i}] before:", layers[i].shape)
            layers[i] = self.up(layers[i])  # ch 256-> 256
            node = getattr(self, 'node_' + str(i))
            layers[i-1] = node(layers[i] + layers[i - 1])
        # layers[startp] = self.up(layers[startp])  # 256=>256
        node = getattr(self, 'node_' + str(startp))
        layers[startp] = node(layers[startp])
        return [layers[startp]]




