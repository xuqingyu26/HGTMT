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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from util.image import transform_preds_with_trans, get_affine_transform
import torch

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)


def generic_post_process(opt, dets, pre_cts, dws, dhs, ratios, filter_by_scores=0.3):
  if not ('scores' in dets):
    return [{}], [{}]

  ret = []
  pre_ret = []
  assert len(dws) == len(dhs) == len(ratios) == len(dets['scores'])
  # batch #
  for i in range(len(dets['scores'])):

    if 'tracking' in dets:
      pre_item = {}
      # B,M,2
      # print("pre_cts.shape ", pre_cts.shape)
      # print("dets['tracking'].shape ", dets['tracking'].shape)
      assert pre_cts.shape == dets['tracking'].shape

      # displacement to original image space
      # print(pre_cts.device)
      # print(dets['tracking'].device)
      tracking = opt.down_ratio * (dets['tracking'][i] + pre_cts[i])

      tracking[:, 0] -= dws[i]
      tracking[:, 1] -= dhs[i]
      tracking /= ratios[i]

      pre_item['pre2cur_cts'] = tracking  # ct in the ct int in original image plan
      pre_ret.append(pre_item)

    preds = []
    # number detections #
    for j in range(len(dets['scores'][i])):

      if dets['scores'][i][j] < filter_by_scores:
        # because dets['scores'][i] is descending ordered, if dets['scores'][i][j] < filter_by_scores,
        # then dets['scores'][i][j+n] < filter_by_scores , so we can safely "break" here.
        break

      # print("I am here.", filter_by_scores)
      item = {}
      item['score'] = dets['scores'][i][j]
      item['class'] = int(dets['clses'][i][j]) + 1

      item['ct'] = opt.down_ratio*dets['cts'][i][j].clone()
      item['ct'][0] -= dws[i]
      item['ct'][1] -= dhs[i]
      item['ct'] /= ratios[i]

      if 'bboxes' in dets:
        #xyxy
        bbox = opt.down_ratio*dets['bboxes'][i][j]
        if opt.clip:
          bbox[0::2] = torch.clamp(bbox[0::2], min=0, max=opt.input_w-1)
          bbox[1::2] = torch.clamp(bbox[1::2], min=0, max=opt.input_h-1)
        bbox[0::2] -= dws[i]
        bbox[1::2] -= dhs[i]
        bbox /= ratios[i]
        item['bbox'] = bbox

      preds.append(item)

    ret.append(preds)
  
  return ret, pre_ret