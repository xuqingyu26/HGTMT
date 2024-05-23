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

import json
import os
try:
  from .generic_dataset_mix import GenericDataset
except:
  from generic_dataset_mix import GenericDataset


class MOT17(GenericDataset):
  num_classes = 1
  num_joints = 17
  default_resolution = [640, 1088]
  max_objs = 500
  class_name = ['person']
  cat_ids = {1: 1}

  def __init__(self, opt, split):
    super(MOT17, self).__init__()
    data_dir = opt.data_dir
    data_dir_ch = opt.data_dir_ch

    if split == 'test':
      img_dir = os.path.join(
        data_dir, 'test')
    else:
      img_dir = os.path.join(
        data_dir, 'train')



    if split == 'train':
      if opt.small:
        ann_path = os.path.join(data_dir, 'annotations_onlySDP', '{}_half.json').format(split)
        ann_path_ch = os.path.join(data_dir_ch, 'annotations_small', '{}.json').format(split)
      else:
        ann_path = os.path.join(data_dir, 'annotations_onlySDP', '{}.json').format(split)
        ann_path_ch = os.path.join(data_dir_ch, 'annotations_mix', '{}.json').format(split)

      img_dir_ch = os.path.join(
      data_dir_ch, 'Images')
    else:
      ann_path = os.path.join(data_dir, 'annotations_onlySDP',
                              '{}_half.json').format(split)

      ann_path_ch = None
      img_dir_ch = None

    print('==> initializing MOT17 {} data and CH {} data.'.format(split, split))

    self.images = None
    # load image list and coco
    super(MOT17, self).__init__(opt, split, ann_path, img_dir, img_dir_ch=img_dir_ch, ann_path_ch=ann_path_ch)

    self.num_samples = len(self.merged_images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))


  def __len__(self):
    return self.num_samples

