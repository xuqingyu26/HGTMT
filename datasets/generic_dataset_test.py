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
import os
from util.misc import NestedTensor, read_MOT17det
import torch.utils.data as data
import sys

curr_pth = os.path.abspath(__file__)
curr_pth = "/".join(curr_pth.split("/")[:-3])
sys.path.append(curr_pth)
import copy
from PIL import Image, ImageDraw, ImageFont
import torchvision
import torch


class GenericDataset_val(data.Dataset):
    # to test
    def __init__(self, root, valset="val", train_ratio=0.5, select_seq=''):
        super(GenericDataset_val, self).__init__()

        self.default_resolution = [640, 1088]
        self.dets_path = 'det/det.txt'
        self.root = root
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._img_paths = []
        self.valset = valset
        self.train_ratio = train_ratio
        self.mytransform = torchvision.transforms.ToTensor()
        self.images_path = 'img1/'
        self.VidPubDet = {}
        if self.valset != 'test':
            self.root += '/train/'
        else:
            self.root += '/test/'
        video_folders = sorted(os.listdir(self.root))

        for video_name in video_folders:
            if select_seq not in video_name:
                continue
            data_path = os.path.join(self.root, video_name, self.images_path)  # + '/' + _data_path

            self.VidPubDet[video_name] = read_MOT17det(
                os.path.join(self.root, video_name, self.dets_path))


            # ordered frame loop
            imgs_list = sorted(os.listdir(data_path))
            seq_len = len(imgs_list)

            for i, im_name in enumerate(imgs_list):
                img_path = os.path.join(data_path, im_name)
                assert os.path.exists(img_path), \
                    f'Path does not exist: {img_path}'
                if self.valset == "val" and i >= 1.5 * (1.0 - self.train_ratio) * seq_len:  # val
                    self._img_paths.append(img_path)
                    # print("im_name", img_path)
                elif self.valset == "train" and i < (1.0 - self.train_ratio) * seq_len:  # train
                    self._img_paths.append(img_path)
                elif self.valset == "test":
                    self._img_paths.append(img_path)

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self._img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_name, video_name = img_path.split('/')[-1], img_path.split('/')[-3]
        h, w, _ = img.shape

        orig_size = torch.as_tensor([int(h), int(w)])
        orig_img = self.mytransform(img.copy())

        # reshape input to input_size #
        inp, padding_mask, ratio, padw, padh = self.letterbox(img, self.default_resolution[0], self.default_resolution[1])

        padding_mask = padding_mask[:, :, 0]
        padding_mask[padding_mask > 0] = 1
        padding_mask = 1 - padding_mask
        padding_mask = torch.from_numpy(padding_mask.astype(np.bool)).unsqueeze(0)
        inp = self.transforms(inp)

        return NestedTensor(inp.unsqueeze(0), padding_mask), [orig_size, im_name, video_name,
                                                                     orig_img.unsqueeze(0).float(),
                                                                     [ratio, padw, padh]]

    @staticmethod
    def letterbox(img, height=608, width=1088,
                  color=(0, 0, 0)):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)


        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        padding_mask = np.ones_like(img)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        padding_mask = cv2.copyMakeBorder(padding_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, padding_mask, ratio, dw, dh

    @staticmethod
    def letterboxV2(im, height=608, width=1088, color=(0, 0, 0), auto=True, scaleup=True, stride=32):
        # from https://github.com/ultralytics/yolov5/blob/6d9b99fc4d700ee9d3e52491b852a8067efadb40/utils/augmentations.py#L92
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = (height, width)
        # print(shape)
        # print(new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        # elif scaleFill:  # stretch
        #     dw, dh = 0.0, 0.0
        #     new_unpad = (new_shape[1], new_shape[0])
        #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        padding_mask = np.ones_like(im)
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_AREA)
            padding_mask = cv2.resize(padding_mask, new_unpad, interpolation=cv2.INTER_AREA)  # resized, no border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        padding_mask = cv2.copyMakeBorder(padding_mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                          value=color)  # padded rectangular

        return im, padding_mask, ratio, dw, dh



