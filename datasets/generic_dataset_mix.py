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
import math
import json
import cv2
import os
from collections import defaultdict
import pycocotools.coco as coco
import torch.utils.data as data
import sys

curr_pth = os.path.abspath(__file__)
curr_pth = "/".join(curr_pth.split("/")[:-3])
sys.path.append(curr_pth)
from util.image import flip, color_aug, GaussianBlur
from util.image import get_affine_transform, affine_transform
from util.image import gaussian_radius, draw_umich_gaussian
import copy
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm
import random


class GenericDataset(data.Dataset):
    is_fusion_dataset = False
    default_resolution = None
    num_categories = None
    class_name = None
    # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
    # Not using 0 because 0 is used for don't care region and ignore loss.
    cat_ids = None
    max_objs = None
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                [11, 12], [13, 14], [15, 16]]
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)

    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                        dtype=np.float32)
    _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    ignore_val = 1

    def __init__(self, opt=None, split=None, ann_path=None, img_dir=None,img_dir_ch=None, ann_path_ch=None):
        super(GenericDataset, self).__init__()
        if opt is not None and split is not None:
            self.split = split
            self.opt = opt
            self._data_rng = np.random.RandomState(123)

        if ann_path is not None and img_dir is not None:
            print('==> initializing {} data from {}, \n images from {} ...'.format(
                split, ann_path, img_dir))
            self.coco = coco.COCO(ann_path)
            self.images = self.coco.getImgIds()

            # ch #
            if ann_path_ch is not None:
                self.ch = coco.COCO(ann_path_ch)
                self.images_ch = self.ch.getImgIds()
            else:
                self.ch = None
                self.images_ch = []

            self.merged_images = self.images + self.images_ch

            if opt.tracking:
                if not ('videos' in self.coco.dataset):
                    self.fake_video_data()
                if self.ch is not None and not ('videos' in self.ch.dataset):
                    self.fake_video_data_ch()
                print('Creating video index!')
                self.video_to_images = defaultdict(list)
                for image in self.coco.dataset['images']:
                    self.video_to_images[image['video_id']].append(image)

                # ch #
                if self.ch is not None:
                    self.video_to_images_ch = defaultdict(list)
                    for image in self.ch.dataset['images']:
                        self.video_to_images_ch[image['video_id']].append(image)

            self.img_dir = img_dir
            self.img_dir_ch = img_dir_ch

            if opt.cache_mode:
                self.cache = {}
                print("caching data into memory...")
                for tmp_im_id in tqdm(self.merged_images):
                    img, anns, img_info, img_path = self._load_image_anns(tmp_im_id, self.coco, self.img_dir, self.ch,
                                                                          self.img_dir_ch)
                    assert tmp_im_id not in self.cache.keys()
                    self.cache[tmp_im_id] = [img, anns, img_info, img_path]
            else:
                self.cache = {}

        self.blur_aug = GaussianBlur(kernel_size=11)

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

        padding_mask = np.ones_like(img)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        padding_mask = cv2.resize(padding_mask, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        padding_mask = cv2.copyMakeBorder(padding_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, padding_mask, ratio, dw, dh

    @staticmethod
    def random_affine(img, pad_img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                      borderValue=(0, 0, 0), M=None, a=None, anns=None):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        border = 0  # width of added border (optional)
        height = img.shape[0]
        width = img.shape[1]

        # print(img.shape)
        # print(pad_img.shape)

        assert img.shape == pad_img.shape

        # if M is None, get new M #
        if M is None:
            # Rotation and Scale
            R = np.eye(3)
            a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
            # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
            s = random.random() * (scale[1] - scale[0]) + scale[0]
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

            # Translation
            T = np.eye(3)
            T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
            T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

            M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

        imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=borderValue)  # BGR order borderValue

        pad_img = cv2.warpPerspective(pad_img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=borderValue)  # BGR order borderValue

        # Return warped points also
        if targets is not None:
            new_anns = []
            if len(targets) > 0:
                n = targets.shape[0]
                points = targets.copy()
                area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

                # warp points
                xy = np.ones((n * 4, 3))
                xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = (xy @ M.T)[:, :2].reshape(n, 8)

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # apply angle-based reduction
                radians = a * math.pi / 180
                reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
                x = (xy[:, 2] + xy[:, 0]) / 2
                y = (xy[:, 3] + xy[:, 1]) / 2
                w = (xy[:, 2] - xy[:, 0]) * reduction
                h = (xy[:, 3] - xy[:, 1]) * reduction
                xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

                # reject warped points outside of image
                w = xy[:, 2] - xy[:, 0]
                h = xy[:, 3] - xy[:, 1]
                area = w * h
                ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)


                # apply labels to anns #
                assert targets.shape[0] == len(anns)
                for k in range(len(anns)):
                    if not i[k]:

                        continue
                    targets[k, :] = xy[k]
                    new_ann = anns[k]
                    if targets[k, 0] < width and targets[k, 2] > 0 and targets[k, 1] < height and targets[k, 3] > 0:
                        # xyxy to xywh
                        new_ann['bbox'] = [targets[k, 0], targets[k, 1], targets[k, 2] - targets[k, 0],
                                           targets[k, 3] - targets[k, 1]]

                        new_anns.append(new_ann)

            return imw, pad_img, new_anns, M, a
        else:
            return imw, pad_img

    def __getitem__(self, index):
        opt = self.opt
        img, anns, img_info, img_path = self._load_data(index)
        img_blurred = False
        if self.opt.image_blur_aug and np.random.rand() < 0.1 and self.split == 'train':
            # print("blur image")
            img = self.blur_aug(img)
            img_blurred = True

        # get image height and width
        height, width = img.shape[0], img.shape[1]

        flipped = 0
        if self.split == 'train':
            # random flip #
            if np.random.random() < opt.flip:
                flipped = 1
        # flip image if flipped, reshape img to input size, get pad mask, if train do random affine, return updated img, pad_mask and anns.
        inp, padding_mask, anns_input, M, a, [ratio, padh, padw] = self._get_input(img, anns=copy.deepcopy(anns),
                                                                                   flip=flipped)

        ret = {'image': inp, 'pad_mask': padding_mask.astype(np.bool)}
        # print(img.shape)
        # ret['orig_image'] = img

        # get pre info, pre info has the same transform then current info
        pre_cts, pre_track_ids = None, None
        if opt.tracking:
            # randomly select a pre image with random interval
            pre_image, pre_anns, frame_dist, pre_img_id = self._load_pre_data(
                img_info['video_id'], img_info['frame_id'],
                img_info['sensor_id'] if 'sensor_id' in img_info else 1)

            if self.opt.image_blur_aug and img_blurred and self.split == 'train':
                # print("blur image")
                pre_image = self.blur_aug(pre_image)

            # if same_aug_pre and pre_img != curr_img, we use the same data aug for this pre image.
            if opt.same_aug_pre and frame_dist != 0:
                pre_M = M
                pre_a = a
            else:
                pre_M = None
                pre_a = None

            # flip image if flipped, reshape img to input size, get pad mask, if train do random affine, return updated img, pad_mask and anns.
            # ret['pre_orig_image'] = pre_image
            pre_img, pre_padding_mask, pre_anns_input, pre_M, pre_a, _ = self._get_input(pre_image,
                                                                                         anns=copy.deepcopy(pre_anns),
                                                                                         flip=flipped, M=pre_M, a=pre_a)
            # todo pre_cts is in the output image plane
            pre_cts, pre_track_ids = self._get_pre_dets(pre_anns_input)

            ret['pre_img'] = pre_img
            ret['pre_pad_mask'] = pre_padding_mask.astype(np.bool)

        ### init samples
        self._init_ret(ret)

        num_objs = min(len(anns_input), self.max_objs)
        curr_track_ids_cts = {}
        for k in range(num_objs):
            ann = anns_input[k]
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -999:

                continue
            # get ground truth bbox in the output image plane,
            # bbox_amodal do not clip by ouput image size, bbox is clipped,
            # todo !!!warning!!! the function performs cxcy2xyxy

            bbox = self._coco_box_to_bbox(ann['bbox']).copy()
            # down ratio to output size #
            bbox /= self.opt.down_ratio
            bbox_amodal = copy.deepcopy(bbox)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)

            if cls_id <= 0 or (not self.opt.ignoreIsCrowd and 'iscrowd' in ann and ann['iscrowd'] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox)
                # print('mask ignore or crowd.')
                continue

            # todo warning track_ids are ids at t-1
            self._add_instance(ret, k, cls_id, bbox, bbox_amodal, ann, curr_track_ids_cts)

        assert len(pre_cts) == len(pre_track_ids)
        if 'tracking' in self.opt.heads:
            # if 'tracking' we produce ground-truth offset heatmap
            # if curr track id exists in pre track ids
            for k, (pre_ct, pre_track_id) in enumerate(zip(pre_cts, pre_track_ids)):

                if pre_track_id in curr_track_ids_cts.keys():
                    # get pre center pos
                    ret['tracking_mask'][k] = 1  # todo warning: of the ordering of pre_cts
                    # pre_cts + ret['tracking'][k] = pre_ct (bring you to cur centers)
                    ret['tracking'][k] = curr_track_ids_cts[pre_track_id] - pre_ct
                # our random noise FPs or new objects at t => todo don't move?
                elif pre_track_id > 0:
                    ret['tracking_mask'][k] = 1  # todo warning: of the ordering of pre_cts
                    ret['tracking'][k] = 0

        # # # # # plot inp, padding mask
        # cur_inp_to_plot = inp.copy().transpose(1, 2, 0)
        # cur_inp_to_plot *= self.std
        # cur_inp_to_plot += self.mean
        # cur_img_pil = Image.fromarray((cur_inp_to_plot*255).astype(np.uint8))
        # cur_img_draw = ImageDraw.Draw(cur_img_pil)
        # colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
        #           (70, 240, 240), (240, 50, 230),
        #           (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200),
        #           (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
        #           (255, 255, 255), (0, 0, 0)]
        #
        # for ann in anns_input:
        #     if ann['iscrowd'] == 1 and not opt.ignoreIsCrowd:
        #         color = (255, 0, 0, 255)
        #     else:
        #         color = (0, 255, 0, 255)
        #     bb_amodal = self._coco_box_to_bbox(ann['bbox']).copy()
        #     bb = np.array(bb_amodal, np.int).copy()
        #     cur_img_draw.rectangle([(bb[0], bb[1]), (bb[2], bb[3])], fill=None,
        #                             outline=color)
        #     cur_img_draw.text((bb[2], bb[3]), str(ann['track_id']), fill=color)
        #
        #     ct_int = [(bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2]
        #     # orig_img_draw.text((bb[2], bb[3]), str(label), fill=(0, 255, 0, 255))
        #     # orig_img_draw.text((bb[0], bb[1]), str(score), fill=(0, 255, 0, 255))
        #     r = 10
        #     leftUpPoint = (ct_int[0] - r, ct_int[1] - r)
        #     rightDownPoint = (ct_int[0] + r, ct_int[1] + r)
        #     twoPointList = [leftUpPoint, rightDownPoint]
        #     cur_img_draw.ellipse(twoPointList, fill=color)
        #
        #
        # # mask_pil = Image.fromarray((padding_mask*255).astype(np.uint8))
        # # mask_pil.save(f"/see_hm/mot/{img_info['id']:06}_pad.png")
        # # # #
        # # # # plot #
        #
        # # # # # # # plot inp, padding mask todo plot boxes
        # inp_to_plot = pre_img.copy().transpose(1, 2, 0)
        # inp_to_plot *= self.std
        # inp_to_plot += self.mean
        #
        # img_pil = Image.fromarray((inp_to_plot * 255).astype(np.uint8))
        # img_draw = ImageDraw.Draw(img_pil)
        #
        # for pre_ann in pre_anns_input:
        #     if pre_ann['iscrowd'] == 1 and not opt.ignoreIsCrowd:
        #         color = (255, 0, 0, 255)
        #     else:
        #         color = (0, 255, 0, 255)
        #     bb_amodal = self._coco_box_to_bbox(pre_ann['bbox']).copy()
        #
        #     bb = np.array(bb_amodal, np.int).copy()
        #
        #     img_draw.rectangle([(bb[0], bb[1]), (bb[2], bb[3])], fill=None,
        #                        outline=color)
        #     img_draw.text((bb[2], bb[3]), str(pre_ann['track_id']), fill=color)
        #
        # for k, (pre_ct, pre_track_id) in enumerate(zip(pre_cts, pre_track_ids)):
        #     if pre_track_id > 0:
        #         pre2cur_ct_int = np.array(pre_ct*self.opt.down_ratio, dtype=int)
        #         r = 5
        #         leftUpPoint = (pre2cur_ct_int[0] - r, pre2cur_ct_int[1] - r)
        #         rightDownPoint = (pre2cur_ct_int[0] + r, pre2cur_ct_int[1] + r)
        #         twoPointList = [leftUpPoint, rightDownPoint]
        #         img_draw.ellipse(twoPointList, fill=
        #         (colors[pre_track_id % len(colors)][0],
        #          colors[pre_track_id % len(colors)][1],
        #          colors[pre_track_id % len(colors)][2], 255))
        #
        #         pre2_cur_ct = self.opt.down_ratio*(ret['tracking'][k] + pre_ct)
        #
        #         pre2cur_ct_int = np.array(pre2_cur_ct, dtype=int)
        #
        #         leftUpPoint = (pre2cur_ct_int[0] - r, pre2cur_ct_int[1] - r)
        #         rightDownPoint = (pre2cur_ct_int[0] + r, pre2cur_ct_int[1] + r)
        #         twoPointList = [leftUpPoint, rightDownPoint]
        #         cur_img_draw.ellipse(twoPointList, fill=
        #         (colors[pre_track_id % len(colors)][0],
        #          colors[pre_track_id % len(colors)][1],
        #          colors[pre_track_id % len(colors)][2], 255))
        # #
        # img_pil.save(f"/see_hm/mot/{img_info['id']:06}_{pre_img_id:06}_pre.png")
        # cur_img_pil.save(f"/see_hm/mot/{img_info['id']:06}.png")
        # # pad_pil = Image.fromarray((pre_padding_mask * 255).astype(np.uint8))
        # # pad_pil.save(f"/see_hm/coco/{img_info['id']:06}_{pre_img_id:06}_pad_pre.png")
        # # print("###################")
        # # print('pre id: ', pre_img_id)
        # # print('curr id: ', img_info['id'])
        # # print("###################")
        # # print()
        # # # plot #
        assert img_info['id'] == self.merged_images[index]
        ret['ratio'] = ratio
        ret['padw'] = padw
        ret['padh'] = padh
        ret['image_id'] = img_info['id']
        ret['output_size'] = np.asarray([self.opt.output_h, self.opt.output_w])
        ret['orig_size'] = np.asarray([height, width])

        pad_pre_cts = np.zeros((self.max_objs, 2), dtype=np.float32)
        valid_num_pre_dets = 0
        if len(pre_cts) > 0:
            pre_cts = np.array(pre_cts)
            pad_pre_cts[:pre_cts.shape[0], :] = pre_cts
            valid_num_pre_dets = pre_cts.shape[0]
        else:
            print("pre_cts ", pre_cts)
            print("pre_track_ids", pre_track_ids)

        ret['pre_cts'] = pad_pre_cts  # at output size = 1/4 input size
        ret["valid_num_pre_dets"] = valid_num_pre_dets

        pad_pre_track_ids = np.zeros((self.max_objs), dtype=np.float32) - 3
        if len(pre_track_ids) > 0:
            pre_track_ids = np.array(pre_track_ids)
            pad_pre_track_ids[:pre_track_ids.shape[0]] = pre_track_ids
        assert pad_pre_track_ids.shape[0] == ret['tracking_mask'].shape[0]
        ret['pre_track_ids'] = pad_pre_track_ids.astype(np.int64)  # at output size = 1/4 input size

        return ret

    def _load_image_anns(self, img_id, coco, img_dir, ch, img_dir_ch):
        # print(coco.loadImgs(ids=[img_id]))
        if img_id in self.images:
            img_info = coco.loadImgs(ids=[img_id])[0]
            file_name = img_info['file_name']
            img_path = os.path.join(img_dir, file_name)
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        else:
            img_info = ch.loadImgs(ids=[img_id])[0]
            file_name = img_info['file_name']
            img_path = os.path.join(img_dir_ch, file_name)
            ann_ids = ch.getAnnIds(imgIds=[img_id])
            anns = copy.deepcopy(ch.loadAnns(ids=ann_ids))
        # bgr=> rgb
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, anns, img_info, img_path

    def _load_data(self, index):
        coco = self.coco
        img_dir = self.img_dir

        ch = self.ch
        img_dir_ch = self.img_dir_ch

        img_id = self.merged_images[index]
        if img_id in self.cache.keys():
            img, anns, img_info, img_path = self.cache[img_id]
        else:
            img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir, ch, img_dir_ch)

        return img, anns, img_info, img_path

    def _load_pre_data(self, video_id, frame_id, sensor_id=1):
        if video_id in self.video_to_images:
            img_infos = self.video_to_images[video_id]
        else:
            img_infos = self.video_to_images_ch[video_id]
        # If training, random sample nearby frames as the "previous" frame
        # If testing, get the exact prevous frame
        if 'train' in self.split:
            img_ids = [(img_info['id'], img_info['frame_id']) \
                       for img_info in img_infos \
                       if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist and \
                       (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
        else:
            img_ids = [(img_info['id'], img_info['frame_id']) \
                       for img_info in img_infos \
                       if (img_info['frame_id'] - frame_id) == -1 and \
                       (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
            if len(img_ids) == 0:
                img_ids = [(img_info['id'], img_info['frame_id']) \
                           for img_info in img_infos \
                           if (img_info['frame_id'] - frame_id) == 0 and \
                           (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
        rand_id = np.random.choice(len(img_ids))

        img_id, pre_frame_id = img_ids[rand_id]
        frame_dist = abs(frame_id - pre_frame_id)
        # print(frame_dist)
        if img_id in self.cache.keys():
            img, anns, _, _ = self.cache[img_id]
        else:
            img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir, self.ch, self.img_dir_ch)

        return img, anns, frame_dist, img_id

    def _get_pre_dets(self, anns_input):
        hm_h, hm_w = self.opt.input_h, self.opt.input_w
        down_ratio = self.opt.down_ratio
        pre_cts, track_ids = [], []
        for ann in anns_input:
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -99 or \
                    (not self.opt.ignoreIsCrowd and 'iscrowd' in ann and ann['iscrowd'] > 0):
                continue
            bbox = self._coco_box_to_bbox(ann['bbox'])
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct0 = ct.copy()
                # conf = 1
                # add some noise to ground-truth pre info
                ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
                pre_cts.append(ct / down_ratio)
                # conf = 1 if np.random.random() > self.opt.lost_disturb else 0
                # ct_int = ct.astype(np.int32)
                # if conf == 0:
                #     pre_cts.append(ct / down_ratio)
                # else:
                #     pre_cts.append(ct0 / down_ratio)
                # conf == 0, lost hm, FN
                track_ids.append(ann['track_id'] if 'track_id' in ann else -1)

                # false positives disturb
                if np.random.random() < self.opt.fp_disturb:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h

                    pre_cts.append(ct2/down_ratio)
                    track_ids.append(-2)
        return pre_cts, track_ids

    def _get_input(self, img, anns=None, flip=0, M=None, a=None):
        img = img.copy()
        h, w, _ = img.shape
        # reshape input to input_size #
        if flip:
            img = img[:, ::-1, :].copy()
        if self.split == 'train' and not self.opt.no_color_aug and np.random.rand() < 0.2:
            img = img.astype(np.float32)/255.0
            color_aug(self._data_rng, img, self._eig_val, self._eig_vec)
            img = (img*255.0).astype(np.uint8)
        inp, padding_mask, ratio, padw, padh = self.letterbox(img, self.opt.input_h, self.opt.input_w)

        # 1) to flip, 2) to resize and pad

        if anns is not None:
            labels = []
            for k in range(len(anns)):
                # x1y1wh
                bbox = anns[k]['bbox']
                if flip:
                    bbox = [w - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

                bbox[0] = ratio * bbox[0] + padw
                bbox[1] = ratio * bbox[1] + padh
                bbox[2] = ratio * bbox[2]
                bbox[3] = ratio * bbox[3]
                anns[k]['bbox'] = bbox

                labels.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels = np.asarray(labels)
        else:
            labels = None

        # if train, random affine #
        if self.split == 'train':
            assert anns is not None
            inp, padding_mask, anns, M, a = self.random_affine(inp, pad_img=padding_mask, targets=labels,
                                                               degrees=(-5, 5), translate=(0.10, 0.10),
                                                               scale=(0.70, 1.20), M=M, a=a, anns=anns)
        else:
            M = None
            a = None

        affine_padding_mask = padding_mask[:, :, 0]
        # print("np.max(affine_padding_mask) ", np.max(affine_padding_mask))
        # print("np.min(affine_padding_mask) ", np.min(affine_padding_mask))
        affine_padding_mask[affine_padding_mask > 0] = 1

        # norm
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp, 1 - affine_padding_mask, anns, M, a, [ratio, padh, padw]

    def _init_ret(self, ret):
        max_objs = self.max_objs * self.opt.dense_reg
        ret['hm'] = np.zeros(
            (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
            np.float32)
        ret['ind'] = np.zeros((max_objs), dtype=np.int64)
        ret['cat'] = np.zeros((max_objs), dtype=np.int64)
        ret['mask'] = np.zeros((max_objs), dtype=np.float32)
        # xyh #
        ret['boxes'] = np.zeros(
            (max_objs, 4), dtype=np.float32)
        ret['boxes_mask'] = np.zeros(
            (max_objs), dtype=np.float32)

        ret['center_offset'] = np.zeros(
            (max_objs, 2), dtype=np.float32)

        regression_head_dims = {
            'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
            'nuscenes_att': 8, 'velocity': 3,
            'dep': 1, 'dim': 3, 'amodel_offset': 2, 'center_offset': 2}

        for head in regression_head_dims:
            if head in self.opt.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_mask'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)

    def _ignore_region(self, region, ignore_val=1):
        np.maximum(region, ignore_val, out=region)

    def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
        # mask out crowd region, only rectangular mask is supported
        if cls_id == 0:  # ignore all classes
            self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])
        else:
            # mask out one specific class
            self._ignore_region(ret['hm'][abs(cls_id) - 1,
                                int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox


    def _add_instance(
            self, ret, k, cls_id, bbox, bbox_amodal, ann, curr_track_ids_cts):

        # box is in the output image plane, add it to gt heatmap
        h, w = bbox_amodal[3] - bbox_amodal[1], bbox_amodal[2] - bbox_amodal[0]
        h_clip, w_clip = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h_clip <= 0 or w_clip <= 0:
            return
        # print(k)
        radius = gaussian_radius((math.ceil(h_clip), math.ceil(w_clip)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        # int(ct)
        curr_track_ids_cts[ann['track_id']] = ct
        ct_int = ct.astype(np.int32)

        # 'cat': categories of shape [num_objects], recording the cat id.
        ret['cat'][k] = cls_id - 1
        # 'mask': mask of shape [num_objects], if mask == 1, to train, if mask == 0, not to train.
        ret['mask'][k] = 1
        if 'wh' in ret:
            # 'wh' = box_amodal size,of shape [num_objects, 2]
            ret['wh'][k] = 1. * w, 1. * h
            ret['wh_mask'][k] = 1
        # 'ind' of shape [num_objects],
        # indicating the position of the object = y*W_output + x in a heatmap of shape [out_h, out_w] #todo warning CT_INT
        ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
        # the .xxx part of the kpts
        ret['reg'][k] = ct - ct_int
        ret['reg_mask'][k] = 1

        # center_offset
        ret['center_offset'][k] = 0.5 * (bbox_amodal[0] + bbox_amodal[2]) - ct[0], \
                                  0.5 * (bbox_amodal[1] + bbox_amodal[3]) - ct[1]

        ret['center_offset_mask'][k] = 1

        # ad pts to ground-truth heatmap
        # print("ct_int", ct_int)

        draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

        # cx, cy, w, h
        # clipped box
        # ret['boxes'][k] = np.asarray([ct[0], ct[1], w, h], dtype=np.float32)
        ret['boxes'][k] = np.asarray([0.5 * (bbox_amodal[0] + bbox_amodal[2]),
                                      0.5 * (bbox_amodal[1] + bbox_amodal[3]),
                                      (bbox_amodal[2] - bbox_amodal[0]),
                                      (bbox_amodal[3] - bbox_amodal[1])], dtype=np.float32)

        # cx, cy, w, h / output size
        ret['boxes'][k][0::2] /= self.opt.output_w
        ret['boxes'][k][1::2] /= self.opt.output_h
        ret['boxes_mask'][k] = 1

    def fake_video_data(self):
        self.coco.dataset['videos'] = []
        for i in range(len(self.coco.dataset['images'])):
            img_id = self.coco.dataset['images'][i]['id']
            self.coco.dataset['images'][i]['video_id'] = img_id
            self.coco.dataset['images'][i]['frame_id'] = 1
            self.coco.dataset['videos'].append({'id': img_id})

        if not ('annotations' in self.coco.dataset):
            return

        for i in range(len(self.coco.dataset['annotations'])):
            self.coco.dataset['annotations'][i]['track_id'] = i + 1

    def fake_video_data_ch(self):
        self.ch.dataset['videos'] = []
        for i in range(len(self.ch.dataset['images'])):
            img_id = self.ch.dataset['images'][i]['id']
            self.ch.dataset['images'][i]['video_id'] = img_id
            self.ch.dataset['images'][i]['frame_id'] = 1
            self.ch.dataset['videos'].append({'id': img_id})