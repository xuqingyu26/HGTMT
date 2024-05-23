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
from util.image import color_aug, GaussianBlur
from util.image import gaussian_radius, draw_umich_gaussian
import copy
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm
import random
import torch
from scipy.spatial.distance import cdist


class GenericDataset_rgbt(data.Dataset):
    is_fusion_dataset = False
    default_resolution = None
    num_categories = None
    class_name = None
    # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
    # Not using 0 because 0 is used for don't care region and ignore loss.
    cat_ids = None
    max_objs = None
    num_joints = 17
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

    normMean = [0.2159822]
    normstd = [0.18660837]

    # normMean = [55.07553042763158]
    # normstd = [47.585183496689936]

    def __init__(self, opt=None, split=None, ann_path_r=None, ann_path_i=None, img_dir=None):
        super(GenericDataset_rgbt, self).__init__()
        if opt is not None and split is not None:
            self.split = split
            self.opt = opt
            self._data_rng = np.random.RandomState(123)

        if ann_path_r is not None and ann_path_i is not None and img_dir is not None:
            print('==> initializing {} data from {} and {}, \n images from {} ...'.format(
                split, ann_path_r, ann_path_i, img_dir))

            self.coco_r = coco.COCO(ann_path_r)
            self.coco_i = coco.COCO(ann_path_i)

            self.images_r = self.coco_r.getImgIds()
            self.images_i = self.coco_i.getImgIds()
            self.images = list(zip(self.images_r, self.images_i))
            # self.images = self.images[:10]

            if opt.tracking:
                if not ('videos' in self.coco_r.dataset):
                    self.fake_video_data()
                print('Creating video index!')

                self.video_to_images_i = defaultdict(list)
                self.video_to_images_r = defaultdict(list)
                for image_r, image_i in zip(self.coco_r.dataset['images'], self.coco_i.dataset['images']):
                    self.video_to_images_r[image_r['video_id']].append(image_r)
                    self.video_to_images_i[image_i['video_id']].append(image_i)

            self.img_dir = img_dir

            if opt.cache_mode:
                self.cache = {}
                print("caching data into memory...")
                for tmp_im_id in tqdm(self.images):
                    assert tmp_im_id not in self.cache.keys()

                    self.cache[tmp_im_id] = ([self._load_image_anns(tmp_im_id[0], self.coco_r, img_dir)],
                                             [self._load_image_anns(tmp_im_id[1], self.coco_i, img_dir)])
            else:
                self.cache = {}

        self.blur_aug = GaussianBlur(kernel_size=11)

    @staticmethod
    def letterbox(img_r, img_i, height, width,
                  color=(0, 0, 0)):  # resize a rectangular image to a padded rectangular
        shape = img_r.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        padding_mask = np.ones_like(img_r)
        img_r = cv2.resize(img_r, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img_i = cv2.resize(img_i, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        padding_mask = cv2.resize(padding_mask, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border

        img_r = cv2.copyMakeBorder(img_r, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color)  # padded rectangular
        img_i = cv2.copyMakeBorder(img_i, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color)  # padded rectangular
        padding_mask = cv2.copyMakeBorder(padding_mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                          value=color)  # padded rectangular
        return img_r, img_i, padding_mask, ratio, dw, dh

    @staticmethod
    def random_affine(img_r, img_i, pad_img, targets_r=None, targets_i=None, degrees=(-10, 10), translate=(.1, .1),
                      scale=(.9, 1.1),
                      shear=(-2, 2),
                      borderValue=(0, 0, 0), M=None, a=None, anns_r=None, anns_i=None):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        border = 0  # width of added border (optional)
        height = img_r.shape[0]
        width = img_r.shape[1]

        # print(img.shape)
        # print(pad_img.shape)

        assert img_r.shape == pad_img.shape

        # if M is None, get new M #
        if M is None:
            # Rotation and Scale
            R = np.eye(3)
            a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
            # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
            s = random.random() * (scale[1] - scale[0]) + scale[0]
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img_r.shape[1] / 2, img_r.shape[0] / 2), scale=s)

            # Translation
            T = np.eye(3)
            T[0, 2] = (random.random() * 2 - 1) * translate[0] * img_r.shape[0] + border  # x translation (pixels)
            T[1, 2] = (random.random() * 2 - 1) * translate[1] * img_r.shape[1] + border  # y translation (pixels)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

            M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

        imw_r = cv2.warpPerspective(img_r, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                    borderValue=borderValue)  # BGR order borderValue
        imw_i = cv2.warpPerspective(img_i, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                    borderValue=borderValue)  # BGR order borderValue

        pad_img = cv2.warpPerspective(pad_img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                      borderValue=borderValue)  # BGR order borderValue

        # Return warped points also
        def warp_target(targets, anns_i):
            if targets is not None:
                new_anns_i = []
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
                    # np.clip(xy[:, 0], 0, width, out=xy[:, 0])
                    # np.clip(xy[:, 2], 0, width, out=xy[:, 2])
                    # np.clip(xy[:, 1], 0, height, out=xy[:, 1])
                    # np.clip(xy[:, 3], 0, height, out=xy[:, 3])
                    w = xy[:, 2] - xy[:, 0]
                    h = xy[:, 3] - xy[:, 1]
                    area = w * h
                    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                    i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

                    # targets = targets[i]
                    # targets[:, :] = xy[i]
                    # targets = targets[targets[:, 0] < width]
                    # targets = targets[targets[:, 2] > 0]
                    # targets = targets[targets[:, 1] < height]
                    # targets = targets[targets[:, 3] > 0]
                    #
                    # anns = anns[i]
                    # anns = anns[targets[:, 0] < width]
                    # anns = anns[targets[:, 2] > 0]
                    # anns = anns[targets[:, 1] < height]
                    # anns = anns[targets[:, 3] > 0]

                    # apply labels to anns #
                    assert targets.shape[0] == len(anns_i)
                    for k in range(len(anns_i)):
                        if not i[k]:
                            continue
                        targets[k, :] = xy[k]
                        # print(k)
                        # print(targets[k, 0], width)
                        # print(targets[k, 2] )
                        # print(targets[k, 1], height )
                        # print(targets[k, 3])
                        new_ann = anns_i[k]
                        if targets[k, 0] < width and targets[k, 2] > 0 and targets[k, 1] < height and targets[k, 3] > 0:
                            # xyxy to xywh
                            new_ann['bbox'] = [targets[k, 0], targets[k, 1], targets[k, 2] - targets[k, 0],
                                               targets[k, 3] - targets[k, 1]]

                            new_anns_i.append(new_ann)
            return new_anns_i

        if targets_i is not None and targets_r is not None:
            new_anns_i = warp_target(targets_i, anns_i)
            new_anns_r = warp_target(targets_r, anns_r)
            return imw_r, imw_i, pad_img, new_anns_r, new_anns_i, M, a
        else:
            return imw_r, imw_i, pad_img

    @staticmethod
    def build_edge_index_full(n_p_crops, n_points):
        """
        build the edge_index of a fully connected bipartite graph for pyg graph convolution algorithms (a.k.a. a
        complete bipartite graph between the previous crops and the current centers). The graph is constructed by stacking
        previous crops on TOP of current centers, for example:
        [crop_0,
         crop_1,
         ...,
         crop_n,
         center_0,
         center_1,
         ...,
         center_m]

        :param n_p_crops: number of previous crops
        :param n_points: number of centers in the current image for detection
        :return: edge_index tensor of shape (2, n_edges)
        """
        all_inds = torch.arange(n_p_crops + n_points)
        p_crop_index = all_inds[:n_p_crops]
        n_points_index = all_inds[n_p_crops:]
        src, dst = torch.meshgrid(p_crop_index, n_points_index)
        edge_index_forward = torch.stack((src.flatten(), dst.flatten()))
        edge_index_backward = torch.stack((dst.flatten(), src.flatten()))
        edge_index = torch.cat((edge_index_forward, edge_index_backward), dim=1)
        return edge_index

    @staticmethod
    def build_edge_index_local(p_labels: np.array, box_length: int, fm_width: int, fm_height: int):
        """
        build the edge_index of a sparsely connected bipartite graph for pyg graph convolution algorithms. Now we only
        connect previous crops to current centers that are within a radius. The graph is constructed by stacking
        previous crops on TOP of current centers, for example:
        [crop_0,
         crop_1,
         ...,
         crop_n,
         center_0,
         center_1,
         ...,
         center_m]

        :param p_labels: tensor containing the location of the previous crops, in [frame, id, center_x, center_y, width, height] format
        :param box_length: pixel length of the square search area
        :param fm_width: feature map width
        :param fm_height: feature map height
        :return: edge_index tensor of shape (2, n_edges)
        """
        # find the pixel coordinates of the p_crops, and round them
        boxes = p_labels[:, 2:6].copy()
        boxes = (boxes * np.array([[fm_width, fm_height, fm_width, fm_height]])).astype(np.int32)

        # calcuate the search area around each box
        centers = boxes[:, :2]
        lefttop = centers - np.array([[box_length // 2, box_length // 2]])
        rightbottom = centers + np.array([[box_length // 2, box_length // 2]])
        search_regions = np.concatenate((lefttop, rightbottom), axis=1)
        search_regions[:, [0, 2]] = np.clip(search_regions[:, [0, 2]], 0, fm_width - 1)
        search_regions[:, [1, 3]] = np.clip(search_regions[:, [1, 3]], 0, fm_height - 1)
        # to torch.tensor
        search_regions = torch.from_numpy(search_regions)

        # convert search region boxes to indices on the feature map
        # construct a default search box for each previous crop
        default_boxes = torch.arange(box_length).repeat(1, box_length, 1)
        row_offsets = torch.arange(box_length) * fm_width
        row_offsets = row_offsets.reshape(1, -1, 1)
        default_boxes = default_boxes + row_offsets
        default_boxes = default_boxes.repeat(search_regions.shape[0], 1, 1)

        # add the left_top corner of each search region as index offset to the default search boxes
        # idx_offsets = search_regions[:, 0] * fm_width + search_regions[:, 1]  # this looks problematic
        idx_offsets = (search_regions[:, 1] - 1) * fm_width + search_regions[:, 0]
        idx_offsets = idx_offsets.unsqueeze(-1).unsqueeze(-1)
        n_points_index = default_boxes + idx_offsets
        n_points_index = n_points_index.flatten()
        # filter out those nodes that are out of scope
        n_points_index = n_points_index[(n_points_index >= 0) & (n_points_index < fm_width * fm_height)]
        n_points_index = torch.unique(n_points_index)
        # TODO: now the n_points_index by default does contain repeating nodes, as the search regions are very likely to
        # TODO: overlap. We will leave it as is right now, but we can experiment with removing the repeating nodes.

        # finally add the index offset for the previous crops, as they are placed before the features on the feature map
        # in node list
        n_points_index += len(p_labels)

        # get the node index of the p_crops in the node list
        p_crop_index = torch.arange(len(p_labels))

        # construct the edge index
        src, dst = torch.meshgrid(p_crop_index, n_points_index)
        edge_index_forward = torch.stack((src.flatten(), dst.flatten()))
        edge_index_backward = torch.stack((dst.flatten(), src.flatten()))
        edge_index = torch.cat((edge_index_forward, edge_index_backward), dim=1)

        return edge_index

    @staticmethod
    def build_edge_index_local_r(box_length: int, fm_width: int, fm_height: int):
        """
        build the edge_index of a sparsely connected bipartite graph for pyg graph convolution algorithms. Now we only
        connect previous crops to current centers that are within a radius. The graph is constructed by stacking
        previous crops on TOP of current centers, for example:
        [crop_0,
         crop_1,
         ...,
         crop_n,
         center_0,
         center_1,
         ...,
         center_m]

        :param p_labels: tensor containing the location of the previous crops, in [frame, id, center_x, center_y, width, height] format
        :param box_length: pixel length of the square search area
        :param fm_width: feature map width
        :param fm_height: feature map height
        :return: edge_index tensor of shape (2, n_edges)
        """
        # find the pixel coordinates of the p_crops, and round them
        num = fm_width * fm_height
        x, y = torch.meshgrid(torch.arange(fm_height), torch.arange(fm_width))

        centers = torch.stack((x.flatten(), y.flatten())).T

        centers = (np.array(centers) * np.array([[fm_width, fm_height]])).astype(np.int32)

        # calcuate the search area around each box

        lefttop = centers - np.array([[box_length // 2, box_length // 2]])
        rightbottom = centers + np.array([[box_length // 2, box_length // 2]])
        search_regions = np.concatenate((lefttop, rightbottom), axis=1)
        search_regions[:, [0, 2]] = np.clip(search_regions[:, [0, 2]], 0, fm_width - 1)
        search_regions[:, [1, 3]] = np.clip(search_regions[:, [1, 3]], 0, fm_height - 1)
        # to torch.tensor
        search_regions = torch.from_numpy(search_regions)

        # convert search region boxes to indices on the feature map
        # construct a default search box for each previous crop
        default_boxes = torch.arange(box_length).repeat(1, box_length, 1)
        row_offsets = torch.arange(box_length) * fm_width
        row_offsets = row_offsets.reshape(1, -1, 1)
        default_boxes = default_boxes + row_offsets
        default_boxes = default_boxes.repeat(search_regions.shape[0], 1, 1)

        # add the left_top corner of each search region as index offset to the default search boxes
        # idx_offsets = search_regions[:, 0] * fm_width + search_regions[:, 1]  # this looks problematic
        idx_offsets = (search_regions[:, 1] - 1) * fm_width + search_regions[:, 0]
        idx_offsets = idx_offsets.unsqueeze(-1).unsqueeze(-1)
        n_points_index = default_boxes + idx_offsets
        n_points_index = n_points_index.flatten()
        # filter out those nodes that are out of scope
        n_points_index = n_points_index[(n_points_index >= 0) & (n_points_index < fm_width * fm_height)]
        n_points_index = torch.unique(n_points_index)
        # TODO: now the n_points_index by default does contain repeating nodes, as the search regions are very likely to
        # TODO: overlap. We will leave it as is right now, but we can experiment with removing the repeating nodes.

        # finally add the index offset for the previous crops, as they are placed before the features on the feature map
        # in node list
        n_points_index += num
        # get the node index of the p_crops in the node list
        p_crop_index = torch.arange(num)

        # construct the edge index
        src, dst = torch.meshgrid(p_crop_index, n_points_index)
        edge_index_forward = torch.stack((src.flatten(), dst.flatten()))
        edge_index_backward = torch.stack((dst.flatten(), src.flatten()))
        edge_index = torch.cat((edge_index_forward, edge_index_backward), dim=1)

        return edge_index

    @staticmethod
    def build_edge_index_sparse(p_labels: np.array, d_labels: np.array, box_length: int, fm_width: int, fm_height: int):
        boxes = p_labels[:, 2:6].copy()
        boxes = (boxes * np.array([[fm_width, fm_height, fm_width, fm_height]])).astype(np.int32)
        centers = boxes[:, :2]

        d_boxes = d_labels[:, 2:6].copy()
        d_boxes = (d_boxes * np.array([[fm_width, fm_height, fm_width, fm_height]])).astype(np.int32)
        d_center = d_boxes[:, :2]

        p_index = torch.arange(len(p_labels))

        dist = cdist(centers, d_center, 'euclidean')
        edge_index_forward = (dist < box_length).nonzero()

        edge_index = torch.cat((edge_index_forward, torch.from_numpy(edge_index_forward.numpy()[::-1, :])), dim=1)
        return edge_index

    def __getitem__(self, index):
        ##### label == infrared
        opt = self.opt
        ([img_r, anns_r, img_info_r, img_path_r], [img_i, anns_i, img_info_i, img_path_i]) = self._load_data(index)
        img_blurred = False
        if self.opt.image_blur_aug and np.random.rand() < 0.1 and self.split == 'train':
            # print("blur image")
            img_r = self.blur_aug(img_r)
            img_blurred = True

        # get image height and width
        height, width = img_r.shape[0], img_r.shape[1]

        flipped = 0
        if self.split == 'train':
            # random flip #
            if np.random.random() < opt.flip:
                flipped = 1

        # flip image if flipped, reshape img to input size, get pad mask, if train do random affine, return updated img, pad_mask and anns.
        inp_r, inp_i, padding_mask, anns_input_r, anns_input_i, M, a, [ratio, padh, padw] = self._get_input(img_r,
                                                                                                            img_i,
                                                                                                            anns_r=copy.deepcopy(
                                                                                                                anns_r),
                                                                                                            anns_i=copy.deepcopy(
                                                                                                                anns_i),
                                                                                                            flip=flipped)
        ret = {'image_r': inp_r, 'image_i': inp_i, 'pad_mask': padding_mask.astype(np.bool)}
        # ret['orig_image'] = img
        # print(img.shape)

        # get pre info, pre info has the same transform then current info
        pre_cts_r, pre_track_ids_r = None, None
        pre_cts_i, pre_track_ids_i = None, None
        if opt.tracking:
            # randomly select a pre image with random interval

            pre_image_r, pre_image_i, pre_anns_r, pre_anns_i, frame_dist, pre_img_id_r, pre_img_id_i = self._load_pre_data(
                img_info_i['video_id'], img_info_i['frame_id'],
                img_info_i['sensor_id'] if 'sensor_id' in img_info_i else 1)

            if self.opt.image_blur_aug and img_blurred and self.split == 'train':
                # print("blur image")
                pre_image_r = self.blur_aug(pre_image_r)
                pre_image_i = self.blur_aug(pre_image_i)

            # if same_aug_pre and pre_img != curr_img, we use the same data aug for this pre image.
            if opt.same_aug_pre and frame_dist != 0:
                pre_M = M
                pre_a = a
            else:
                pre_M = None
                pre_a = None

            pre_img_r, pre_img_i, pre_padding_mask, pre_anns_input_r, pre_anns_input_i, pre_M, pre_a, _ = self._get_input(
                pre_image_r, pre_image_i,
                anns_r=copy.deepcopy(pre_anns_r), anns_i=copy.deepcopy(pre_anns_i),
                flip=flipped, M=pre_M, a=pre_a)
            # todo warning pre_cts is in the output image plane
            # todo warning pre_cts are with random shift noise, and random FPs are added with track_id = -2
            pre_cts_r, pre_track_ids_r = self._get_pre_dets(pre_anns_input_r)
            pre_cts_i, pre_track_ids_i = self._get_pre_dets(pre_anns_input_i)
            assert len(pre_cts_i) == len(pre_track_ids_i)
            ret['pre_img_r'] = pre_img_r
            ret['pre_img_i'] = pre_img_i
            ret['pre_pad_mask'] = pre_padding_mask.astype(np.bool)
            ret['frame_dist'] = frame_dist

        # init samples #
        self._init_ret(ret)

        # for #cts at time t
        # num_objs = min(len(anns_input), self.max_objs)
        # todo infrared label is the groundtruth
        num_objs_i = len(anns_input_i)
        num_objs_r = len(anns_input_r)
        curr_track_ids_cts_r = {}
        curr_track_ids_cts_i = {}
        for k in range(num_objs_i):
            ann_i = anns_input_i[k]
            cls_id = int(self.cat_ids[ann_i['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -999:
                continue
            bbox_i = self._coco_box_to_bbox(ann_i['bbox']).copy()
            bbox_i /= self.opt.down_ratio
            bbox_amodal_i = copy.deepcopy(bbox_i)
            bbox_i[[0, 2]] = np.clip(bbox_i[[0, 2]], 0, self.opt.output_w - 1)
            bbox_i[[1, 3]] = np.clip(bbox_i[[1, 3]], 0, self.opt.output_h - 1)

            if cls_id <= 0 or (not self.opt.ignoreIsCrowd and 'iscrowd' in ann_i and ann_i['iscrowd'] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox_i)
                continue

            # todo warning track_ids are ids at t-1
            self._add_instance(ret, k, cls_id, bbox_i, bbox_amodal_i, ann_i, curr_track_ids_cts_i, False)

        for k in range(num_objs_r):
            ann_r = anns_input_r[k]
            cls_id = int(self.cat_ids[ann_r['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -999:
                continue

            bbox_r = self._coco_box_to_bbox(ann_r['bbox']).copy()
            # down ratio to output size #
            bbox_r /= self.opt.down_ratio
            bbox_amodal_r = copy.deepcopy(bbox_r)
            bbox_r[[0, 2]] = np.clip(bbox_r[[0, 2]], 0, self.opt.output_w - 1)
            bbox_r[[1, 3]] = np.clip(bbox_r[[1, 3]], 0, self.opt.output_h - 1)

            if cls_id <= 0 or (not self.opt.ignoreIsCrowd and 'iscrowd' in ann_r and ann_r['iscrowd'] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox_r)
                continue

            self._add_instance(ret, k, cls_id, bbox_r, bbox_amodal_r, ann_r, curr_track_ids_cts_r, True)

        assert len(pre_cts_i) == len(pre_track_ids_i)
        for k, (pre_ct_r, pre_track_id_r) in enumerate(zip(pre_cts_r, pre_track_ids_r)):
            if 'tracking' in self.opt.heads:
                # if 'tracking' we produce ground-truth offset heatmap
                # if curr track id exists in pre track ids
                if pre_track_id_r in curr_track_ids_cts_r.keys():
                    # get pre center pos
                    ret['tracking_mask_r'][k] = 1  # todo warning: of the ordering of pre_cts
                    # pre_cts + ret['tracking'][k] = pre_ct (bring you to cur centers)
                    ret['tracking_r'][k] = curr_track_ids_cts_r[pre_track_id_r] - pre_ct_r
                # our random noise FPs or new objects at t => todo don't move?
                elif int(pre_track_id_r) > 0:
                    ret['tracking_mask_r'][k] = 1  # todo warning: of the ordering of pre_cts
                    ret['tracking_r'][k] = 0

        for k, (pre_ct_i, pre_track_id_i) in enumerate(zip(pre_cts_i, pre_track_ids_i)):
            if 'tracking' in self.opt.heads:
                # if 'tracking' we produce ground-truth offset heatmap
                # if curr track id exists in pre track ids
                if pre_track_id_i in curr_track_ids_cts_i.keys():
                    # get pre center pos
                    ret['tracking_mask_i'][k] = 1  # todo warning: of the ordering of pre_cts
                    # pre_cts + ret['tracking'][k] = pre_ct (bring you to cur centers)
                    ret['tracking_i'][k] = curr_track_ids_cts_i[pre_track_id_i] - pre_ct_i
                # our random noise FPs or new objects at t => todo don't move?
                elif int(pre_track_id_i) > 0:
                    ret['tracking_mask_i'][k] = 1  # todo warning: of the ordering of pre_cts
                    ret['tracking_i'][k] = 0

        ret['ratio'] = ratio
        ret['padw'] = padw
        ret['padh'] = padh
        ret['image_id_rgbt'] = self.images[index]
        ret['output_size'] = np.asarray([self.opt.output_h, self.opt.output_w])
        ret['orig_size'] = np.asarray([height, width])

        pad_pre_cts_i = np.zeros((self.max_objs, 2), dtype=np.float32)
        valid_num_pre_dets_i = 0
        if len(pre_cts_i) > 0:
            pre_cts_i = np.array(pre_cts_i)
            pad_pre_cts_i[:pre_cts_i.shape[0], :] = pre_cts_i
            valid_num_pre_dets_i = pre_cts_i.shape[0]

        ret['pre_cts_i'] = pad_pre_cts_i  # at output size = 1/4 input size
        ret["valid_num_pre_dets_i"] = valid_num_pre_dets_i

        pad_pre_cts_r = np.zeros((self.max_objs, 2), dtype=np.float32)
        valid_num_pre_dets_r = 0
        if len(pre_cts_r) > 0:
            pre_cts_r = np.array(pre_cts_r)
            pad_pre_cts_r[:pre_cts_r.shape[0], :] = pre_cts_r
            valid_num_pre_dets_r = pre_cts_r.shape[0]

        ret['pre_cts_r'] = pad_pre_cts_r  # at output size = 1/4 input size
        ret["valid_num_pre_dets_r"] = valid_num_pre_dets_r

        pad_pre_track_ids_r = np.zeros((self.max_objs), dtype=np.float32) - 3
        if len(pre_track_ids_r) > 0:
            pre_track_ids_r = np.array(pre_track_ids_r)
            pad_pre_track_ids_r[:pre_track_ids_r.shape[0]] = pre_track_ids_r
        assert pad_pre_track_ids_r.shape[0] == ret['tracking_mask_r'].shape[0]
        ret['pre_track_ids_r'] = pad_pre_track_ids_r.astype(np.int64)
        pad_pre_track_ids_i = np.zeros((self.max_objs), dtype=np.float32) - 3
        if len(pre_track_ids_i) > 0:
            pre_track_ids_i = np.array(pre_track_ids_i)
            pad_pre_track_ids_i[:pre_track_ids_i.shape[0]] = pre_track_ids_i
        assert pad_pre_track_ids_i.shape[0] == ret['tracking_mask_i'].shape[0]
        ret['pre_track_ids_i'] = pad_pre_track_ids_i.astype(np.int64)

        return ret

    def _load_image_anns(self, img_id, coco, img_dir):
        # print(coco.loadImgs(ids=[img_id]))
        img_info = coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        # if len(file_name) < 20:
        #     file_name = file_name+ '.jpg'
        img_path = os.path.join(img_dir, file_name)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        # bgr=> rgb
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, anns, img_info, img_path

    def _load_data(self, index):

        img_dir = self.img_dir
        img_id = self.images[index]
        if img_id in self.cache.keys():
            ([img_r, anns_r, img_info_r, img_path_r], [img_i, anns_i, img_info_i, img_path_i]) = self.cache[img_id]
        else:
            ([img_r, anns_r, img_info_r, img_path_r], [img_i, anns_i, img_info_i, img_path_i]) = (
                self._load_image_anns(img_id[0], self.coco_r, img_dir),
                self._load_image_anns(img_id[1], self.coco_i, img_dir))

        return ([img_r, anns_r, img_info_r, img_path_r], [img_i, anns_i, img_info_i, img_path_i])

    def _load_pre_data(self, video_id, frame_id, sensor_id=1):
        img_infos_r = self.video_to_images_r[video_id]
        img_infos_i = self.video_to_images_i[video_id]

        # If training, random sample nearby frames as the "previous" frame
        # If testing, get the exact prevous frame

        if 'train' in self.split:
            img_ids_r = [(img_info['id'], img_info['frame_id']) \
                         for img_info in img_infos_r \
                         if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist and \
                         (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
            img_ids_i = [(img_info['id'], img_info['frame_id']) \
                         for img_info in img_infos_i \
                         if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist and \
                         (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
        else:
            img_ids_r = [(img_info['id'], img_info['frame_id']) \
                         for img_info in img_infos_r \
                         if (img_info['frame_id'] - frame_id) == -1 and \
                         (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
            img_ids_i = [(img_info['id'], img_info['frame_id']) \
                         for img_info in img_infos_i \
                         if (img_info['frame_id'] - frame_id) == -1 and \
                         (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]

            if len(img_ids_r) == 0:
                img_ids_r = [(img_info['id'], img_info['frame_id']) \
                             for img_info in img_infos_r \
                             if (img_info['frame_id'] - frame_id) == 0 and \
                             (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
                img_ids_i = [(img_info['id'], img_info['frame_id']) \
                             for img_info in img_infos_i \
                             if (img_info['frame_id'] - frame_id) == 0 and \
                             (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]

        rand_id = np.random.choice(len(img_ids_r))

        img_id_r, pre_frame_id = img_ids_r[rand_id]
        img_id_i, pre_frame_id = img_ids_i[rand_id]
        frame_dist = abs(frame_id - pre_frame_id)
        # print(frame_dist)
        if (img_id_r, img_id_i) in self.cache.keys():
            # img, anns, _, _ = self.cache[(img_id_r, img_id_i)]
            ([img_r, anns_r, _, _], [img_i, anns_i, _, _]) = self.cache[
                (img_id_r, img_id_i)]
        else:
            # img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
            ([img_r, anns_r, _, _], [img_i, anns_i, _, _]) = (
                self._load_image_anns(img_id_r, self.coco_r, self.img_dir),
                self._load_image_anns(img_id_i, self.coco_i, self.img_dir))

        return img_r, img_i, anns_r, anns_i, frame_dist, img_id_r, img_id_i

    def _get_pre_dets(self, anns_input):
        hm_h, hm_w = self.opt.input_h, self.opt.input_w
        down_ratio = self.opt.down_ratio
        pre_cts, track_ids = [], []
        for ann in anns_input:
            cls_id = int(self.cat_ids[ann['category_id']])
            # if cls_id > self.opt.num_classes or cls_id <= -99 or \
            #         (not self.opt.ignoreIsCrowd and 'iscrowd' in ann and ann['iscrowd'] > 0):
            #     continue
            if cls_id > self.opt.num_classes or cls_id <= -99:
                print(cls_id)
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

                    pre_cts.append(ct2 / down_ratio)
                    track_ids.append(-2)
        return pre_cts, track_ids

    def _get_input(self, img_r, img_i, anns_r=None, anns_i=None, flip=0, M=None, a=None):
        img_r = img_r.copy()
        img_i = img_i.copy()
        h, w, _ = img_r.shape
        # reshape input to input_size #
        if flip:
            img_r = img_r[:, ::-1, :].copy()
            img_i = img_i[:, ::-1, :].copy()
        if self.split == 'train' and not self.opt.no_color_aug and np.random.rand() < 0.2:
            img_r = img_r.astype(np.float32) / 255.0
            color_aug(self._data_rng, img_r, self._eig_val, self._eig_vec)
            img_r = (img_r * 255.0).astype(np.uint8)
            # if self.opt.same_couple:
            #     img_i = img_r.copy()
        inp_r, inp_i, padding_mask, ratio, padw, padh = self.letterbox(img_r, img_i, self.opt.input_h, self.opt.input_w)

        # 1) to flip, 2) to resize and pad

        if anns_r is not None and anns_i is not None:
            def lf(anns_i, flip):
                labels = []
                for k in range(len(anns_i)):
                    # x1y1wh
                    bbox = anns_i[k]['bbox']
                    if flip:
                        bbox = [w - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

                    bbox[0] = ratio * bbox[0] + padw
                    bbox[1] = ratio * bbox[1] + padh
                    bbox[2] = ratio * bbox[2]
                    bbox[3] = ratio * bbox[3]
                    anns_i[k]['bbox'] = bbox

                    labels.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                labels = np.asarray(labels)
                return labels

            labels_r = lf(anns_r, flip)
            labels_i = lf(anns_i, flip)
        else:
            labels_r = None
            labels_i = None

        # if train, random affine #
        if self.split == 'train':
            assert anns_i is not None
            inp_r, inp_i, padding_mask, anns_r, anns_i, M, a = self.random_affine(inp_r, inp_i, pad_img=padding_mask,
                                                                                  targets_r=labels_r,
                                                                                  targets_i=labels_i,
                                                                                  degrees=(-5, 5),
                                                                                  translate=(0.10, 0.10),
                                                                                  scale=(0.70, 1.20), M=M, a=a,
                                                                                  anns_r=anns_r,
                                                                                  anns_i=anns_i)
        else:
            M = None
            a = None

        affine_padding_mask = padding_mask[:, :, 0]
        # print("np.max(affine_padding_mask) ", np.max(affine_padding_mask))
        # print("np.min(affine_padding_mask) ", np.min(affine_padding_mask))
        affine_padding_mask[affine_padding_mask > 0] = 1

        # norm
        inp_r = (inp_r.astype(np.float32) / 255.)
        inp_r = (inp_r - self.mean) / self.std
        inp_r = inp_r.transpose(2, 0, 1)
        inp_i = (inp_i.astype(np.float32) / 255.)
        inp_i = (inp_i - self.mean) / self.std
        inp_i = inp_i.transpose(2, 0, 1)

        return inp_r, inp_i, 1 - affine_padding_mask, anns_r, anns_i, M, a, [ratio, padh, padw]

    def _init_ret(self, ret):
        max_objs = self.max_objs * self.opt.dense_reg
        ret['hm_r'] = np.zeros(
            (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
            np.float32)
        ret['hm_i'] = np.zeros(
            (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
            np.float32)
        ret['ind_r'] = np.zeros((max_objs), dtype=np.int64)
        ret['ind_i'] = np.zeros((max_objs), dtype=np.int64)
        ret['cat_r'] = np.zeros((max_objs), dtype=np.int64)
        ret['mask_r'] = np.zeros((max_objs), dtype=np.float32)
        ret['cat_i'] = np.zeros((max_objs), dtype=np.int64)
        ret['mask_i'] = np.zeros((max_objs), dtype=np.float32)
        # xyh #
        ret['boxes_r'] = np.zeros(
            (max_objs, 4), dtype=np.float32)
        ret['boxes_mask_r'] = np.zeros(
            (max_objs), dtype=np.float32)
        ret['boxes_i'] = np.zeros(
            (max_objs, 4), dtype=np.float32)
        ret['boxes_mask_i'] = np.zeros(
            (max_objs), dtype=np.float32)

        ret['center_offset_r'] = np.zeros(
            (max_objs, 2), dtype=np.float32)
        ret['center_offset_i'] = np.zeros(
            (max_objs, 2), dtype=np.float32)

        regression_head_dims = {
            'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
            'nuscenes_att': 8, 'velocity': 3,
            'dep': 1, 'dim': 3, 'amodel_offset': 2, 'center_offset': 2}

        for head in regression_head_dims:
            if head in self.opt.heads:
                ret[head + '_r'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_i'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_mask_r'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_mask_i'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)

    def _ignore_region(self, region, ignore_val=1):
        np.maximum(region, ignore_val, out=region)

    def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
        # mask out crowd region, only rectangular mask is supported
        if cls_id == 0:  # ignore all classes
            self._ignore_region(ret['hm_r'][:, int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])
            self._ignore_region(ret['hm_i'][:, int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])
        else:
            # mask out one specific class
            self._ignore_region(ret['hm_r'][abs(cls_id) - 1,
                                int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])
            self._ignore_region(ret['hm_i'][abs(cls_id) - 1,
                                int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1])

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    # ret, k, cls_id, bbox_r, bbox_amodal_r, ann_r, curr_track_ids_cts_r, bbox_i, bbox_amodal_i, ann_i, curr_track_ids_cts_i
    def _add_instance(self, ret, k, cls_id, bbox, bbox_amodal, ann, curr_track_ids_cts, isV=True):

        if isV:
            bbox_r = bbox
            bbox_amodal_r = bbox_amodal
            ann_r = ann
            curr_track_ids_cts_r = curr_track_ids_cts
            h, w = bbox_amodal_r[3] - bbox_amodal_r[1], bbox_amodal_r[2] - bbox_amodal_r[0]
            h_clip, w_clip = bbox_r[3] - bbox_r[1], bbox_r[2] - bbox_r[0]
            if h_clip <= 0 or w_clip <= 0:
                return
            radius = gaussian_radius((math.ceil(h_clip), math.ceil(w_clip)))
            radius = max(0, int(radius))
            ct_r = np.array(
                [(bbox_r[0] + bbox_r[2]) / 2, (bbox_r[1] + bbox_r[3]) / 2], dtype=np.float32)
            curr_track_ids_cts_r[ann_r['track_id']] = ct_r
            ct_int_r = ct_r.astype(np.int32)
            if 'wh_r' in ret:
                # 'wh' = box_amodal size,of shape [num_objects, 2]
                ret['wh_r'][k] = 1. * w, 1. * h
                ret['wh_mask_r'][k] = 1
            ret['ind_r'][k] = ct_int_r[1] * self.opt.output_w + ct_int_r[0]
            ret['reg_r'][k] = ct_r - ct_int_r
            ret['reg_mask_r'][k] = 1
            ret['center_offset_r'][k] = 0.5 * (bbox_amodal_r[0] + bbox_amodal_r[2]) - ct_r[0], \
                                        0.5 * (bbox_amodal_r[1] + bbox_amodal_r[3]) - ct_r[1]
            ret['center_offset_mask_r'][k] = 1
            ret['hm_r'][cls_id - 1] = draw_umich_gaussian(ret['hm_r'][cls_id - 1], ct_int_r, radius)
            ret['boxes_r'][k] = np.asarray([0.5 * (bbox_amodal_r[0] + bbox_amodal_r[2]),
                                            0.5 * (bbox_amodal_r[1] + bbox_amodal_r[3]),
                                            (bbox_amodal_r[2] - bbox_amodal_r[0]),
                                            (bbox_amodal_r[3] - bbox_amodal_r[1])], dtype=np.float32)
            ret['boxes_r'][k][0::2] /= self.opt.output_w
            ret['boxes_r'][k][1::2] /= self.opt.output_h
            ret['boxes_mask_r'][k] = 1
            ret['cat_r'][k] = cls_id - 1
            ret['mask_r'][k] = 1
        else:
            bbox_i = bbox
            bbox_amodal_i = bbox_amodal
            ann_i = ann
            curr_track_ids_cts_i = curr_track_ids_cts
            h, w = bbox_amodal_i[3] - bbox_amodal_i[1], bbox_amodal_i[2] - bbox_amodal_i[0]
            h_clip, w_clip = bbox_i[3] - bbox_i[1], bbox_i[2] - bbox_i[0]
            if h_clip <= 0 or w_clip <= 0:
                return
            radius = gaussian_radius((math.ceil(h_clip), math.ceil(w_clip)))
            radius = max(0, int(radius))
            ct_i = np.array(
                [(bbox_i[0] + bbox_i[2]) / 2, (bbox_i[1] + bbox_i[3]) / 2], dtype=np.float32)
            curr_track_ids_cts_i[ann_i['track_id']] = ct_i
            ct_int_i = ct_i.astype(np.int32)
            if 'wh_i' in ret:
                ret['wh_i'][k] = 1. * w, 1. * h
                ret['wh_mask_i'][k] = 1
            # 'cat': categories of shape [num_objects], recording the cat id.

            ret['ind_i'][k] = ct_int_i[1] * self.opt.output_w + ct_int_i[0]
            ret['reg_i'][k] = ct_i - ct_int_i

            ret['reg_mask_i'][k] = 1
            ret['center_offset_i'][k] = 0.5 * (bbox_amodal_i[0] + bbox_amodal_i[2]) - ct_i[0], \
                                        0.5 * (bbox_amodal_i[1] + bbox_amodal_i[3]) - ct_i[1]
            ret['center_offset_mask_i'][k] = 1
            ret['hm_i'][cls_id - 1] = draw_umich_gaussian(ret['hm_i'][cls_id - 1], ct_int_i, radius)
            ret['boxes_i'][k] = np.asarray([0.5 * (bbox_amodal_i[0] + bbox_amodal_i[2]),
                                            0.5 * (bbox_amodal_i[1] + bbox_amodal_i[3]),
                                            (bbox_amodal_i[2] - bbox_amodal_i[0]),
                                            (bbox_amodal_i[3] - bbox_amodal_i[1])], dtype=np.float32)
            ret['boxes_i'][k][0::2] /= self.opt.output_w
            ret['boxes_i'][k][1::2] /= self.opt.output_h
            ret['boxes_mask_i'][k] = 1
            ret['cat_i'][k] = cls_id - 1
            ret['mask_i'][k] = 1

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
