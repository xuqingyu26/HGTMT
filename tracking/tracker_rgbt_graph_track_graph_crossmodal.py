import copy
import sys
import os
import csv

# dirty insert path #
cur_path = os.path.realpath(__file__)
cur_dir = "/".join(cur_path.split('/')[:-2])
sys.path.insert(0, cur_dir)

import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import cv2
from PIL import Image, ImageDraw, ImageFont
from util.tracker_util import bbox_overlaps

from torchvision.ops.boxes import clip_boxes_to_image, nms
import lap
from post_processing.decode import generic_decode
from util import box_ops
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch
from functools import partial
from models.PVT_v2 import PyramidVisionTransformerV2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib


class Tracker:
    """The main tracking file, here is where magic happens."""

    def __init__(self, obj_detect, reid_network, flownet, tracker_cfg, postprocessor=None,
                 main_args=None):

        self.obj_detect = obj_detect
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']

        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model_cfg = tracker_cfg['motion_model']
        self.postprocessor = postprocessor
        self.main_args = main_args
        self.birth_active = main_args.birth_active

        self.inactive_tracks_r = []
        self.inactive_tracks_i = []
        self.track_num_r = 0
        self.track_num_i = 0
        self.im_index = 0
        self.results_r = {}
        self.results_i = {}
        self.img_features = None
        self.encoder_pos_encoding = None
        self.transforms = transforms.ToTensor()
        self.last_image = None
        self.pre_sample_r = None
        self.pre_sample_i = None
        self.sample_r = None
        self.sample_i = None

        self.pre_img_features = None
        self.pre_encoder_pos_encoding = None
        self.flow = None
        self.det_thresh = main_args.det_thresh  # share detect threshold
        self.ssim = []

    def reset(self):
        self.tracks_r = []
        self.tracks_i = []
        self.inactive_tracks_r = []
        self.inactive_tracks_i = []
        self.last_image_r = None
        self.last_image_i = None
        self.pre_sample_r = None
        self.pre_sample_i = None
        self.obj_detect.pre_memory_r = None
        self.obj_detect.pre_memory_i = None
        self.sample_r = None
        self.sample_i = None
        self.pre_img_features_r = None
        self.pre_img_features_i = None
        self.pre_encoder_pos_encoding = None
        self.flow = None
        self.obj_detect.masks_flatten = None
        self.track_num_r = 0
        self.results_r = {}
        self.im_index = 0
        self.track_num_i = 0
        self.results_i = {}

    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []

        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)

        # matches = [[match_row_idx, match_column_idx]...], it gives you all the matches (assignments)
        # unmatched_a gives all the unmatched row indexes
        # unmatched_b gives all the unmatched column indexes
        return matches, unmatched_a, unmatched_b

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features, new_det_cls, isV=True):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        if isV:
            for i in range(num_new):
                self.tracks_r.append(Track(
                    new_det_pos[i].view(1, -1),
                    new_det_scores[i],
                    new_det_cls[i],
                    self.track_num_r + i,
                    new_det_features[i].view(1, -1),
                    self.inactive_patience,
                    self.max_features_num,
                    self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
                ))
            self.track_num_r += num_new
        else:
            for i in range(num_new):
                self.tracks_i.append(Track(
                    new_det_pos[i].view(1, -1),
                    new_det_scores[i],
                    new_det_cls[i],
                    self.track_num_i + i,
                    new_det_features[i].view(1, -1),
                    self.inactive_patience,
                    self.max_features_num,
                    self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
                ))
            self.track_num_i += num_new

    def dual_tracks_dets_matching_tracking(self, raw_dets, raw_scores, pre2cur_cts, pos, gather_feat, reid_cts,
                                           raw_det_feats, raw_cls, hs_m=None, isV=True):

        if pos is None:
            pos = self.get_pos(isV).clone()

        # iou matching #
        if len(pre2cur_cts.shape) == 3:
            pre2cur_cts = pre2cur_cts.squeeze()
        assert pos.nelement() > 0 and pos.shape[0] == pre2cur_cts.shape[0]
        # todo we can directly output warped_pos for faster inference #
        if raw_dets.nelement() > 0:
            assert raw_dets.shape[0] == raw_scores.shape[0]
            pos_w = pos[:, [2]] - pos[:, [0]]
            pos_h = pos[:, [3]] - pos[:, [1]]

            warped_pos = torch.cat([pre2cur_cts[:, [0]] - 0.5 * pos_w,
                                    pre2cur_cts[:, [1]] - 0.5 * pos_h,
                                    pre2cur_cts[:, [0]] + 0.5 * pos_w,
                                    pre2cur_cts[:, [1]] + 0.5 * pos_h], dim=1)

            # index low-score dets #
            raw_det_feats = F.grid_sample(raw_det_feats, reid_cts.unsqueeze(0).unsqueeze(0),
                                          mode='bilinear', padding_mode='zeros', align_corners=False)[0, :, 0,
                            :].transpose(0, 1)
            gnn_det_feats = F.grid_sample(hs_m, reid_cts.unsqueeze(0).unsqueeze(0),
                                          mode='bilinear', padding_mode='zeros', align_corners=False)[0, :, 0,
                            :].transpose(0, 1)
            inds_low = raw_scores > self.main_args.track_thresh_low
            inds_high = raw_scores < self.main_args.track_thresh_high
            inds_second = torch.logical_and(inds_low, inds_high)
            dets_second = raw_dets[inds_second]
            scores_second = raw_scores[inds_second]
            cls_second = raw_cls[inds_second]
            reid_cts_second = reid_cts[inds_second]
            second_det_feats = raw_det_feats[inds_second]

            # index high-score dets #
            remain_inds = raw_scores > self.main_args.track_thresh_high
            dets = raw_dets[remain_inds]
            det_feats = raw_det_feats[remain_inds]
            scores_keep = raw_scores[remain_inds]
            cls_keep = raw_cls[remain_inds]
            reid_cts_keep = reid_cts[remain_inds]
            pre_dets = []
            tracks = self.tracks_r if isV else self.tracks_i
            for t in tracks:
                pre_dets.append(t.pos)
            pre_dets = torch.cat(pre_dets, dim=0)
            # Step 1: first assignment #
            if len(dets) > 0:

                assert dets.shape[0] == scores_keep.shape[0]

                sim_mat = 1
                # matching with gIOU
                ##################
                # data_list = []
                # dets_cts = torch.zeros([dets.shape[0], 2]).to(dets.device)
                # dets_cts[:, 0] = (dets[:, 0] + dets[:, 2]) / 2
                # dets_cts[:, 1] = (dets[:, 1] + dets[:, 3]) / 2
                # edge_index = self.obj_detect.transformer.build_edge_index_sparse(pre2cur_cts, dets_cts, 15,
                #                                                                  fm_height=
                #                                                                  self.obj_detect.transformer.default_backbone_feature_resolution[
                #                                                                      0],
                #                                                                  fm_width=
                #                                                                  self.obj_detect.transformer.default_backbone_feature_resolution[
                #                                                                      1])  # need to change
                # edge_index = torch.cat([edge_index], dim=1).long()
                #
                # graph_nodes = torch.cat((gather_feat[0], gnn_det_feats[remain_inds].contiguous()), dim=0)
                # data_list.append(gData(x=graph_nodes, edge_index=edge_index))
                # graph = Batch.from_data_list(data_list)
                #
                # # pass through gnn
                # gnn_feat = graph.x
                # for gnn in self.obj_detect.transformer.gnn:
                #     gnn_out, (edge_index_r, alpha_r) = gnn(gnn_feat, graph.edge_index, return_attention_weights=True)
                #     if self.obj_detect.transformer.use_residual:
                #         gnn_feat = gnn_feat + gnn_out
                #     else:
                #         gnn_feat = gnn_out
                #
                # alpha = alpha_r[:edge_index.shape[1], :].mean(dim=1)
                # sim = torch.sparse_coo_tensor(edge_index_r[:, :len(alpha)], alpha,
                #                               (gather_feat[0].shape[0] + gnn_det_feats[remain_inds].shape[0],
                #                                gather_feat[0].shape[0] + gnn_det_feats[remain_inds].shape[0]))
                # sim_matrix = sim.to_dense()
                # sim_mat = sim_matrix[:gather_feat[0].shape[0], gather_feat[0].shape[0]:]
                # sim_mat = sim_mat + torch.ones_like(sim_mat) * (1e-4)
                ##################
                iou_dist = box_ops.generalized_box_iou(pre_dets, dets)

                # todo fuse with dets scores here.
                if self.main_args.fuse_scores:
                    iou_dist *= scores_keep[None, :]

                if self.main_args.fuse_gnn_sim:
                    iou_dist *= sim_mat

                iou_dist = 1 - iou_dist

                # todo recover inactive tracks here ?

                matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(),
                                                                       thresh=self.main_args.match_thresh)
                det_feats = raw_det_feats[remain_inds]

                # det_feats = F.grid_sample(raw_det_feats, reid_cts_keep.unsqueeze(0).unsqueeze(0),
                #                           mode='bilinear', padding_mode='zeros', align_corners=False)[0, :, 0,
                #             :].transpose(0, 1)

                if matches.shape[0] > 0:

                    # update track dets, scores #
                    for idx_track, idx_det in zip(matches[:, 0], matches[:, 1]):
                        if isV:
                            t = self.tracks_r[idx_track]
                        else:
                            t = self.tracks_i[idx_track]
                        t.pos = dets[[idx_det]]
                        t.add_features(det_feats[idx_det, :].view(1, -1))
                        t.birth_active = t.birth_active + 1
                        t.cls = cls_keep[[idx_det]]
                        t.score = scores_keep[[idx_det]]

                pos_birth = dets[u_detection, :]
                scores_birth = scores_keep[u_detection]
                dets_features_birth = det_feats[u_detection]
                cls_birth = cls_keep[u_detection]


            else:
                # no detection, kill all
                if isV:
                    u_track = list(range(len(self.tracks_r)))
                else:
                    u_track = list(range(len(self.tracks_i)))
                pos_birth = torch.zeros(size=(0, 4), device=pos.device, dtype=pos.dtype)
                scores_birth = torch.zeros(size=(0,), device=pos.device).long()
                cls_birth = torch.zeros(size=(0,), device=pos.device).int()
                dets_features_birth = torch.zeros(size=(0, 64), device=pos.device, dtype=pos.dtype)

            # Step 2: second assignment #
            # get remained tracks
            if len(u_track) > 0:

                if len(dets_second) > 0:
                    remained_tracks_pos = pre_dets[u_track]
                    track_indices = copy.deepcopy(u_track)
                    # print("track_indices: ", track_indices)
                    # matching with gIOU
                    iou_dist = 1 - box_ops.generalized_box_iou(remained_tracks_pos, dets_second)  # [0, 2]

                    matches, u_track_second, u_detection_second = self.linear_assignment(iou_dist.cpu().numpy(),
                                                                                         thresh=0.7)  # stricter with low-score dets

                    # update u_track here
                    u_track = [track_indices[t_idx] for t_idx in u_track_second]

                    if matches.shape[0] > 0:
                        # second_det_feats = F.grid_sample(raw_det_feats, reid_cts_second.unsqueeze(0).unsqueeze(0),
                        #                                  mode='bilinear', padding_mode='zeros', align_corners=False)[0,
                        #                    :, 0,
                        #                    :].transpose(0, 1)
                        # second_det_feats = F.grid_sample(reid_feats,
                        #                                  reid_cts_second[matches[:, 1]].unsqueeze(0).unsqueeze(0),
                        #                                  mode='bilinear', padding_mode='zeros', align_corners=False)[:,
                        #                    :, 0, :]
                        # update track dets, scores #
                        for cc, (idx_match, idx_det) in enumerate(zip(matches[:, 0], matches[:, 1])):
                            idx_track = track_indices[idx_match]
                            # print("low score match:", idx_track)
                            if isV:
                                t = self.tracks_r[idx_track]
                            else:
                                t = self.tracks_i[idx_track]
                            t.pos = dets_second[[idx_det]]
                            gather_feat_t = second_det_feats[cc, :]
                            t.add_features(gather_feat_t.view(1, -1))
                            t.birth_active = t.birth_active + 1
                            t.score = scores_second[[idx_det]]
                            t.cls = cls_second[[idx_det]]
        else:
            # no detection, kill all
            if isV:
                u_track = list(range(len(self.tracks_r)))
            else:
                u_track = list(range(len(self.tracks_i)))
            pos_birth = torch.zeros(size=(0, 4), device=pos.device, dtype=pos.dtype)
            scores_birth = torch.zeros(size=(0,), device=pos.device).long()
            cls_birth = torch.zeros(size=(0,), device=pos.device).int()
            dets_features_birth = torch.zeros(size=(0, 64), device=pos.device, dtype=pos.dtype)

        if isV:
            self.new_tracks = []
            for i, t in enumerate(self.tracks_r):
                if i in u_track:  # inactive
                    t.pos = t.last_pos[-1]
                    self.inactive_tracks_r += [t]
                else:  # keep
                    self.new_tracks.append(t)

            self.tracks_r = self.new_tracks
        else:
            self.new_tracks = []
            for i, t in enumerate(self.tracks_i):
                if i in u_track:  # inactive
                    t.pos = t.last_pos[-1]
                    self.inactive_tracks_i += [t]
                else:  # keep
                    self.new_tracks.append(t)

            self.tracks_i = self.new_tracks

        return [pos_birth, scores_birth, dets_features_birth, cls_birth]

    def get_pos(self, isV=-1):
        """Get the positions of all active tracks."""
        if len(self.tracks_r) == 1:
            pos_r = self.tracks_r[0].pos
        elif len(self.tracks_r) > 1:
            pos_r = torch.cat([t.pos for t in self.tracks_r], dim=0)
        else:
            pos_r = torch.zeros(size=(0, 4), device=self.sample_r.tensors.device).float()

        if len(self.tracks_i) == 1:
            pos_i = self.tracks_i[0].pos
        elif len(self.tracks_i) > 1:
            pos_i = torch.cat([t.pos for t in self.tracks_i], dim=0)
        else:
            pos_i = torch.zeros(size=(0, 4), device=self.sample_i.tensors.device).float()

        if isV == 0:
            return pos_r
        elif isV > 0:
            return pos_i
        else:
            return [pos_r, pos_i]

    def get_features(self, isV=True):
        """Get the features of all active tracks."""
        if isV:
            if len(self.tracks_r) == 1:
                features = self.tracks_r[0].features
            elif len(self.tracks_r) > 1:
                features = torch.cat([t.features for t in self.tracks_r], 0)
            else:
                features = torch.zeros(size=(0,), device=self.sample_r.tensors.device).float()
            return features
        else:
            if len(self.tracks_i) == 1:
                features = self.tracks_i[0].features
            elif len(self.tracks_i) > 1:
                features = torch.cat([t.features for t in self.tracks_i], 0)
            else:
                features = torch.zeros(size=(0,), device=self.sample_r.tensors.device).float()
            return features

    def get_inactive_features(self, isV=True):
        """Get the features of all inactive tracks."""
        if isV:
            if len(self.inactive_tracks_r) == 1:
                features = self.inactive_tracks_r[0].features
            elif len(self.inactive_tracks_r) > 1:
                features = torch.cat([t.features for t in self.inactive_tracks_r], 0)
            else:
                features = torch.zeros(0).cuda()
            return features
        else:
            if len(self.inactive_tracks_i) == 1:
                features = self.inactive_tracks_i[0].features
            elif len(self.inactive_tracks_i) > 1:
                features = torch.cat([t.features for t in self.inactive_tracks_i], 0)
            else:
                features = torch.zeros(0).cuda()
            return features

    # todo check the need to reid in both modalities
    def reid(self, new_det_pos, new_det_scores, new_det_features, new_det_cls, isV=True):
        """Tries to ReID inactive tracks with provided detections."""

        if self.do_reid:
            if isV:
                if len(self.inactive_tracks_r) > 0:
                    # calculate appearance distances
                    dist_mat, pos = [], []
                    for t in self.inactive_tracks_r:
                        dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                                                   for feat in new_det_features], dim=1))
                        pos.append(t.pos)
                    if len(dist_mat) > 1:
                        dist_mat = torch.cat(dist_mat, 0)
                        pos = torch.cat(pos, 0)
                    else:
                        dist_mat = dist_mat[0]
                        pos = pos[0]

                    # # calculate IoU distances
                    if self.main_args.iou_recover:
                        iou_dist = 1 - box_ops.generalized_box_iou(pos, new_det_pos)

                        matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(),
                                                                               thresh=self.main_args.match_thresh)
                    else:

                        # assigned by appearance
                        matches, u_track, u_detection = self.linear_assignment(dist_mat.cpu().numpy(),
                                                                               thresh=self.reid_sim_threshold)

                    assigned = []
                    remove_inactive = []
                    if matches.shape[0] > 0:
                        for r, c in zip(matches[:, 0], matches[:, 1]):
                            # inactive tracks reactivation #
                            if dist_mat[r, c] <= self.reid_sim_threshold or not self.main_args.iou_recover:
                                t = self.inactive_tracks_r[r]
                                self.tracks_r.append(t)
                                t.count_inactive = 0
                                t.pos = new_det_pos[c].view(1, -1)
                                t.reset_last_pos()
                                t.add_features(new_det_features[c].view(1, -1))
                                t.birth_active = t.birth_active + 1
                                t.cls = new_det_cls[c]
                                assigned.append(c)
                                remove_inactive.append(t)

                    for t in remove_inactive:
                        self.inactive_tracks_r.remove(t)

                    keep = [i for i in range(new_det_pos.size(0)) if i not in assigned]
                    if len(keep) > 0:
                        new_det_pos = new_det_pos[keep]
                        new_det_scores = new_det_scores[keep]
                        new_det_features = new_det_features[keep]
                        new_det_cls = new_det_cls[keep]
                    else:
                        new_det_pos = torch.zeros(size=(0, 4), device=self.sample_r.tensors.device).float()
                        new_det_scores = torch.zeros(size=(0,), device=self.sample_r.tensors.device).long()
                        new_det_cls = torch.zeros(size=(0,), device=self.sample_r.tensors.device).int()
                        new_det_features = torch.zeros(size=(0, 128), device=self.sample_r.tensors.device).float()
            else:
                if len(self.inactive_tracks_i) > 0:
                    # calculate appearance distances
                    dist_mat, pos = [], []
                    for t in self.inactive_tracks_i:
                        dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                                                   for feat in new_det_features], dim=1))
                        pos.append(t.pos)
                    if len(dist_mat) > 1:
                        dist_mat = torch.cat(dist_mat, 0)
                        pos = torch.cat(pos, 0)
                    else:
                        dist_mat = dist_mat[0]
                        pos = pos[0]

                    # # calculate IoU distances
                    if self.main_args.iou_recover:
                        iou_dist = 1 - box_ops.generalized_box_iou(pos, new_det_pos)

                        matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(),
                                                                               thresh=self.main_args.match_thresh)
                    else:

                        # assigned by appearance
                        matches, u_track, u_detection = self.linear_assignment(dist_mat.cpu().numpy(),
                                                                               thresh=self.reid_sim_threshold)

                    assigned = []
                    remove_inactive = []
                    if matches.shape[0] > 0:
                        for r, c in zip(matches[:, 0], matches[:, 1]):
                            # inactive tracks reactivation #
                            if dist_mat[r, c] <= self.reid_sim_threshold or not self.main_args.iou_recover:
                                t = self.inactive_tracks_i[r]
                                self.tracks_i.append(t)
                                t.count_inactive = 0
                                t.pos = new_det_pos[c].view(1, -1)
                                t.reset_last_pos()
                                t.add_features(new_det_features[c].view(1, -1))
                                t.birth_active = t.birth_active + 1
                                t.cls = new_det_cls[c]
                                assigned.append(c)
                                remove_inactive.append(t)

                    for t in remove_inactive:
                        self.inactive_tracks_i.remove(t)

                    keep = [i for i in range(new_det_pos.size(0)) if i not in assigned]
                    if len(keep) > 0:
                        new_det_pos = new_det_pos[keep]
                        new_det_scores = new_det_scores[keep]
                        new_det_features = new_det_features[keep]
                        new_det_cls = new_det_cls[keep]
                    else:
                        new_det_pos = torch.zeros(size=(0, 4), device=self.sample_r.tensors.device).float()
                        new_det_scores = torch.zeros(size=(0,), device=self.sample_r.tensors.device).long()
                        new_det_cls = torch.zeros(size=(0,), device=self.sample_r.tensors.device).int()
                        new_det_features = torch.zeros(size=(0, 128), device=self.sample_r.tensors.device).float()

        return new_det_pos, new_det_scores, new_det_features, new_det_cls

    @torch.no_grad()
    def dual_step_reidV3_pre_tracking_vit(self, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # Nested tensor #
        self.sample_r = blob['samples_r']
        self.sample_i = blob['samples_i']

        if self.pre_sample_r is None:
            self.pre_sample_r = self.sample_r
        if self.pre_sample_i is None:
            self.pre_sample_i = self.sample_i
        self.ssim.append(ssim(self.sample_r.tensors.squeeze().permute(1, 2, 0).cpu().numpy(),
                              self.pre_sample_r.tensors.squeeze().permute(1, 2, 0).cpu().numpy(), multichannel=True))
        [ratio, padw, padh] = blob['trans']  # transition
        pos_r, pos_i = self.get_pos()
        mypos_r = pos_r.clone()
        mypos_i = pos_i.clone()

        if (mypos_r.shape[0] > 0):
            # make pre_cts #
            # bboxes to centers
            hm_h, hm_w = self.sample_r.tensors.shape[2], self.sample_r.tensors.shape[3]
            bboxes_r = mypos_r.clone()
            # bboxes

            bboxes_r[:, 0] += bboxes_r[:, 2]
            bboxes_r[:, 1] += bboxes_r[:, 3]
            pre_cts_r = bboxes_r[:, 0:2] / 2.0

            # to input image plane
            pre_cts_r *= ratio
            pre_cts_r[:, 0] += padw
            pre_cts_r[:, 1] += padh
            pre_cts_r[:, 0] = torch.clamp(pre_cts_r[:, 0], 0, hm_w - 1)
            pre_cts_r[:, 1] = torch.clamp(pre_cts_r[:, 1], 0, hm_h - 1)

            # to output image plane
            pre_cts_r /= self.main_args.down_ratio

            no_pre_cts_r = False
        else:
            pre_cts_r = torch.zeros(size=(2, 2), device=mypos_r.device, dtype=mypos_r.dtype)
            no_pre_cts_r = True
            print("No Pre Cts_r!")

        if (mypos_i.shape[0] > 0):
            # make pre_cts #
            # bboxes to centers
            hm_h, hm_w = self.sample_r.tensors.shape[2], self.sample_r.tensors.shape[3]

            bboxes_i = mypos_i.clone()
            # bboxes
            bboxes_i[:, 0] += bboxes_i[:, 2]
            bboxes_i[:, 1] += bboxes_i[:, 3]
            pre_cts_i = bboxes_i[:, 0:2] / 2.0
            # to input image plane
            pre_cts_i *= ratio
            pre_cts_i[:, 0] += padw
            pre_cts_i[:, 1] += padh
            pre_cts_i[:, 0] = torch.clamp(pre_cts_i[:, 0], 0, hm_w - 1)
            pre_cts_i[:, 1] = torch.clamp(pre_cts_i[:, 1], 0, hm_h - 1)
            # to output image plane
            pre_cts_i /= self.main_args.down_ratio
            no_pre_cts_i = False
        else:
            pre_cts_i = torch.zeros(size=(2, 2), device=mypos_r.device, dtype=mypos_r.dtype)
            no_pre_cts_i = True
            print("No Pre Cts_i!")

        no_pre_cts = no_pre_cts_r and no_pre_cts_i

        # w = hm_w /self.main_args.down_ratio
        # h = hm_h / self.main_args.down_ratio
        # pre_ct_sample_r = pre_cts_r.clone()
        # pre_ct_sample_r[:, :, 0].clamp_(min=0, max=w - 1)
        # pre_ct_sample_r[:, :, 1].clamp_(min=0, max=h - 1)
        #
        # pre_ct_sample_r[:, :, 0] /= w
        # pre_ct_sample_r[:, :, 1] /= h
        #
        # pre_ct_sample_i = pre_cts_i.clone()
        # pre_ct_sample_i[:, :, 0].clamp_(min=0, max=w - 1)
        # pre_ct_sample_i[:, :, 1].clamp_(min=0, max=h - 1)
        #
        # pre_ct_sample_i[:, :, 0] /= w
        # pre_ct_sample_i[:, :, 1] /= h

        # pre_cts = pre_cts.squeeze()
        # todo check
        # samples_r: NestedTensor, samples_i, pre_samples_r: NestedTensor, pre_samples_i, pre_cts: Tensor,
        #                 mypos, trans

        outputs, [gather_feat_r, gather_feat_i] = self.obj_detect(samples_r=self.sample_r,
                                                                  pre_samples_r=self.pre_sample_r,
                                                                  samples_i=self.sample_i,
                                                                  pre_samples_i=self.pre_sample_i,
                                                                  pre_cts_r=pre_cts_r.clone().unsqueeze(
                                                                      0),
                                                                  pre_cts_i=pre_cts_i.clone().unsqueeze(
                                                                      0),
                                                                  no_pre_cts=no_pre_cts)
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('tkagg')
        # hm = outputs[1]['hm'].squeeze()
        # for i in range(hm.shape[0]):
        #     plt.figure(i + 1)
        #     plt.imshow(hm[i, :, :].cpu().numpy())
        # plt.show()

        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('tkagg')
        # hm = outputs[1]['hm'].squeeze()
        # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))
        # for i in range(hm.shape[0]):
        #     axes[i%2, i//2].imshow(hm[i, :, :].cpu().numpy())
        #     axes[i%2, i//2].set_title(f'Channel {i+1}.')
        #     # plt.figure(i + 1)
        #     # plt.imshow(hm[i, :, :].cpu().numpy())
        # axes[-1, -1].axis('off')
        # plt.tight_layout()
        #
        # fim_id = 197
        # plt.savefig(f'{video_name}Frame{fim_id}.png')
        # plt.show()

        video_name = blob['video_name']
        fim_id = int(blob['frame_name'][:-4])
        det_pos_r, det_scores_r, pre2cur_cts_r, mypos_r, reid_cts_r, reid_feat_r, det_cls_r = self.out_decode(
            outputs[0],
            pre_cts_r,
            no_pre_cts,
            mypos_r,
            padw, padh,
            ratio)

        det_pos_i, det_scores_i, pre2cur_cts_i, mypos_i, reid_cts_i, reid_feat_i, det_cls_i = self.out_decode(
            outputs[1],
            pre_cts_i,
            no_pre_cts,
            mypos_i,
            padw, padh,
            ratio)

        ###########################################################################################################################################################
        # label_r_txt = os.path.join(
        #     '/home/user/PycharmProjects/MOT_Project/mmtracking/data/MOT_00/labels_with_ids/test',
        #     blob['video_name'], 'img1', blob['frame_name'].replace('.jpg', '.txt'))
        # label_i_txt = os.path.join(
        #     '/home/user/PycharmProjects/MOT_Project/mmtracking/data/MOT_01/labels_with_ids/test',
        #     blob['video_name'], 'img1', blob['frame_name'].replace('.jpg', '.txt'))
        # if not os.path.exists(self.main_args.score_save_r):
        #     os.makedirs(self.main_args.score_save_r)
        #     os.makedirs(self.main_args.score_save_i)
        # if os.path.exists(label_r_txt) and os.path.exists(label_i_txt):
        #     bboxes_r = np.loadtxt(label_r_txt, dtype=np.float32).reshape(-1, 6)
        #     bboxes_i = np.loadtxt(label_i_txt, dtype=np.float32).reshape(-1, 6)
        #     # label_r = torch.ones(bboxes_r.shape[0], dtype=np.int)
        #     # label_i = torch.ones(bboxes_i.shape[0], dtype=np.int)
        #
        #     with open(f'{self.main_args.score_save_r}/{video_name}.txt', "a+") as f:
        #         writer = csv.writer(f, delimiter=',')
        #         # x = det_scores_r[bboxes_r.shape[0] - 1]  # last detection value
        #         # y = det_scores_r[bboxes_r.shape[0]]  # first background value
        #         # z = det_scores_r[bboxes_r.shape[0]:].mean()
        #         # m = det_scores_r.mean()  # score mean value
        #         # v = det_scores_r.std()
        #         writer.writerow([fim_id, bboxes_r.shape[0]] + [x.item() for x in list(det_scores_r.cpu().chunk(300))])
        #         # writer.writerow([fim_id, bboxes_r.shape[0], '%f' % x, '%f' % y, '%f' % z, '%f' % m, '%f' % v])
        #     with open(f'{self.main_args.score_save_i}/{video_name}.txt', "a+") as f:
        #         writer = csv.writer(f, delimiter=',')
        #         # x = det_scores_i[bboxes_i.shape[0] - 1]
        #         # y = det_scores_i[bboxes_i.shape[0]]
        #         # z = det_scores_i[bboxes_i.shape[0]:].mean()
        #         # m = det_scores_i.mean()
        #         # v = det_scores_i.std()
        #         # writer.writerow([fim_id, bboxes_i.shape[0], '%f' % x, '%f' % y, '%f' % z, '%f' % m, '%f' % v])
        #         writer.writerow([fim_id, bboxes_i.shape[0]] + [x.item() for x in list(det_scores_i.cpu().chunk(300))])
        ##########################################################################################################

        ############################################################################################################

        ####################################################
        # plot valid dets
        # import matplotlib.pyplot as plt
        # import matplotlib
        # img_r = blob['img_r'][0, :, :, :].permute(1, 2, 0).cpu().numpy()
        # img_i = blob['img_i'][0, :, :, :].permute(1, 2, 0).cpu().numpy()

        # img_r = blob['img_r'].cpu().numpy()
        # img_i = blob['img_i'].cpu().numpy()
        # # cv2.imshow('img_r', img_r)
        # # matplotlib.use('tkagg')
        # # plt.figure()
        # # plt.imshow(img_r)
        # # plt.show()
        # from mmdet.core.visualization import imshow_det_bboxes
        # # from util.image_mmdet import imshow_det_bboxes
        # from util.image import bbox_ncxcywh_to_xyxy
        # CLASSES = ('ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane')
        # label_r_txt = os.path.join(
        #     '/home/user/PycharmProjects/MOT_Project/mmtracking/data/MOT_00/labels_with_ids/test',
        #     blob['video_name'], 'img1', blob['frame_name'].replace('.jpg', '.txt'))
        #
        # bboxes_r = np.loadtxt(label_r_txt, dtype=np.float32).reshape(-1, 6)
        # label_r = torch.ones(bboxes_r.shape[0], dtype=np.int)
        # gt_bboxes_r = bbox_ncxcywh_to_xyxy(bboxes_r[:, 2:])
        #
        # label_i_txt = os.path.join(
        #     '/home/user/PycharmProjects/MOT_Project/mmtracking/data/MOT_01/labels_with_ids/test',
        #     blob['video_name'], 'img1', blob['frame_name'].replace('.jpg', '.txt'))
        #
        # bboxes_i = np.loadtxt(label_i_txt, dtype=np.float32).reshape(-1, 6)
        # label_i = torch.ones(bboxes_i.shape[0], dtype=np.int)
        # gt_bboxes_i = bbox_ncxcywh_to_xyxy(bboxes_i[:, 2:])

        # if not os.path.exists(f'gt_images/00/{video_name}'):
        #     os.makedirs(f'gt_images/01/{video_name}')
        #     os.makedirs(f'gt_images/00/{video_name}')
        # img_r_gt = imshow_det_bboxes(
        #     img_r,
        #     gt_bboxes_r.numpy(),
        #     label_r,
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'gt_images/00/{video_name}/{fim_id}_gt_r.png')
        # img_r_det = imshow_det_bboxes(
        #     img_r,
        #     det_pos_r[det_scores_r > det_scores_r.mean()][:-2, :].cpu().numpy(),
        #     torch.ones(det_pos_r[det_scores_r > det_scores_r.mean()].shape[0], dtype=np.int),
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'{video_name}/{fim_id}_det_r.png')
        # img_i_gt = imshow_det_bboxes(
        #     img_i,
        #     gt_bboxes_i.numpy(),
        #     label_i,
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'gt_images/01/{video_name}/{fim_id}_gt_i.png')
        # img_i_det = imshow_det_bboxes(
        #     img_i,
        #     det_pos_i[det_scores_i > det_scores_i.mean()][:-2, :].cpu().numpy(),
        #     torch.ones(det_pos_i[det_scores_i > det_scores_i.mean()].shape[0], dtype=np.int),
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'{video_name}/{fim_id}_det_i.png')
        # img_r_gt = imshow_det_bboxes(
        #     img_r ,
        #     gt_bboxes_r.numpy(),
        #     label_r,
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'debug_images/00/{video_name}/{fim_id}_gt_r.png')
        # img_r_det = imshow_det_bboxes(
        #     img_r ,
        #     det_pos_r[det_scores_r > det_scores_r.mean()][:-2, :].cpu().numpy(),
        #     torch.ones(det_pos_r[det_scores_r > det_scores_r.mean()].shape[0], dtype=np.int),
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'debug_images/00/{video_name}/{fim_id}_det_r.png')
        # img_i_gt = imshow_det_bboxes(
        #     img_i ,
        #     gt_bboxes_i.numpy(),
        #     label_i,
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'debug_images/01/{video_name}/{fim_id}_gt_i.png')
        # img_i_det = imshow_det_bboxes(
        #     img_i ,
        #     det_pos_i[det_scores_i > det_scores_i.mean()][:-2, :].cpu().numpy(),
        #     torch.ones(det_pos_i[det_scores_i > det_scores_i.mean()].shape[0], dtype=np.int),
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'debug_images/01/{video_name}/{fim_id}_det_i.png')

        ####################################################

        ##################
        # Predict tracks #
        ##################
        if len(self.tracks_r):
            [det_pos_r, det_scores_r, dets_features_birth_r, det_cls_r] = self.dual_tracks_dets_matching_tracking(
                raw_dets=det_pos_r, raw_scores=det_scores_r, pre2cur_cts=pre2cur_cts_r, pos=mypos_r,
                gather_feat=gather_feat_r, reid_cts=reid_cts_r, raw_det_feats=reid_feat_r, raw_cls=det_cls_r,
                hs_m=outputs[0]['reid_mem'], isV=True)
        else:
            dets_features_birth_r = \
                F.grid_sample(reid_feat_r, reid_cts_r.unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='zeros',
                              align_corners=False)[:, :, 0, :].transpose(1, 2)[0]

        if len(self.tracks_i):
            [det_pos_i, det_scores_i, dets_features_birth_i, det_cls_i] = self.dual_tracks_dets_matching_tracking(
                raw_dets=det_pos_i, raw_scores=det_scores_i, pre2cur_cts=pre2cur_cts_i, pos=mypos_i,
                gather_feat=gather_feat_i, reid_cts=reid_cts_i, raw_det_feats=reid_feat_i, raw_cls=det_cls_i,
                hs_m=outputs[1]['reid_mem'], isV=False)
        else:
            dets_features_birth_i = \
                F.grid_sample(reid_feat_i, reid_cts_i.unsqueeze(0).unsqueeze(0), mode='bilinear',
                              padding_mode='zeros',
                              align_corners=False)[:, :, 0, :].transpose(1, 2)[0]
        ###############################################################################################################

        #####################
        # Create new tracks for r#
        #####################
        # filter birth candidates by scores

        valid_dets_idx_r = det_scores_r >= self.det_thresh
        det_pos_r = det_pos_r[valid_dets_idx_r]
        det_scores_r = det_scores_r[valid_dets_idx_r]
        det_cls_r = det_cls_r[valid_dets_idx_r]
        dets_features_birth_r = dets_features_birth_r[valid_dets_idx_r]

        if self.public_detections:
            # no pub dets => in def detect = no private detection
            # case 1: No pub det, private dets OR
            # case 2: No pub det, no private dets

            if blob['dets'][0].shape[0] == 0:
                det_pos_r = torch.zeros(size=(0, 4), device=self.sample_r.tensors.device).float()
                det_scores_r = torch.zeros(size=(0,), device=self.sample_r.tensors.device).long()
                dets_features_birth_r = torch.zeros(size=(0, 64), device=self.sample_r.tensors.device).float()

            # case 3: Pub det, private dets
            elif det_pos_r.shape[0] > 0:
                _, _, orig_h, orig_w = blob['img'].shape
                pub_dets = blob['dets'][0]
                # using centers
                M = pub_dets.shape[0]

                # # iou of shape [#private, #public]#
                if self.main_args.clip:  # for mot20 clip box
                    iou = bbox_overlaps(det_pos_r, clip_boxes_to_image(pub_dets, (orig_h - 1, orig_w - 1)))
                else:
                    iou = bbox_overlaps(det_pos_r, pub_dets)
                # having overlap ?
                valid_private_det_idx_r = []
                for j in range(M):
                    # print("pub dets")
                    i = iou[:, j].argmax()
                    if iou[i, j] > 0:
                        iou[i, :] = -1
                        valid_private_det_idx_r.append(i.item())
                det_pos_r = det_pos_r[valid_private_det_idx_r]
                det_scores_r = det_scores_r[valid_private_det_idx_r]
                dets_features_birth_r = dets_features_birth_r[valid_private_det_idx_r]
                det_cls_r = det_cls_r[valid_private_det_idx_r]

            # case 4: No pub det, no private dets
            else:
                det_pos_r = torch.zeros(size=(0, 4), device=self.sample_r.tensors.device).float()
                det_scores_r = torch.zeros(size=(0,), device=self.sample_r.tensors.device).long()
                det_cls_r = torch.zeros(size=(0,), device=self.sample_r.tensors.device).int()
                dets_features_birth_r = torch.zeros(size=(0, 64), device=self.sample_r.tensors.device).float()

        else:
            pass

        if det_pos_r.nelement() > 0:

            assert det_pos_r.shape[0] == dets_features_birth_r.shape[0] == det_scores_r.shape[0]
            # try to re-identify tracks
            det_pos_r, det_scores_r, dets_features_birth_r, det_cls_r = self.reid(det_pos_r, det_scores_r,
                                                                                  dets_features_birth_r, det_cls_r,
                                                                                  isV=True)

            assert det_pos_r.shape[0] == dets_features_birth_r.shape[0] == det_scores_r.shape[0] == det_cls_r.shape[0]

            # add new
            if det_pos_r.nelement() > 0:
                self.add(det_pos_r, det_scores_r, dets_features_birth_r, det_cls_r, isV=True)

        #####################
        # Create new tracks for i#
        #####################
        # filter birth candidates by scores
        valid_dets_idx_i = det_scores_i >= self.det_thresh
        det_pos_i = det_pos_i[valid_dets_idx_i]
        det_scores_i = det_scores_i[valid_dets_idx_i]
        det_cls_i = det_cls_i[valid_dets_idx_i]
        dets_features_birth_i = dets_features_birth_i[valid_dets_idx_i]

        if self.public_detections:
            # no pub dets => in def detect = no private detection
            # case 1: No pub det, private dets OR
            # case 2: No pub det, no private dets

            if blob['dets'][1].shape[0] == 0:
                det_pos_i = torch.zeros(size=(0, 4), device=self.sample_r.tensors.device).float()
                det_scores_i = torch.zeros(size=(0,), device=self.sample_r.tensors.device).long()
                dets_features_birth_i = torch.zeros(size=(0, 64), device=self.sample_r.tensors.device).float()

            # case 3: Pub det, private dets
            # todo check the need to modify blob['img']
            elif det_pos_i.shape[0] > 0:
                _, _, orig_h, orig_w = blob['img'].shape
                pub_dets = blob['dets'][1]
                # using centers
                M = pub_dets.shape[0]

                # # iou of shape [#private, #public]#
                if self.main_args.clip:  # for mot20 clip box
                    iou = bbox_overlaps(det_pos_i, clip_boxes_to_image(pub_dets, (orig_h - 1, orig_w - 1)))
                else:
                    iou = bbox_overlaps(det_pos_i, pub_dets)
                # having overlap ?
                valid_private_det_idx_i = []
                for j in range(M):
                    # print("pub dets")
                    i = iou[:, j].argmax()
                    if iou[i, j] > 0:
                        iou[i, :] = -1
                        valid_private_det_idx_i.append(i.item())
                det_pos_i = det_pos_i[valid_private_det_idx_i]
                det_scores_i = det_scores_i[valid_private_det_idx_i]
                dets_features_birth_i = dets_features_birth_i[valid_private_det_idx_i]
                det_cls_i = det_cls_i[valid_private_det_idx_i]

            # case 4: No pub det, no private dets
            else:
                det_pos_i = torch.zeros(size=(0, 4), device=self.sample_r.tensors.device).float()
                det_scores_i = torch.zeros(size=(0,), device=self.sample_r.tensors.device).long()
                det_cls_i = torch.zeros(size=(0,), device=self.sample_r.tensors.device).int()
                dets_features_birth_i = torch.zeros(size=(0, 64), device=self.sample_r.tensors.device).float()

        else:
            pass

        if det_pos_i.nelement() > 0:

            assert det_pos_i.shape[0] == dets_features_birth_i.shape[0] == det_scores_i.shape[0]
            # try to re-identify tracks
            det_pos_i, det_scores_i, dets_features_birth_i, det_cls_i = self.reid(det_pos_i, det_scores_i,
                                                                                  dets_features_birth_i, det_cls_i,
                                                                                  isV=False)

            assert det_pos_i.shape[0] == dets_features_birth_i.shape[0] == det_scores_i.shape[0] == det_cls_i.shape[
                0]

            # add new
            if det_pos_i.nelement() > 0:
                self.add(det_pos_i, det_scores_i, dets_features_birth_i, det_cls_i, isV=False)

        # Cross modal matching
        if len(self.inactive_tracks_r) > 0 and len(self.tracks_i) > 0:
            utrs = []
            for utr in self.inactive_tracks_r:
                utrs.append(utr.pos)

            candidates_i = []
            for t in self.tracks_i:
                if t.birth_active >= 3:
                    candidates_i.append(t.pos)
            utrs = torch.cat(utrs, dim=0)
            if len(candidates_i) > 0:
                candidates_i = torch.cat(candidates_i, dim=0)
                iou_dist_r = box_ops.generalized_box_iou(utrs, candidates_i)
                matches, u_track, u_detection = self.linear_assignment(iou_dist_r.cpu().numpy(),
                                                                       thresh=0.4)

                if matches.shape[0] > 0:
                    # update track dets, scores #
                    for idx_track, idx_det in zip(matches[:, 0], matches[:, 1]):
                        t = self.inactive_tracks_r[idx_track]
                        t.pos = candidates_i[[idx_det]]
                        t.birth_active = t.birth_active + 1
                        t.inactive = 0

                self.new_inactive_tracks_r = []
                for i, t in enumerate(self.inactive_tracks_r):
                    if i in matches[:, 0]:
                        self.tracks_r.append(t)
                    else:
                        t.inactive += 1
                        if t.inactive < 10:
                            self.new_inactive_tracks_r.append(t)
                self.inactive_tracks_r = self.new_inactive_tracks_r

        #################################################################################
        if len(self.inactive_tracks_i) > 0 and len(self.tracks_r) > 0:
            utis = []
            for uti in self.inactive_tracks_i:
                utis.append(uti.pos)
            candidates_r = []
            for t in self.tracks_r:
                if t.birth_active >= 3:
                    candidates_r.append(t.pos)
            utis = torch.cat(utis, dim=0)
            if len(candidates_r) > 0:
                candidates_r = torch.cat(candidates_r, dim=0)
                iou_dist_i = box_ops.generalized_box_iou(utis, candidates_r)
                matches, u_track, u_detection = self.linear_assignment(iou_dist_i.cpu().numpy(),
                                                                       thresh=0.4)

                if matches.shape[0] > 0:
                    # update track dets, scores #
                    for idx_track, idx_det in zip(matches[:, 0], matches[:, 1]):
                        t = self.inactive_tracks_i[idx_track]
                        t.pos = candidates_r[[idx_det]]
                        t.birth_active = t.birth_active + 1
                        t.inactive = 0

                self.new_inactive_tracks_i = []
                for i, t in enumerate(self.inactive_tracks_i):
                    if i in matches[:, 0]:
                        self.tracks_i.append(t)
                    else:
                        t.inactive += 1
                        if t.inactive < 10:
                            self.new_inactive_tracks_i.append(t)
                self.inactive_tracks_i = self.new_inactive_tracks_i

        #######################################################################################

        ####################
        # Generate Results #
        ####################
        online_tlwhs_r = []
        online_ids_r = []
        online_tlwhs_i = []
        online_ids_i = []

        # if pre2cur_cts_r.shape[0] > 0 and fim_id == 8:
        #     # we = 25
        #     # he = 25
        #     # pre2cur_bb_r = self.ct_to_bbox(pre2cur_cts_r, we, he)
        #     # pre2cur_bb_i = self.ct_to_bbox(pre2cur_cts_i, we, he)
        #     color = (0, 255, 0)
        #     thickness = 2
        #     for i in range(pre_cts_r.shape[0]):
        #         cv2.arrowedLine(img_r, pre_cts_r[i].cpu().numpy().astype(np.int16)*self.main_args.down_ratio,
        #                         pre2cur_cts_r[i].cpu().numpy().astype(np.int16), color, thickness)
        #     cv2.imwrite(f'{video_name}/{fim_id}_det_track_r.png', img_r)
        #     for i in range(pre_cts_i.shape[0]):
        #         cv2.arrowedLine(img_i, pre_cts_i[i].cpu().numpy().astype(np.int16)*self.main_args.down_ratio,
        #                         pre2cur_cts_i[i].cpu().numpy().astype(np.int16), color, thickness)
        #     cv2.imwrite(f'{video_name}/{fim_id}_det_track_i.png', img_i)

        # for i in range(pre_cts_r.shape[0]):
        #     cv2.arrowedLine(img_r, (60, 73), (234, 319), color, thickness)
        # cv2.imwrite(f'{video_name}/{fim_id}_det_track_r.png', img_r)

        # img_r_det = imshow_det_bboxes(
        #     img_r,
        #     pre2cur_bb_r.cpu().numpy(),
        #     torch.ones(pre2cur_bb_r.shape[0], dtype=np.int),
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'{video_name}/{fim_id}_det_track_r.png')
        #
        # img_i_det = imshow_det_bboxes(
        #     img_i,
        #     pre2cur_bb_i.cpu().numpy(),
        #     torch.ones(pre2cur_bb_i.shape[0], dtype=np.int),
        #     None,
        #     class_names=CLASSES,
        #     bbox_color=None,
        #     text_color=None,
        #     show=False,
        #     out_file=f'{video_name}/{fim_id}_det_track_i.png')

        for t in self.tracks_r:
            if t.id not in self.results_r.keys():
                self.results_r[t.id] = {}
            if t.birth_active > 3:
                self.results_r[t.id][self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
                t.traj[self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
            elif t.birth_active == 3:

                self.results_r[t.id][self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
                if self.im_index - 1 in t.traj.keys():
                    self.results_r[t.id][self.im_index - 1] = t.traj[self.im_index - 1]
                if self.im_index - 2 in t.traj.keys():
                    self.results_r[t.id][self.im_index - 2] = t.traj[self.im_index - 2]
                t.traj[self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
            else:
                t.traj[self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
            if t.birth_active > 3:
                tlwh = t.bbox_to_tlwh()
                online_tlwhs_r.append(tlwh.squeeze().cpu().numpy())
                online_ids_r.append(t.id)

        for t in self.tracks_i:
            if t.id not in self.results_i.keys():
                self.results_i[t.id] = {}
            if t.birth_active > 3:
                self.results_i[t.id][self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
                t.traj[self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
            elif t.birth_active == 3 and t.inactive:
                self.results_i[t.id][self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
                self.results_i[t.id][self.im_index - 1] = t.traj[self.im_index - 1]
                self.results_i[t.id][self.im_index - 2] = t.traj[self.im_index - 2]
                t.traj[self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
            else:
                t.traj[self.im_index] = np.concatenate(
                    [t.pos[0].cpu().numpy(), np.array([t.score.cpu()]), np.array([t.cls.cpu()])])
            if t.birth_active > 3:
                tlwh = t.bbox_to_tlwh()
                online_tlwhs_i.append(tlwh.squeeze().cpu().numpy())
                online_ids_i.append(t.id)

        new_inactive_tracks = []
        for t in self.inactive_tracks_r:
            t.count_inactive += 1
            if t.birth_active < 3:
                self.results_r.pop(t.id)
            if t.has_positive_area() and t.count_inactive <= self.inactive_patience and t.birth_active >= 3:
                new_inactive_tracks.append(t)
        self.inactive_tracks_r = new_inactive_tracks
        new_inactive_tracks = []
        for t in self.inactive_tracks_i:
            t.count_inactive += 1
            if t.birth_active < 3:
                self.results_i.pop(t.id)
            if t.has_positive_area() and t.count_inactive <= self.inactive_patience and t.birth_active >= 3:
                new_inactive_tracks.append(t)
        self.inactive_tracks_i = new_inactive_tracks

        self.im_index += 1
        self.pre_sample_r = self.sample_r
        self.pre_sample_i = self.sample_i
        return [self.im_index - 1, online_tlwhs_r, online_ids_r], [self.im_index - 1, online_tlwhs_i, online_ids_i]

    def get_results(self):
        return self.results_r, self.results_i

    def out_decode(self, outputs, pre_cts, no_pre_cts, mypos, padw, padh, ratio):
        # # post processing #

        # todo check the output of det model
        output = {k: v[-1] for k, v in outputs.items() if k != 'boxes'}

        # 'hm' is not _sigmoid!
        output['hm'] = torch.clamp(output['hm'].sigmoid(), min=1e-4, max=1 - 1e-4)

        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('tkagg')
        # hm = output['hm'].squeeze()
        # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))
        # for i in range(hm.shape[0]):
        #     axes[i%2, i//2].imshow(hm[i, :, :].cpu().numpy())
        #     axes[i%2, i//2].set_title(f'Channel {i+1}.')
        #     # plt.figure(i + 1)
        #     # plt.imshow(hm[i, :, :].cpu().numpy())
        # axes[-1, -1].axis('off')
        # plt.tight_layout()

        # fim_id = 197
        # plt.savefig(f'Frame{fim_id}.png')
        plt.show()
        decoded = generic_decode(output, K=self.main_args.K, opt=self.main_args,
                                 pre_cts=pre_cts)
        # decoded, pre_cts_scores, pre_cts_scores_track = generic_decode(output, K=self.main_args.K, opt=self.main_args,
        #                                                                pre_cts=pre_cts)

        out_scores = decoded['scores'][0]
        out_cls = decoded['clses'][0].int()
        # labels_out = decoded['clses'][0].int() + 1

        # # reid features #
        # torch.Size([1, 64, 152, 272])

        if no_pre_cts:
            pre2cur_cts = torch.zeros_like(mypos)[..., :2]
        else:
            pre2cur_cts = self.main_args.down_ratio * (decoded['tracking'][0] + pre_cts)
            pre2cur_cts[:, 0] -= padw
            pre2cur_cts[:, 1] -= padh
            pre2cur_cts /= ratio

        # extract reid features #
        boxes = decoded['bboxes'][0].clone()
        reid_cts = torch.stack([0.5 * (boxes[:, 0] + boxes[:, 2]), 0.5 * (boxes[:, 1] + boxes[:, 3])], dim=1)
        reid_cts[:, 0] /= outputs['reid'][0].shape[3]
        reid_cts[:, 1] /= outputs['reid'][0].shape[2]
        reid_cts = torch.clamp(reid_cts, min=0.0, max=1.0)
        reid_cts = (2.0 * reid_cts - 1.0)
        # print(reid_cts.shape)

        out_boxes = decoded['bboxes'][0] * self.main_args.down_ratio
        out_boxes[:, 0::2] -= padw
        out_boxes[:, 1::2] -= padh
        out_boxes /= ratio

        # filtered by scores # do not need filter in multiple classes
        # filtered_idx = labels_out == 1  # todo warning, wrong for multiple classes
        # out_scores = out_scores[filtered_idx]
        # out_boxes = out_boxes[filtered_idx]
        #
        # reid_cts = reid_cts[filtered_idx]
        # post processing #

        return out_boxes, out_scores, pre2cur_cts, mypos, reid_cts, outputs['reid'][0], out_cls

    def ct_to_bbox(self, ct, w, h):
        cts = ct.clone()
        x0 = (cts[:, 0] - w).unsqueeze(1)
        y0 = (cts[:, 1] - h).unsqueeze(1)
        x1 = (cts[:, 0] + w).unsqueeze(1)
        y1 = (cts[:, 1] + h).unsqueeze(1)
        bboxs = torch.cat([x0, y0, x1, y1], dim=1)
        return bboxs


# def draw_arrow(img, prects, cts):


# def compare(blob, det_pos_r, descores_r):
#     import matplotlib
#     img_r = blob['img_r'][0, :, :, :].permute(1, 2, 0).cpu().numpy()
#     # cv2.imshow('img_r', img_r)
#     matplotlib.use('tkagg')
#     # plt.figure()
#     # plt.imshow(img_r)
#     # plt.show()
#     from mmdet.core.visualization import imshow_det_bboxes
#     # from util.image_mmdet import imshow_det_bboxes
#     from util.image import bbox_ncxcywh_to_xyxy
#     CLASSES = ('ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane')
#     label_r_txt = os.path.join(
#         '/home/user/PycharmProjects/MOT_Project/mmtracking/data/MOT_00/labels_with_ids/test',
#         blob['video_name'], 'img1', blob['frame_name'].replace('.jpg', '.txt'))
#
#     bboxes_r = np.loadtxt(label_r_txt, dtype=np.float32).reshape(-1, 6)
#     label_r = torch.ones(bboxes_r.shape[0], dtype=np.int)
#     gt_bboxes_r = bbox_ncxcywh_to_xyxy(bboxes_r[:, 2:])
#     img_r_gt = imshow_det_bboxes(
#         img_r * 256,
#         gt_bboxes_r.numpy(),
#         label_r,
#         None,
#         class_names=CLASSES,
#         bbox_color=None,
#         text_color=None,
#         show=False,
#         out_file='img_r_gt.png')
#     img_r_det = imshow_det_bboxes(
#         img_r * 256,
#         det_pos_r[det_scores_r > 1.0000e-04][:-2, :].cpu().numpy(),
#         torch.ones(det_pos_r[det_scores_r > 1.0000e-04].shape[0] - 2, dtype=np.int),
#         None,
#         class_names=CLASSES,
#         bbox_color=None,
#         text_color=None,
#         show=False,
#         out_file='img_r_det.png')
#

class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, cls, track_id, features, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.cls = cls
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.last_v = torch.Tensor([])
        self.gt_id = None
        self.birth_active = 1
        self.inactive = 0
        self.traj = {}
        self.related_id = None

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            try:
                features = torch.cat(list(self.features), dim=0)
            except:
                print("error")
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())

    def bbox_to_tlwh(self):
        tlwh = self.pos.clone()
        tlwh[:, 2] = self.pos[:, 2] - self.pos[:, 0]
        tlwh[:, 3] = self.pos[:, 3] - self.pos[:, 1]
        return tlwh

# single det model to two det models
# rewrite function step_reidV3_pre_tracking_vit
# rewrite function step_reidV3_pre_tracking_vit
