from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
from .utils import _nms, _topk, _topk_channel
from util.image import gaussian_radius, draw_umich_gaussian
import math
import numpy as np
from util import box_ops
import lap


def linear_assignment(cost_matrix, thresh):
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


def _update_kps_with_hm(
        kps, output, batch, num_joints, K, bboxes=None, scores=None):
    if 'hm_hp' in output:
        hm_hp = output['hm_hp']
        hm_hp = _nms(hm_hp)
        thresh = 0.2
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if 'hp_offset' in output or 'reg' in output:
            hp_offset = output['hp_offset'] if 'hp_offset' in output \
                else output['reg']
            hp_offset = _tranpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        mask = (hm_score < thresh)

        if bboxes is not None:
            l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                   (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
        else:
            l = kps[:, :, :, 0:1].min(dim=1, keepdim=True)[0]
            t = kps[:, :, :, 1:2].min(dim=1, keepdim=True)[0]
            r = kps[:, :, :, 0:1].max(dim=1, keepdim=True)[0]
            b = kps[:, :, :, 1:2].max(dim=1, keepdim=True)[0]
            margin = 0.25
            l = l - (r - l) * margin
            r = r + (r - l) * margin
            t = t - (b - t) * margin
            b = b + (b - t) * margin
            mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                   (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
            # sc = (kps[:, :, :, :].max(dim=1, keepdim=True) - kps[:, :, :, :].min(dim=1))
        # mask = mask + (min_dist > 10)
        mask = (mask > 0).float()
        kps_score = (1 - mask) * hm_score + mask * \
                    scores.unsqueeze(-1).expand(batch, num_joints, K, 1)  # bJK1
        kps_score = scores * kps_score.mean(dim=1).view(batch, K)
        # kps_score[scores < 0.1] = 0
        mask = mask.expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
        return kps, kps_score
    else:
        return kps, kps


def dets_match(raw_dets_r, raw_scores_r, raw_dets_i, raw_scores_i, reid_cts_r, reid_cts_i, reid_feats_r, reid_feats_i):
    # index low-score dets
    diou_dist = box_ops.generalized_box_iou(raw_dets_r, raw_dets_i)
    diou_dist = diou_dist * raw_scores_r[:, None] * raw_scores_i[None, :]
    diou_dist = 1 - diou_dist

    matches_d, u_det_r, u_det_i = linear_assignment(diou_dist.cpu().numpy(),
                                                    thresh=0.3)

    # matches = [[match_row_idx, match_column_idx]...], it gives you all the matches (assignments)
    # unmatched_a gives all the unmatched row indexes
    # unmatched_b gives all the unmatched column indexes

    matches_di = [raw_scores_i(x[1]) for x in matches_d]
    matches_dr = [raw_scores_r(x[0]) for x in matches_d]

    '''
    matches : ir_feature, rgb_feature in C channel;
    unmatches: rgb_feature, rgb_feature in C channel;
    total: match, unmatched_r, unmatched_i
    '''

    dets = torch.cat([raw_dets_r[matches_dr], raw_dets_r[u_det_r], raw_dets_i[u_det_i]], dim=0)
    reid_cts = torch.cat([reid_cts_r[matches_dr], reid_cts_r[u_det_r], reid_cts_i[u_det_i]], dim=0)

    reid_feats = torch.cat([torch.cat([reid_feats_r[matches_dr], reid_feats_i[matches_di]], dim=1),
                            reid_feats_r[u_det_r].repeat(1, 2, 1, 1), reid_feats_i[u_det_i].repeat(1, 2, 1, 1)],
                           dim=0)
    return dets, reid_cts, reid_feats


def generic_decode(output, K=100, opt=None, pre_cts=None, edge=None):
    if not ('hm' in output):
        return {}

    # if not opt.tracking:
    #   output['tracking'] = 0

    heat = output['hm']
    batch, cat, height, width = heat.size()

    heat = _nms(heat)

    scores, inds, clses, ys0, xs0 = _topk(heat, K=K)

    clses = clses.view(batch, K)
    scores = scores.view(batch, K)

    cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
    ret = {'scores': scores, 'clses': clses.float(),
           'xs': xs0, 'ys': ys0, 'cts': cts}

    if 'reg' in output:
        reg = output['reg']
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs0.view(batch, K, 1) + 0.5
        ys = ys0.view(batch, K, 1) + 0.5

    if edge == None:
        edge = [edge]
    else:
        N = edge.shape[2]
        edge = edge.squeeze().reshape(batch, N, height , width)
        edge = _tranpose_and_gather_feat(edge, inds)

    ret['edge'] = edge
    # xyh center_offset #
    if 'center_offset' in output:
        # print("I am here!")
        center_offset = output['center_offset']

        center_offset = _tranpose_and_gather_feat(center_offset, inds)  # B x K x (F)
        # wh = wh.view(batch, K, -1)
        center_offset = center_offset.view(batch, K, 2)

        xs = xs + center_offset[:, :, 0:1]
        ys = ys + center_offset[:, :, 1:2]

    if 'wh' in output:
        wh = output['wh']

        wh = _tranpose_and_gather_feat(wh, inds)  # B x K x (F)
        # wh = wh.view(batch, K, -1)
        wh = wh.view(batch, K, 2)
        wh[wh < 0] = 0
        if wh.size(2) == 2 * cat:  # cat spec
            wh = wh.view(batch, K, -1, 2)
            cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
            wh = wh.gather(2, cats.long()).squeeze(2)  # B x K x 2
        else:
            pass
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        ret['bboxes'] = bboxes

    if 'tracking' in output:
        ret['tracking'] = output['tracking']

    if 'pre_inds' in output and output['pre_inds'] is not None:
        pre_inds = output['pre_inds']  # B x pre_K
        pre_K = pre_inds.shape[1]
        pre_ys = (pre_inds / width).int().float()
        pre_xs = (pre_inds % width).int().float()

        ret['pre_cts'] = torch.cat(
            [pre_xs.unsqueeze(2), pre_ys.unsqueeze(2)], dim=2)
    # pre_cts_scores = []
    # pre_cts_scores_track = []
    # pre_cts_track = (ret['tracking'][0] + pre_cts).int()
    # for i in range(pre_cts.shape[0]):
    #     pre_cts_scores.append(heat[0, :, pre_cts.int()[i, 0], pre_cts.int()[i, 1]].max())
    #     if pre_cts_track[i, 0] > 128 or pre_cts_track[i, 1] > 160:
    #         pre_cts_scores_track.append(0)
    #     else:
    #         pre_cts_scores_track.append(heat[0, :, pre_cts_track[i, 0], pre_cts_track[i, 1]].max())

    return ret  # , pre_cts_scores, pre_cts_scores_track
from mmcv.ops import nms, soft_nms
def generic_decode2(output, K=100, opt=None, pre_cts=None, edge=None):
    if not ('hm' in output):
        return {}

    # if not opt.tracking:
    #   output['tracking'] = 0

    heat = output['hm']
    batch, cat, height, width = heat.size()

    heat = _nms(heat, 1)

    scores, inds, clses, ys0, xs0 = _topk(heat, K=K)

    clses = clses.view(batch, K)
    scores = scores.view(batch, K)

    cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
    ret = {'scores': scores, 'clses': clses.float(),
           'xs': xs0, 'ys': ys0, 'cts': cts}

    if 'reg' in output:
        reg = output['reg']
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs0.view(batch, K, 1) + 0.5
        ys = ys0.view(batch, K, 1) + 0.5

    if edge == None:
        edge = [edge]
    else:
        N = edge.shape[2]
        edge = edge.squeeze().reshape(batch, N, height , width)
        edge = _tranpose_and_gather_feat(edge, inds)

    ret['edge'] = edge
    # xyh center_offset #
    if 'center_offset' in output:
        # print("I am here!")
        center_offset = output['center_offset']

        center_offset = _tranpose_and_gather_feat(center_offset, inds)  # B x K x (F)
        # wh = wh.view(batch, K, -1)
        center_offset = center_offset.view(batch, K, 2)

        xs = xs + center_offset[:, :, 0:1]
        ys = ys + center_offset[:, :, 1:2]

    if 'wh' in output:
        wh = output['wh']

        wh = _tranpose_and_gather_feat(wh, inds)  # B x K x (F)
        # wh = wh.view(batch, K, -1)
        wh = wh.view(batch, K, 2)
        wh[wh < 0] = 0
        if wh.size(2) == 2 * cat:  # cat spec
            wh = wh.view(batch, K, -1, 2)
            cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
            wh = wh.gather(2, cats.long()).squeeze(2)  # B x K x 2
        else:
            pass
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        ret['bboxes'] = bboxes




    dets, inds = nms(bboxes.squeeze(), scores.squeeze(), 0.3)
    for key in ret.keys():
        if key != 'edge':
            if len(ret[key].shape) == 2:
                ret[key] = ret[key][:, inds]
            else:
                ret[key] = ret[key][:, inds, :]



    if 'tracking' in output:
        ret['tracking'] = output['tracking']

    if 'pre_inds' in output and output['pre_inds'] is not None:
        pre_inds = output['pre_inds']  # B x pre_K
        pre_K = pre_inds.shape[1]
        pre_ys = (pre_inds / width).int().float()
        pre_xs = (pre_inds % width).int().float()

        ret['pre_cts'] = torch.cat(
            [pre_xs.unsqueeze(2), pre_ys.unsqueeze(2)], dim=2)




    return ret  # , pre_cts_scores, pre_cts_scores_track


def dual_generic_decode2(outputs, K=100, opt=None):
    assert type(outputs) == list

    ret0 = generic_decode(outputs[0], K, opt)
    ret1 = generic_decode(outputs[1], K, opt)

    ret = {}  # scores, clses, bboxes, tracking, pre_cts

    raw_scores_r = ret0['scores']
    raw_scores_i = ret1['scores']
    raw_dets_r = ret0['bboxes']
    raw_dets_i = ret1['bboxes']
    # dets.shape = [ K, 4]
    diou_dist = box_ops.generalized_box_iou(raw_dets_r.squeeze(), raw_dets_i.squeeze())
    diou_dist = diou_dist * raw_scores_r[:, None] * raw_scores_i[None, :]
    diou_dist = 1 - diou_dist

    matches_d, u_det_r, u_det_i = linear_assignment(diou_dist.cpu().numpy(),
                                                    thresh=0.3)

    # todo consider the clses of the matched dets
    matches_dd = matches_d.copy()
    for x in matches_d:
        if ret0['clses'][x[0]] != ret1['clses'][x[1]]:
            matches_dd.remove(x)
            u_det_r.append(x[0])
            u_det_i.append(x[1])
    matches_di = [raw_scores_i(x[1]) for x in matches_d]
    matches_dr = [raw_scores_r(x[0]) for x in matches_d]
    dets = torch.cat([raw_dets_r[matches_dr], raw_dets_r[u_det_r], raw_dets_i[u_det_i]], dim=0)
    ret['bboxes'] = dets
    scores_match = torch.tensor([max(raw_scores_i(x), raw_scores_r(y)) for x, y in zip(matches_di, matches_dr)])
    scores = torch.cat([scores_match, raw_scores_r[u_det_r], raw_scores_i[u_det_i]], dim=0)
    ret['scores'] = scores
    tracking_match = torch.tensor([(x > y) * ret0['tracking'] + (x <= y) * ret1['tracking'] for x, y in
                                   zip(raw_scores_r[matches_dr], raw_scores_i[matches_di])])
    ret['tracking'] = torch.cat([tracking_match, ret0['tracking'][u_det_r], ret1['tracking'][u_det_i]], dim=0)
    prects_match = torch.tensor([(x > y) * ret0['pre_cts'] + (x <= y) * ret1['pre_cts'] for x, y in
                                 zip(raw_scores_r[matches_dr], raw_scores_i[matches_di])])
    ret['pre_cts'] = torch.cat([prects_match, ret0['pre_cts'][u_det_r], ret1['pre_cts'][u_det_i]], dim=0)
    cts_match = torch.tensor([(x > y) * ret0['cts'] + (x <= y) * ret1['cts'] for x, y in
                              zip(raw_scores_r[matches_dr], raw_scores_i[matches_di])])
    ret['cts'] = torch.cat([cts_match, ret0['cts'][u_det_r], ret1['cts'][u_det_i]], dim=0)

    return ret


def dual_generic_decode(outputs, K=100, opt=None):
    assert type(outputs) == list

    ret0 = generic_decode(outputs[0], K, opt)
    ret1 = generic_decode(outputs[1], K, opt)

    ret = {}  # scores, clses, bboxes, tracking, pre_cts
    ret['bboxes'] = torch.zeros_like(ret0['bboxes'])
    ret['scores'] = torch.zeros_like(ret0['scores'])
    ret['tracking'] = torch.zeros_like(ret0['tracking'])
    if 'pre_cts' in ret0.keys():
        ret['pre_cts'] = torch.zeros_like(ret0['pre_cts'])
    ret['cts'] = torch.zeros_like(ret0['cts'])
    b, k, _ = ret0['bboxes'].shape
    for i in range(b):
        raw_scores_r = ret0['scores'][i]
        raw_scores_i = ret1['scores'][i]
        raw_dets_r = ret0['bboxes'][i]
        raw_dets_i = ret1['bboxes'][i]
        raw_clses_r = ret0['clses'][i]
        raw_clses_i = ret1['clses'][i]
        raw_tracking_r = ret0['tracking'][i]
        raw_tracking_i = ret1['tracking'][i]
        if 'pre_cts' in ret0.keys():
            raw_prects_r = ret0['pre_cts'][i]
            raw_prects_i = ret1['pre_cts'][i]
        raw_cts_r = ret0['cts'][i]
        raw_cts_i = ret1['cts'][i]
        # dets.shape = [ K, 4]
        diou_dist = box_ops.generalized_box_iou(raw_dets_r.squeeze(), raw_dets_i.squeeze())
        diou_dist = diou_dist * raw_scores_r[:, None] * raw_scores_i[None, :]
        diou_dist = 1 - diou_dist

        matches_d, u_det_r, u_det_i = linear_assignment(diou_dist.cpu().numpy(),
                                                        thresh=0.3)

        # todo consider the clses of the matched dets
        matches_dd = matches_d.copy()
        for x in matches_d:
            if raw_clses_r[x[0]] != raw_clses_i[x[1]]:
                matches_dd.remove(x)
                u_det_r.append(x[0])
                u_det_i.append(x[1])
        matches_di = [raw_scores_i(x[1]) for x in matches_d]
        matches_dr = [raw_scores_r(x[0]) for x in matches_d]
        dets = torch.cat([raw_dets_r[matches_dr], raw_dets_r[u_det_r], raw_dets_i[u_det_i]], dim=0)

        scores_match = torch.tensor([max(raw_scores_i(x), raw_scores_r(y)) for x, y in zip(matches_di, matches_dr)])
        scores = torch.cat([scores_match, raw_scores_r[u_det_r], raw_scores_i[u_det_i]], dim=0)
        scores_sort = scores.sort(descending=True)[1]

        tracking_match = torch.tensor([(x > y) * raw_tracking_r + (x <= y) * raw_tracking_i for x, y in
                                       zip(raw_scores_r[matches_dr], raw_scores_i[matches_di])])
        tracking = torch.cat([tracking_match, raw_tracking_r[u_det_r], raw_tracking_i[u_det_i]], dim=0)
        if 'pre_cts' in ret0.keys():
            prects_match = torch.tensor([(x > y) * raw_prects_r + (x <= y) * raw_prects_i for x, y in
                                         zip(raw_scores_r[matches_dr], raw_scores_i[matches_di])])
            pre_cts = torch.cat([prects_match, raw_prects_r[u_det_r], raw_prects_i[u_det_i]], dim=0)
        cts_match = torch.tensor([(x > y) * raw_cts_r + (x <= y) * raw_cts_i for x, y in
                                  zip(raw_scores_r[matches_dr], raw_scores_i[matches_di])])
        cts = torch.cat([cts_match, raw_cts_r[u_det_r], raw_cts_i[u_det_i]], dim=0)
        ret['bboxes'][i] = dets[scores_sort[:K]]
        ret['scores'][i] = scores[scores_sort[:K]]
        ret['tracking'][i] = tracking[scores_sort[:K]]
        if 'pre_cts' in ret0.keys():
            ret['pre_cts'][i] = pre_cts[scores_sort[:K]]
        ret['cts'][i] = cts[scores_sort[:K]]

    return ret
