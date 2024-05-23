"""
Deformable DETR model and criterion classes.
"""

from models.deformable_transformer_lite_dual_track_graph3_gnnloss import build_deforamble_transformer
import torch
from torch import nn
from losses.utils import _sigmoid
from util.misc import NestedTensor
from post_processing.decode import generic_decode
from post_processing.decode import dual_generic_decode
from post_processing.post_process import generic_post_process
import copy
from models.dla import IDAUpV3_bis
from torch import Tensor
from losses.losses_graph_track import FastFocalLoss, RegWeightedL1Loss, loss_boxes, loss_boxes_sdiou, \
    SparseRegWeightedL1Loss
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from util import box_ops
import lap
from torch_geometric.data import HeteroData
from models.models_GAT import GAT
import numpy as np


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class GenericLoss(torch.nn.Module):
    def __init__(self, opt, weight_dict):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        self.sparse_Crit_reg = SparseRegWeightedL1Loss()
        self.opt = opt
        self.weight_dict = weight_dict

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        return output

    def forward(self, outputs, batch, isV=True):
        opt = self.opt
        regression_heads = ['reg', 'wh', 'center_offset']
        losses = {}

        outputs = self._sigmoid_output(outputs)
        if isV:
            for s in range(outputs['hm'].shape[0]):
                if s < outputs['hm'].shape[0] - 1:
                    end_str = f'_{s}'
                else:
                    end_str = ''

                # only 'hm' is use focal loss for heatmap regression. #
                if 'hm' in outputs:
                    losses['hm' + end_str] = self.crit(
                        outputs['hm'][s], batch['hm_r'], batch['ind_r'],
                        batch['mask_r'], batch['cat_r']) / opt.norm_factor

                # sparse tracking #
                if "tracking" in outputs:
                    head = "tracking"
                    losses[head + end_str] = self.sparse_Crit_reg(
                        outputs[head][s], batch[head + '_mask_r'], batch[head + '_r']) / opt.norm_factor

                for head in regression_heads:
                    if head in outputs:
                        # print(head)
                        losses[head + end_str] = self.crit_reg(
                            outputs[head][s], batch[head + '_mask_r'],
                            batch['ind_r'], batch[head + '_r']) / opt.norm_factor

                losses['boxes' + end_str], losses['giou' + end_str] = loss_boxes(outputs['boxes'][s], batch, isV)
                losses['boxes' + end_str] /= opt.norm_factor
                losses['giou' + end_str] /= opt.norm_factor

            return losses
        else:
            for s in range(outputs['hm'].shape[0]):
                if s < outputs['hm'].shape[0] - 1:
                    end_str = f'_{s}'
                else:
                    end_str = ''

                # only 'hm' is use focal loss for heatmap regression. #
                if 'hm' in outputs:
                    losses['hm' + end_str] = self.crit(
                        outputs['hm'][s], batch['hm_i'], batch['ind_i'],
                        batch['mask_i'], batch['cat_i']) / opt.norm_factor

                # sparse tracking #
                if "tracking" in outputs:
                    head = "tracking"
                    losses[head + end_str] = self.sparse_Crit_reg(
                        outputs[head][s], batch[head + '_mask_i'], batch[head + '_i']) / opt.norm_factor

                for head in regression_heads:
                    if head in outputs:
                        # print(head)
                        losses[head + end_str] = self.crit_reg(
                            outputs[head][s], batch[head + '_mask_i'],
                            batch['ind_i'], batch[head + '_i']) / opt.norm_factor

                losses['boxes' + end_str], losses['giou' + end_str] = loss_boxes(outputs['boxes'][s], batch, isV)
                losses['boxes' + end_str] /= opt.norm_factor
                losses['giou' + end_str] /= opt.norm_factor

            return losses


class GenericLoss_sdiou(torch.nn.Module):
    def __init__(self, opt, weight_dict):
        super(GenericLoss_sdiou, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        self.sparse_Crit_reg = SparseRegWeightedL1Loss()
        self.opt = opt
        self.weight_dict = weight_dict

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        return output

    def forward(self, outputs, batch):
        opt = self.opt
        regression_heads = ['reg', 'wh', 'center_offset']
        losses = {}
        if type(outputs) is list:
            outputs = [self._sigmoid_output(output) for output in outputs]
        else:
            outputs = self._sigmoid_output(outputs)

        for s in range(outputs['hm'].shape[0]):
            if s < outputs['hm'].shape[0] - 1:
                end_str = f'_{s}'
            else:
                end_str = ''

            # only 'hm' is use focal loss for heatmap regression. #
            if 'hm' in outputs:
                losses['hm' + end_str] = self.crit(
                    outputs['hm'][s], batch['hm'], batch['ind'],
                    batch['mask'], batch['cat']) / opt.norm_factor

            # sparse tracking #
            if "tracking" in outputs:
                head = "tracking"
                losses[head + end_str] = self.sparse_Crit_reg(
                    outputs[head][s], batch[head + '_mask'], batch[head]) / opt.norm_factor

            for head in regression_heads:
                if head in outputs:
                    # print(head)
                    losses[head + end_str] = self.crit_reg(
                        outputs[head][s], batch[head + '_mask'],
                        batch['ind'], batch[head]) / opt.norm_factor

            losses['boxes' + end_str], losses['giou' + end_str] = loss_boxes_sdiou(outputs['boxes'][s], batch)
            losses['boxes' + end_str] /= opt.norm_factor
            losses['giou' + end_str] /= opt.norm_factor

        return losses


class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, transformer, num_classes, output_shape, args, half=True):
        """ Initializes the model.
        """
        super().__init__()

        self.transformer = transformer
        self.output_shape = output_shape
        self.main_args = args
        # # different ida up for tracking and detection
        # self.ida_up_tracking = IDAUpV3(
        #     64, [256, 256, 256], [])
        self.hidden_dim = self.main_args.hidden_dim
        # todo checkout the necessity of setting different modalities of modules
        self.ida_up = IDAUpV3_bis(
            64, [self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim])

        '''
        (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
        '''
        self.data = HeteroData()
        self.hm = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True),
            SiLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.ct_offset_reg_wh = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=3 // 2, bias=True),
            SiLU(),
            nn.Conv2d(256, 6, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.relu = nn.ReLU()

        # future tracking offset
        self.tracking = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            SiLU(),
            nn.Linear(256, 2)
        )

        # init weights #
        # prior bias
        self.hm[-1].bias.data.fill_(-4.6)
        fill_fc_weights(self.ct_offset_reg_wh)
        fill_fc_weights(self.tracking)
        self.query_embed = None

    def forward(self, samples_r: NestedTensor, samples_i, pre_samples_r: NestedTensor, pre_samples_i, pre_cts_r: Tensor,
                pre_cts_i: Tensor, no_pre_cts, tracking_r, tracking_i):
        assert isinstance(samples_i, NestedTensor)

        merged_hs_r, merged_hs_i, _ = self.transformer(samples_r, pre_samples_r, samples_i, pre_samples_i,
                                                                 pre_cts_r, pre_cts_i, no_pre_cts, tracking_r, tracking_i)

        out_i = self.regress(merged_hs_i)
        out_r = self.regress(merged_hs_r)

        return [out_r, out_i], _

    def regress(self, merged_hs):
        hs = []
        pre_hs = []

        for hs_m, pre_hs_m in merged_hs:
            hs.append(hs_m)
            pre_hs.append(pre_hs_m)

        outputs_hms = []
        outputs_regs = []
        outputs_whs = []
        outputs_ct_offsets = []
        outputs_tracking = []
        outputs_reid = []
        outputs_coords = []

        # for idx, out in enumerate(hs[0]):
        #     print(idx, out.shape)

        for layer_lvl in range(len(hs)):
            # torch.cuda.synchronize()
            # tic = time.time()
            hs[layer_lvl] = self.ida_up(hs[layer_lvl], 0, len(hs[layer_lvl]))[-1]
            # torch.cuda.synchronize()
            # print(f"Runtime for detect: {1 / (time.time() - tic) :.2f} fps.")

            ct_offset, wh_head, reg_head = torch.chunk(self.ct_offset_reg_wh(hs[layer_lvl]), 3, dim=1)
            wh_head = self.relu(wh_head)
            reg_head = self.relu(reg_head)
            hm_head = self.hm(hs[layer_lvl])
            tracking_head = self.tracking(pre_hs[layer_lvl])
            outputs_coords.append(torch.cat([reg_head + ct_offset, wh_head], dim=1))
            outputs_whs.append(wh_head)
            outputs_ct_offsets.append(ct_offset)
            outputs_regs.append(reg_head)
            outputs_hms.append(hm_head)
            outputs_tracking.append(tracking_head)
            outputs_reid.append(hs[layer_lvl])

        out = {'hm': torch.stack(outputs_hms), 'boxes': torch.stack(outputs_coords),
               'wh': torch.stack(outputs_whs), 'reg': torch.stack(outputs_regs),
               'center_offset': torch.stack(outputs_ct_offsets), 'tracking': torch.stack(outputs_tracking),
               'reid': torch.stack(outputs_reid)}

        return out

    def out_decode(self, outputs, pre_cts, padw, padh, ratio):
        # # post processing #

        # todo check the output of det model
        output = {k: v[-1] for k, v in outputs.items() if k != 'boxes'}

        # 'hm' is not _sigmoid!
        output['hm'] = torch.clamp(output['hm'].sigmoid(), min=1e-4, max=1 - 1e-4)

        decoded = generic_decode(output, K=300, opt=None)

        out_scores = decoded['scores'][0]
        out_cls = decoded['clses'][0].int()
        # labels_out = decoded['clses'][0].int() + 1

        # # reid features #
        # torch.Size([1, 64, 152, 272])
        if len(pre_cts.shape) == 3:
            pre_cts = pre_cts[0]
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

        return [out_boxes, out_scores, pre2cur_cts, reid_cts, outputs['reid'][0], out_cls]


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, args, valid_ids):
        self.args = args
        self._valid_ids = valid_ids
        print("valid_ids: ", self._valid_ids)
        super().__init__()

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        return output

    @torch.no_grad()
    def forward(self, outputs, target_sizes, pre_cts, filter_score=True):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # for map you don't need to filter
        if filter_score:
            out_thresh = self.args.pre_thresh
        else:
            out_thresh = 0.0
        # get the output of last layer of transformer

        if type(outputs) is list:
            output0 = {k: v[-1].cpu() for k, v in outputs[0].items() if k != 'boxes'}
            output1 = {k: v[-1].cpu() for k, v in outputs[1].items() if k != 'boxes'}
            outputs = [output0, output1]
            outputs = [self._sigmoid_output(output) for output in outputs]
            dets = dual_generic_decode(outputs, K=self.args.K, opt=self.args)
        else:
            output = {k: v[-1].cpu() for k, v in outputs.items() if k != 'boxes'}
            output = self._sigmoid_output(output)
            dets = generic_decode(output, K=self.args.K, opt=self.args)
        # 'hm' is not _sigmoid!

        dws = []
        dhs = []
        ratios = []
        height, width = self.args.input_h, self.args.input_w

        for target_size in target_sizes:
            shape = target_size.cpu().numpy()  # shape = [height, width]
            ratio = min(float(height) / shape[0], float(width) / shape[1])
            new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
            dw = (width - new_shape[0]) / 2  # width padding
            dh = (height - new_shape[1]) / 2  # height padding

            dws.append(dw)
            dhs.append(dh)
            ratios.append(ratio)
        results, pre_results = generic_post_process(opt=self.args, pre_cts=pre_cts.cpu(), dets=dets, dws=dws, dhs=dhs,
                                                    ratios=ratios, filter_by_scores=out_thresh)
        coco_results = []
        for btch_idx in range(len(results)):
            boxes = []
            scores = []
            labels = []

            for det in results[btch_idx]:
                if det['bbox'][2] - det['bbox'][0] < 1 or det['bbox'][3] - det['bbox'][1] < 1:
                    continue
                boxes.append(det['bbox'].unsqueeze(0))
                scores.append(det['score'])
                labels.append(self._valid_ids[det['class'] - 1])

            if len(boxes) > 0:
                coco_results.append({'scores': torch.as_tensor(scores).float(),
                                     'labels': torch.as_tensor(labels).int(),
                                     'boxes': torch.cat(boxes, dim=0).float(),
                                     # 'boxes': boxes.float(),
                                     'pre2cur_cts': pre_results[btch_idx]['pre2cur_cts']})
            else:
                coco_results.append({'scores': torch.zeros(0).float(),
                                     'labels': torch.zeros(0).int(),
                                     'boxes': torch.zeros(0, 4).float(),
                                     'pre2cur_cts': torch.zeros(0, 2).float()
                                     })
        return coco_results


class PostProcess_o(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, args, valid_ids):
        self.args = args
        self._valid_ids = valid_ids
        print("valid_ids: ", self._valid_ids)
        super().__init__()

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        return output

    @torch.no_grad()
    def forward(self, outputs, target_sizes, pre_cts, filter_score=True):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # for map you don't need to filter
        if filter_score:
            out_thresh = self.args.pre_thresh
        else:
            out_thresh = 0.0
        # get the output of last layer of transformer

        output = {k: v[-1].cpu() for k, v in outputs.items() if k != 'boxes'}

        # 'hm' is not _sigmoid!
        output = self._sigmoid_output(output)

        dets = generic_decode(output, K=self.args.K, opt=self.args)

        dws = []
        dhs = []
        ratios = []
        height, width = self.args.input_h, self.args.input_w

        for target_size in target_sizes:
            shape = target_size.cpu().numpy()  # shape = [height, width]
            ratio = min(float(height) / shape[0], float(width) / shape[1])
            new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
            dw = (width - new_shape[0]) / 2  # width padding
            dh = (height - new_shape[1]) / 2  # height padding

            dws.append(dw)
            dhs.append(dh)
            ratios.append(ratio)
        results, pre_results = generic_post_process(opt=self.args, pre_cts=pre_cts.cpu(), dets=dets, dws=dws, dhs=dhs,
                                                    ratios=ratios, filter_by_scores=out_thresh)
        coco_results = []
        for btch_idx in range(len(results)):
            boxes = []
            scores = []
            labels = []

            for det in results[btch_idx]:
                if det['bbox'][2] - det['bbox'][0] < 1 or det['bbox'][3] - det['bbox'][1] < 1:
                    continue
                boxes.append(det['bbox'])
                scores.append(det['score'])
                labels.append(self._valid_ids[det['class'] - 1])

            if len(boxes) > 0:
                coco_results.append({'scores': torch.as_tensor(scores).float(),
                                     'labels': torch.as_tensor(labels).int(),
                                     'boxes': torch.as_tensor(boxes).float(),
                                     'pre2cur_cts': pre_results[btch_idx]['pre2cur_cts']})
            else:
                coco_results.append({'scores': torch.zeros(0).float(),
                                     'labels': torch.zeros(0).int(),
                                     'boxes': torch.zeros(0, 4).float(),
                                     'pre2cur_cts': torch.zeros(0, 2).float()
                                     })
        return coco_results


def build(args):
    num_classes = args.num_classes

    if args.dataset_file == 'coco':
        valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
    elif args.dataset_file == 'rgbt_tiny':
        valid_ids = [0, 1, 2, 3, 4, 5, 6]

    device = torch.device(args.device)

    transformer = build_deforamble_transformer(args)
    print("num_classes", num_classes)
    model = DeformableDETR(
        transformer,
        num_classes=num_classes,
        output_shape=(args.input_h / args.down_ratio, args.input_w / args.down_ratio),
        args=args,
        half=args.half
    )
    # weights
    weight_dict = {'hm': args.hm_weight, 'reg': args.off_weight, 'wh': args.wh_weight, 'boxes': args.boxes_weight,
                   'giou': args.giou_weight, 'center_offset': args.ct_offset_weight, 'tracking': args.tracking_weight,
                   'gnn': args.gnn_weight}

    # if args.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)

    criterion = GenericLoss(args, weight_dict).to(device)
    postprocessors = {'bbox': PostProcess(args, valid_ids)}
    return model, criterion, postprocessors
