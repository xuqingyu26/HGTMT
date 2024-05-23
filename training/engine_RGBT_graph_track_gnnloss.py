"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
import copy


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, adaptive_clip: bool = False,
                    scaler: torch.nn.Module = None, half: bool = True, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    data_type = torch.float16 if half else torch.float32

    for ret in metric_logger.log_every(data_loader, print_freq, header):

        samples_r = utils.NestedTensor(ret['image_r'], ret['pad_mask'])
        samples_r = samples_r.to(device)
        pre_samples_r = utils.NestedTensor(ret['pre_img_r'], ret['pre_pad_mask'])
        pre_samples_r = pre_samples_r.to(device)

        samples_i = utils.NestedTensor(ret['image_i'], ret['pad_mask'])
        samples_i.tensors = samples_i.tensors.to(data_type)
        samples_i = samples_i.to(device)
        pre_samples_i = utils.NestedTensor(ret['pre_img_i'], ret['pre_pad_mask'])
        pre_samples_i.tensors = pre_samples_i.tensors.to(data_type)
        pre_samples_i = pre_samples_i.to(device)
        if args.dataset_file == 'rgbt_tiny':
            targets = {k: v.to(device) for k, v in ret.items() if
                       k != 'orig_image' and k != 'image_r' and k != 'image_i' and 'pad_mask' not in k and 'pre_img' not in k and k != 'image_id_rgbt'}
        else:

            targets = {k: v.to(device) for k, v in ret.items() if
                       k != 'orig_image' and k != 'image' and 'pad_mask' not in k and 'pre_img' not in k}

        if args.dataset_file == 'rgbt_tiny' and args.datatype == 'infrared':
            targets['image_id'] = ret['image_id_rgbt'][1]
        else:
            targets['image_id'] = ret['image_id_rgbt'][0]

        # save memory, reduce max_dets#

        max_dets = torch.cat([targets["valid_num_pre_dets_r"], targets["valid_num_pre_dets_i"]]).max()
        max_dets = int(max_dets)
        if max_dets == 0:
            max_dets = 5

        targets['pre_cts_r'] = targets['pre_cts_r'][:, :max_dets, :]
        targets['tracking_r'] = targets['tracking_r'][:, :max_dets, :]
        targets['tracking_mask_r'] = targets['tracking_mask_r'][:, :max_dets, :]

        targets['pre_cts_i'] = targets['pre_cts_i'][:, :max_dets, :]
        targets['tracking_i'] = targets['tracking_i'][:, :max_dets, :]
        targets['tracking_mask_i'] = targets['tracking_mask_i'][:, :max_dets, :]

        with torch.cuda.amp.autocast(enabled=half):
            # samples_r: NestedTensor, samples_i, pre_samples_r: NestedTensor, pre_samples_i, pre_cts_r: Tensor,
            #                 pre_cts_i: Tensor, no_pre_cts
            if len(targets['pre_cts_r']) == 0 and len(targets['pre_cts_i']) == 0:
                no_pre_cts = True
            else:
                no_pre_cts = False

            outputs, [edge_r, edge_i] = model(samples_r, samples_i, pre_samples_r, pre_samples_i, targets['pre_cts_r'],
                                              targets['pre_cts_i'], no_pre_cts, targets['tracking_r'],
                                              targets['tracking_i'])

            loss_dict = criterion(outputs[0], targets)
            loss_dict2 = criterion(outputs[1], targets)

            for k in loss_dict.keys():
                loss_dict[k] += loss_dict2[k]

            eye_edge = torch.eye(max_dets).reshape(1, 1, max_dets, max_dets).expand(edge_r.shape[0], -1, -1, -1).to(
                edge_r.device)

            mask_r = torch.zeros_like(edge_r)
            mask_i = torch.zeros_like(edge_i)
            for i in range(edge_r.shape[0]):
                mask_r[i, 0, :targets["valid_num_pre_dets_r"][i], :targets["valid_num_pre_dets_r"][i]] = 1
                mask_i[i, 0, :targets["valid_num_pre_dets_i"][i], :targets["valid_num_pre_dets_i"][i]] = 1

            loss_bce = torch.nn.functional.binary_cross_entropy(edge_r * mask_r,
                                                                eye_edge * mask_r) + torch.nn.functional.binary_cross_entropy(
                edge_i * mask_i, eye_edge * mask_i)
            loss_ce = torch.nn.functional.cross_entropy(edge_r * mask_r, eye_edge * mask_r) + torch.nn.functional.cross_entropy(
                edge_i * mask_i, eye_edge * mask_i)
            loss_dict['gnn'] = loss_bce + loss_ce

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        assert len(weight_dict.keys()) == len(loss_dict_reduced.keys())

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        for param in model.parameters():
            param.grad = None
        scaler.scale(losses).backward()

        if adaptive_clip:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                utils.clip_grad_norm(model.parameters())
                grad_total_norm = utils.get_total_grad_norm(model.parameters())
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters())
        else:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds_r, base_ds_i, device, output_dir, half=True):
    model.eval()  # no back gradients
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    data_type = torch.float16 if half else torch.float32

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator_r = CocoEvaluator(base_ds_r, iou_types)
    coco_evaluator_i = CocoEvaluator(base_ds_i, iou_types)
    # set max Dets to 300
    coco_evaluator_r.coco_eval[iou_types[0]].params.maxDets = [300, 300, 300]
    coco_evaluator_i.coco_eval[iou_types[0]].params.maxDets = [300, 300, 300]

    for ret in metric_logger.log_every(data_loader, 50, header):
        samples_r = utils.NestedTensor(ret['image_r'], ret['pad_mask'])
        samples_r = samples_r.to(device)
        pre_samples_r = utils.NestedTensor(ret['pre_img_r'], ret['pre_pad_mask'])
        pre_samples_r = pre_samples_r.to(device)

        samples_i = utils.NestedTensor(ret['image_i'], ret['pad_mask'])
        samples_i.tensors = samples_i.tensors.to(data_type)
        samples_i = samples_i.to(device)
        pre_samples_i = utils.NestedTensor(ret['pre_img_i'], ret['pre_pad_mask'])
        pre_samples_i.tensors = pre_samples_i.tensors.to(data_type)
        pre_samples_i = pre_samples_i.to(device)

        targets = {k: v.to(device) for k, v in ret.items() if
                   k != 'orig_image' and k != 'image_r' and k != 'image_i' and 'pad_mask' not in k and 'pre_img' not in k and k != 'image_id_rgbt'}
        targets['image_id'] = ret['image_id_rgbt'][1]

        # max_dets, _ = torch.max(targets['tracking_mask'][:,:,0].sum(-1), dim=0)
        max_dets = torch.cat([targets["valid_num_pre_dets_r"], targets["valid_num_pre_dets_i"]]).max()
        max_dets = int(max_dets)
        if max_dets == 0:
            max_dets = 5

        targets['pre_cts_r'] = targets['pre_cts_r'][:, :max_dets, :]
        targets['tracking_r'] = targets['tracking_r'][:, :max_dets, :]
        targets['tracking_mask_r'] = targets['tracking_mask_r'][:, :max_dets, :]

        targets['pre_cts_i'] = targets['pre_cts_i'][:, :max_dets, :]
        targets['tracking_i'] = targets['tracking_i'][:, :max_dets, :]
        targets['tracking_mask_i'] = targets['tracking_mask_i'][:, :max_dets, :]

        with torch.cuda.amp.autocast(enabled=half):
            if len(targets['pre_cts_r']) == 0 and len(targets['pre_cts_i']) == 0:
                no_pre_cts = True
            else:
                no_pre_cts = False

            outputs, [edge_r, edge_i] = model(samples_r, samples_i, pre_samples_r, pre_samples_i, targets['pre_cts_r'],
                                              targets['pre_cts_i'], no_pre_cts, targets['tracking_r'],
                                              targets['tracking_i'])

            loss_dict = criterion(outputs[0], targets)
            loss_dict2 = criterion(outputs[1], targets)

            for k in loss_dict.keys():
                loss_dict[k] += loss_dict2[k]

            eye_edge = torch.eye(max_dets).reshape(1, 1, max_dets, max_dets).expand(edge_r.shape[0], -1, -1, -1).to(
                edge_r.device)

            mask_r = torch.zeros_like(edge_r)
            mask_i = torch.zeros_like(edge_i)
            for i in range(edge_r.shape[0]):
                mask_r[i, 0, :targets["valid_num_pre_dets_r"][i], :targets["valid_num_pre_dets_r"][i]] = 1
                mask_i[i, 0, :targets["valid_num_pre_dets_i"][i], :targets["valid_num_pre_dets_i"][i]] = 1

            loss_bce = torch.nn.functional.binary_cross_entropy(edge_r * mask_r,
                                                                eye_edge * mask_r) + torch.nn.functional.binary_cross_entropy(
                edge_i * mask_i, eye_edge * mask_i)
            loss_ce = torch.nn.functional.cross_entropy(edge_r * mask_r,
                                                        eye_edge * mask_r) + torch.nn.functional.cross_entropy(
                edge_i * mask_i, eye_edge * mask_i)
            loss_dict['gnn'] = loss_bce + loss_ce

        weight_dict = criterion.weight_dict
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        results_r = postprocessors['bbox'](outputs[0], targets['orig_size'], pre_cts=targets['pre_cts_r'],
                                         filter_score=False)
        results_i = postprocessors['bbox'](outputs[1], targets['orig_size'], pre_cts=targets['pre_cts_i'],
                                           filter_score=False)

        res_r = {img_id.item(): output for img_id, output in zip(targets['image_id_rgbt'][0], results_r)}
        res_i = {img_id.item(): output for img_id, output in zip(targets['image_id_rgbt'][1], results_i)}
        if coco_evaluator_r is not None:
            coco_evaluator_r.update(res_r)
        if coco_evaluator_i is not None:
            coco_evaluator_i.update(res_i)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator_r is not None:
        coco_evaluator_r.synchronize_between_processes()
    if coco_evaluator_i is not None:
        coco_evaluator_i.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator_r is not None:
        coco_evaluator_r.accumulate()
        coco_evaluator_r.summarize()
    if coco_evaluator_i is not None:
        coco_evaluator_i.accumulate()
        coco_evaluator_i.summarize()
    stats_r = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats_i = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator_r is not None:
        if 'bbox' in postprocessors.keys():
            stats_r['coco_eval_bbox'] = coco_evaluator_r.coco_eval['bbox'].stats.tolist()
        if 'bbox' in postprocessors.keys():
            stats_i['coco_eval_bbox'] = coco_evaluator_i.coco_eval['bbox'].stats.tolist()
    return stats_r, coco_evaluator_r, stats_i, coco_evaluator_i


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, half=True, isV=True):
    model.eval()  # no back gradients
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    data_type = torch.float16 if half else torch.float32

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # set max Dets to 300
    coco_evaluator.coco_eval[iou_types[0]].params.maxDets = [300, 300, 300]

    for ret in metric_logger.log_every(data_loader, 50, header):
        samples_r = utils.NestedTensor(ret['image_r'], ret['pad_mask'])
        samples_r = samples_r.to(device)
        pre_samples_r = utils.NestedTensor(ret['pre_img_r'], ret['pre_pad_mask'])
        pre_samples_r = pre_samples_r.to(device)

        samples_i = utils.NestedTensor(ret['image_i'], ret['pad_mask'])
        samples_i.tensors = samples_i.tensors.to(data_type)
        samples_i = samples_i.to(device)
        pre_samples_i = utils.NestedTensor(ret['pre_img_i'], ret['pre_pad_mask'])
        pre_samples_i.tensors = pre_samples_i.tensors.to(data_type)
        pre_samples_i = pre_samples_i.to(device)

        targets = {k: v.to(device) for k, v in ret.items() if
                   k != 'orig_image' and k != 'image_r' and k != 'image_i' and 'pad_mask' not in k and 'pre_img' not in k and k != 'image_id_rgbt'}
        if isV:
            targets['image_id'] = ret['image_id_rgbt'][0]
        else:
            targets['image_id'] = ret['image_id_rgbt'][1]

        # max_dets, _ = torch.max(targets['tracking_mask'][:,:,0].sum(-1), dim=0)
        max_dets = torch.cat([targets["valid_num_pre_dets_r"], targets["valid_num_pre_dets_i"]]).max()
        max_dets = int(max_dets)
        if max_dets == 0:
            max_dets = 5

        targets['pre_cts_r'] = targets['pre_cts_r'][:, :max_dets, :]
        targets['tracking_r'] = targets['tracking_r'][:, :max_dets, :]
        targets['tracking_mask_r'] = targets['tracking_mask_r'][:, :max_dets, :]

        targets['pre_cts_i'] = targets['pre_cts_i'][:, :max_dets, :]
        targets['tracking_i'] = targets['tracking_i'][:, :max_dets, :]
        targets['tracking_mask_i'] = targets['tracking_mask_i'][:, :max_dets, :]


        with torch.cuda.amp.autocast(enabled=half):
            if len(targets['pre_cts_r']) == 0 and len(targets['pre_cts_i']) == 0:
                no_pre_cts = True
            else:
                no_pre_cts = False

            outputs, [edge_r, edge_i] = model(samples_r, samples_i, pre_samples_r, pre_samples_i, targets['pre_cts_r'],
                                              targets['pre_cts_i'], no_pre_cts, targets['tracking_r'],
                                              targets['tracking_i'])

            loss_dict = criterion(outputs[0], targets)
            loss_dict2 = criterion(outputs[1], targets)

            for k in loss_dict.keys():
                loss_dict[k] += loss_dict2[k]

            eye_edge = torch.eye(max_dets).reshape(1, 1, max_dets, max_dets).expand(edge_r.shape[0], -1, -1, -1).to(
                edge_r.device)

            mask_r = torch.zeros_like(edge_r)
            mask_i = torch.zeros_like(edge_i)
            for i in range(edge_r.shape[0]):
                mask_r[i, 0, :targets["valid_num_pre_dets_r"][i], :targets["valid_num_pre_dets_r"][i]] = 1
                mask_i[i, 0, :targets["valid_num_pre_dets_i"][i], :targets["valid_num_pre_dets_i"][i]] = 1

            loss_bce = torch.nn.functional.binary_cross_entropy(edge_r * mask_r,
                                                                eye_edge * mask_r) + torch.nn.functional.binary_cross_entropy(
                edge_i * mask_i, eye_edge * mask_i)
            loss_ce = torch.nn.functional.cross_entropy(edge_r * mask_r,
                                                        eye_edge * mask_r) + torch.nn.functional.cross_entropy(
                edge_i * mask_i, eye_edge * mask_i)
            loss_dict['gnn'] = loss_bce + loss_ce
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if isV:
            output = outputs[0]
            pre_cts = targets['pre_cts_r']
        else:
            output = outputs[1]
            pre_cts = targets['pre_cts_i']
        results = postprocessors['bbox'](output, targets['orig_size'], pre_cts=pre_cts,
                                         filter_score=False)

        res = {img_id.item(): output for img_id, output in zip(targets['image_id'], results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator


def train_one_epoch_rgbt(model: torch.nn.Module, criterion: torch.nn.Module,
                         data_loader: Iterable, optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, max_norm: float = 0, adaptive_clip: bool = False,
                         scaler: torch.nn.Module = None, half: bool = True, dataset=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    data_type = torch.float16 if half else torch.float32

    for ret in metric_logger.log_every(data_loader, print_freq, header):

        samples = utils.NestedTensor(ret['image'], ret['pad_mask'])
        # samples.tensors = samples.tensors.to(data_type)
        samples = samples.to(device)
        pre_samples = utils.NestedTensor(ret['pre_img'], ret['pre_pad_mask'])
        # pre_samples.tensors = pre_samples.tensors.to(data_type)
        pre_samples = pre_samples.to(device)
        if dataset == 'rgbt_tiny':
            targets = {k: v.to(device) for k, v in ret.items() if
                       k != 'orig_image' and k != 'image_r' and k != 'image_i' and 'pad_mask' not in k and 'pre_img' not in k}
        else:

            targets = {k: v.to(device) for k, v in ret.items() if
                       k != 'orig_image' and k != 'image' and 'pad_mask' not in k and 'pre_img' not in k}

        # save memory, reduce max_dets#
        max_dets, _ = torch.max(targets["valid_num_pre_dets"], dim=0)
        max_dets = int(max_dets)
        if max_dets == 0:
            max_dets = 5

        targets['pre_cts'] = targets['pre_cts'][:, :max_dets, :]
        targets['tracking'] = targets['tracking'][:, :max_dets, :]
        targets['tracking_mask'] = targets['tracking_mask'][:, :max_dets, :]
        with torch.cuda.amp.autocast(enabled=half):
            outputs = model(samples, pre_samples=pre_samples, pre_hm=targets['pre_cts'])
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        assert len(weight_dict.keys()) == len(loss_dict_reduced.keys())

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        for param in model.parameters():
            param.grad = None
        scaler.scale(losses).backward()

        if adaptive_clip:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                utils.clip_grad_norm(model.parameters())
                grad_total_norm = utils.get_total_grad_norm(model.parameters())
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters())
        else:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
