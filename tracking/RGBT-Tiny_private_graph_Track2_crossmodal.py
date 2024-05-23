import sys
import os
import os.path as osp
import csv
import motmetrics as mm
# dirty insert path #
cur_path = os.path.realpath(__file__)
cur_dir = "/".join(cur_path.split('/')[:-2])
sys.path.insert(0, cur_dir)
os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
import yaml
from datasets.rgbt_dataset_test import GenericDataset_val
from util import visualization as vis
import cv2
from util.timer import Timer
from util.utils import mkdir_if_missing
from util.evaluation import Evaluator

import shutil
from tracking.tracker_rgbt_graph_track_graph_crossmodal2 import Tracker
from tracking.deformable_detr_graph_track_crossmodal import build as build_model

# xyh #
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detemain_RGBT-Tiny_graph.pyctor', add_help=False)
    parser.add_argument('--ignoreIsCrowd', action='store_true')
    parser.add_argument('--pretrained', type=str,
                        default="/home/user/PycharmProjects/MOT_Project/TransCenter/model_zoo/pvtv2_backbone/pvt_v2_b2.pth",
                        help="pretrained")

    # * Transformer
    parser.add_argument('--d_model', default=[64, 128, 320, 512], type=int, nargs='+',
                        help="model dimensions in the transformer")

    parser.add_argument('--nheads', default=[1, 2, 5, 8], type=int, nargs='+',
                        help="Number of attention heads inside the transformer's attentions")

    parser.add_argument('--num_encoder_layers', default=[3, 4, 6, 3], type=int, nargs='+',
                        help="Number of encoding layers in the transformer")

    parser.add_argument('--num_decoder_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")

    parser.add_argument('--dim_feedforward_ratio', default=[8, 8, 4, 4], type=int, nargs='+',
                        help="Intermediate size of the feedforward layers dim ratio in the transformer blocks")

    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")

    parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')

    parser.add_argument('--dec_n_points', default=9, type=int)

    parser.add_argument('--enc_n_points', default=[8, 8, 8, 8], type=int, nargs='+')

    parser.add_argument('--down_sample_ratio', default=[8, 4, 2, 1], type=int, nargs='+')

    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--linear', action='store_true',
                        help='linear vit')

    parser.add_argument('--heads', default=['hm', 'reg', 'wh', 'center_offset', 'tracking'], type=str, nargs='+')

    # * Loss coefficients
    parser.add_argument('--hm_weight', default=1, type=float)
    parser.add_argument('--off_weight', default=1, type=float)
    parser.add_argument('--wh_weight', default=0.1, type=float)
    parser.add_argument('--ct_offset_weight', default=0.1, type=float)
    parser.add_argument('--boxes_weight', default=0.5, type=float)
    parser.add_argument('--giou_weight', default=0.4, type=float)
    parser.add_argument('--norm_factor', default=1.0, type=float)
    parser.add_argument('--tracking_weight', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='rgbt_tiny')
    parser.add_argument('--datatype', default='rgb')
    parser.add_argument('--data_dir', default='/home/user/PycharmProjects/data/MOT/RGBT-Tiny', type=str)
    parser.add_argument('--data_dir_ch', default='', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--gnn_layer_num', default=1, type=int,
                        help="Layer numbers of gnn")
    parser.add_argument('--exp_name', default='1gnn_edgeloss_birth0_2_id',
                        help='Tracke Name')
    parser.add_argument('--output_file',
                        default='/home/user/PycharmProjects/MOT_Project/[RGBT-HGT]/TransCenter_rgbt_graph/tracking/test_models/rgbt/',
                        help='path where to save, empty for no saving')  # det0,track0.04
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--half', default=False, action='store_true', help='half precision')

    # centers
    parser.add_argument('--num_classes', default=7, type=int)
    parser.add_argument('--input_h', default=512, type=int)
    parser.add_argument('--input_w', default=640, type=int)
    parser.add_argument('--down_ratio', default=4, type=int)
    parser.add_argument('--dense_reg', type=int, default=1, help='')

    parser.add_argument('--K', type=int, default=300,
                        help='max number of output objects.')

    parser.add_argument('--debug', action='store_true')

    # noise
    parser.add_argument('--not_rand_crop', action='store_true',
                        help='not use the random crop data augmentation'
                             'from CornerNet.')
    parser.add_argument('--not_max_crop', action='store_true',
                        help='used when the training dataset has'
                             'inbalanced aspect ratios.')
    parser.add_argument('--shift', type=float, default=0.05,
                        help='when not using random crop'
                             'apply shift augmentation.')
    parser.add_argument('--scale', type=float, default=0.05,
                        help='when not using random crop'
                             'apply scale augmentation.')
    parser.add_argument('--rotate', type=float, default=0,
                        help='when not using random crop'
                             'apply rotation augmentation.')
    parser.add_argument('--flip', type=float, default=0.5,
                        help='probability of applying flip augmentation.')
    parser.add_argument('--no_color_aug', action='store_true',
                        help='not use the color augmenation '
                             'from CornerNet')
    parser.add_argument('--aug_rot', type=float, default=0,
                        help='probability of applying '
                             'rotation augmentation.')

    # tracking
    parser.add_argument('--max_frame_dist', type=int, default=3)
    parser.add_argument('--merge_mode', type=int, default=1)
    parser.add_argument('--tracking', default=True, action='store_true')
    parser.add_argument('--pre_hm', action='store_true')
    parser.add_argument('--same_aug_pre', action='store_true')
    parser.add_argument('--zero_pre_hm', action='store_true')
    parser.add_argument('--hm_disturb', type=float, default=0.05)
    parser.add_argument('--lost_disturb', type=float, default=0.4)
    parser.add_argument('--fp_disturb', type=float, default=0.1)
    parser.add_argument('--pre_thresh', type=float, default=-1)
    parser.add_argument('--track_thresh', type=float, default=0.3)
    parser.add_argument('--new_thresh', type=float, default=0.3)
    parser.add_argument('--ltrb_amodal', action='store_true')
    parser.add_argument('--ltrb_amodal_weight', type=float, default=0.1)
    parser.add_argument('--public_det', action='store_true')
    parser.add_argument('--no_pre_img', action='store_true')
    parser.add_argument('--zero_tracking', action='store_true')
    parser.add_argument('--hungarian', action='store_true')
    parser.add_argument('--max_age', type=int, default=-1)
    parser.add_argument('--out_thresh', type=float, default=-1,
                        help='')
    parser.add_argument('--image_blur_aug', action='store_true',
                        help='blur image for aug.')
    parser.add_argument('--adaptive_clip', action='store_true',
                        help='adaptive_clip')

    parser.add_argument('--small', action='store_true',
                        help='smaller dataset')

    parser.add_argument('--recover', action='store_true',
                        help='recovery optimizer.')
    parser.add_argument('--mode', default='duel vit', type=str)
    return parser


curr_pth = '/'.join(osp.dirname(__file__).split('/'))


def write_results(all_tracks, out_dir, seq_name=None, frame_offset=0):
    output_dir = out_dir + "/txt/"
    """Write the tracks in the format for MOT16/MOT17 sumbission

    all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

    Each file contains these lines:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

    assert seq_name is not None, "[!] No seq_name, probably using combined database"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = osp.join(output_dir, seq_name + '.txt')

    with open(file, "w") as of:
        writer = csv.writer(of, delimiter=',')
        for i, track in all_tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                score = bb[4]
                cls = bb[5]
                writer.writerow(
                    [frame + frame_offset, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, cls, 1, 1])


def write_results_pedestrian(all_tracks, out_dir, seq_name=None, frame_offset=0):
    output_dir = out_dir + "/txt/"
    """Write the tracks in the format for MOT16/MOT17 sumbission

    all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

    Each file contains these lines:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

    assert seq_name is not None, "[!] No seq_name, probably using combined database"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = osp.join(output_dir, seq_name + '.txt')

    with open(file, "w") as of:
        writer = csv.writer(of, delimiter=',')
        for i, track in all_tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                score = bb[4]
                cls = 3
                writer.writerow(
                    [frame + frame_offset, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, cls, 1, 1])


def main(tracktor):


    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    np.random.seed(12345)
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    # load model
    main_args = get_args_parser().parse_args()
    main_args.pre_hm = True
    main_args.tracking = True
    main_args.noprehm = True
    device = torch.device(main_args.device)

    ds = GenericDataset_val(root=main_args.data_dir, valset='test', select_seq='')

    ds.default_resolution[0], ds.default_resolution[1] = main_args.input_h, main_args.input_w
    print(main_args.input_h, main_args.input_w)
    main_args.output_h = main_args.input_h // main_args.down_ratio
    main_args.output_w = main_args.input_w // main_args.down_ratio
    main_args.input_res = max(main_args.input_h, main_args.input_w)
    main_args.output_res = max(main_args.output_h, main_args.output_w)
    # threshold
    main_args.det_thresh = 0.4
    main_args.track_thresh_high = 0.5
    main_args.track_thresh_low = 0.2
    main_args.match_thresh = 0.9
    main_args.birth_active = 3
    main_args.clip = True
    main_args.fuse_scores = False
    main_args.fuse_gnn_sim = False
    main_args.iou_recover = True
    main_args.score_save_r = '/home/user/PycharmProjects/MOT_Project/[RGBT-HGT]/TransCenter_rgbt_graph/tracking/det_thresh_new_r'
    main_args.score_save_i = '/home/user/PycharmProjects/MOT_Project/[RGBT-HGT]/TransCenter_rgbt_graph/tracking/det_thresh_new_i'

    model, criterion, postprocessors = build_model(main_args)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)
    # load flowNet
    liteFlowNet = None

    # tracker
    tracker = Tracker(model, None, liteFlowNet, tracktor['tracker'], postprocessor=postprocessors['bbox'],
                      main_args=main_args)
    #############################################

    tracker.public_detections = False
    tracker.mode = main_args.mode

    # dataloader
    def collate_fn(batch):
        batch = list(zip(*batch))
        return tuple(batch)

    data_loader = DataLoader(ds, 1, shuffle=False, drop_last=False, num_workers=8,
                             pin_memory=True, collate_fn=collate_fn)

    # todo check new ckpt
    models = [
        "/home/user/PycharmProjects/MOT_Project/[RGBT-HGT]/TransCenter_rgbt_graph/training/worksdir/rgbt/det_graph_track2_1gnnlayer_gnnloss/log_result/checkpoint.pth",
    ]
    output_dirs = [
        main_args.output_file + main_args.exp_name,
    ]

    for model_dir, output_dir in zip(models, output_dirs):
        if not os.path.exists(output_dir + '/00'):
            os.makedirs(output_dir + '/00')
            os.makedirs(output_dir + '/01')
        # load pretrained #
        tracktor['obj_detect_model'] = model_dir
        tracktor['output_dir_r'] = output_dir + '/00'
        tracktor['output_dir_i'] = output_dir + '/01'
        print("Loading: ", tracktor['obj_detect_model'])
        model.load_state_dict(torch.load(tracktor['obj_detect_model'], map_location=torch.device(device))["model"])
        #############################
        # pathpth = '/home/user/PycharmProjects/MOT_Project/[RGBT-HGT]/TransCenter_rgbt_graph/training/worksdir/rgbt/det_graph_track2/checkpoint.pth'
        # save_model = torch.load(pathpth)
        # print(save_model.keys())
        # transformermodel = save_model['model']
        # gnn_dict = tracker.gnn.state_dict()
        # pro_dict = tracker.input_proj.state_dict()
        # # state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        # gnn_state_dict = {k: transformermodel['transformer.gnn.' + k] for k in gnn_dict.keys()}
        # pro_state_dict = {k: transformermodel['transformer.input_proj.' + k] for k in pro_dict.keys()}
        # # print(state_dict.keys())
        # gnn_dict.update(gnn_state_dict)
        # pro_dict.update(pro_state_dict)
        # tracker.gnn.load_state_dict(gnn_dict)
        # tracker.input_proj.load_state_dict(pro_dict)
        ###########################################
        model.to(device)
        model.eval()

        pre_seq_name = ''
        frame_offset = 1
        num_frames = 0
        pub_dets = None
        start = 0
        show_image = False
        save_images = True
        save_videos = True
        accs_r = []
        accs_i = []
        data_root_r = '/home/user/PycharmProjects/MOT_Project/mmtracking/data/MOT_00/test'
        data_root_i = '/home/user/PycharmProjects/MOT_Project/mmtracking/data/MOT_01/test'
        for idx, [samples_r, samples_i, meta] in enumerate(data_loader):
            # for idx, x in enumerate(data_loader):
            num_frames += 1
            [orig_size, im_name, video_name, orig_img_r, orig_img_i, trans] = meta[0]

            if os.path.exists(output_dir + "txt/" + video_name + '.txt'):
                continue

            if video_name != pre_seq_name:
                print("video_name", video_name)
                timer = Timer()
                save_dir = os.path.join(main_args.output_file, '..', 'outputs', main_args.exp_name,
                                        video_name) if save_images or save_videos else None
                mkdir_if_missing(save_dir + '/00')
                mkdir_if_missing(save_dir + '/01')
                # save results #
                output_dir_r = output_dir + "/00"
                output_dir_i = output_dir + "/01"
                if not (os.path.exists(output_dir + "/00/txt/" + pre_seq_name + '.txt') or os.path.exists(
                        output_dir + "/01/txt/" + pre_seq_name + '.txt')) and idx != 0:
                    # save results #
                    [results_r, results_i] = tracker.get_results()
                    print(f"r_Tracks found: {len(results_r)}")
                    print(f"i_Tracks found: {len(results_i)}")
                    print(f"Runtime for {pre_seq_name}: {time.time() - start :.2f} s.")
                    write_results_pedestrian(results_r, output_dir_r, seq_name=pre_seq_name,
                                             frame_offset=frame_offset)
                    write_results_pedestrian(results_i, output_dir_i, seq_name=pre_seq_name,
                                             frame_offset=frame_offset)

                    evaluator_r = Evaluator(data_root_r, pre_seq_name, 'mot')
                    evaluator_i = Evaluator(data_root_i, pre_seq_name, 'mot')
                    accs_r.append(evaluator_r.eval_file(output_dir + "/00/txt/" + pre_seq_name + '.txt'))
                    accs_i.append(evaluator_i.eval_file(output_dir + "/01/txt/" + pre_seq_name + '.txt'))

                # update pre_seq_name #
                pre_seq_name = video_name
                # pub_dets = ds.VidPubDet[video_name]

                # reset tracker #
                tracker.reset()
                # update inactive patience according to framerate
                seq_info_path = os.path.join(main_args.data_dir, "test2017", video_name, 'seqinfo.ini')
                print("seq_info_path ", seq_info_path)
                assert os.path.exists(seq_info_path)
                with open(seq_info_path, 'r') as f:
                    reader = csv.reader(f, delimiter='=')
                    for row in reader:
                        if 'frameRate' in row:
                            framerate = int(row[1])

                print('frameRate', framerate)
                tracker.inactive_patience = framerate / 30 * 60

                # init offset #
                frame_offset = 1  # int(im_name[:-4])
                print("frame offset : ", frame_offset)
                start = time.time()

            # starts with 0 #
            if pub_dets is not None:
                pub_det = pub_dets[int(im_name[:-4]) - 1]
            else:
                pub_det = []

            print("step frame: ", im_name)

            batch = {'frame_name': im_name, 'video_name': video_name, 'img_r': torch.from_numpy(orig_img_r).to(device),
                     'img_i': torch.from_numpy(orig_img_i).to(device),
                     'samples_r': samples_r[0].to(device),
                     'samples_i': samples_i[0].to(device),
                     'orig_size': orig_size.unsqueeze(0).to(device),
                     'dets': torch.FloatTensor(pub_det)[:, :-1].to(device) if len(pub_det) > 0 else [torch.zeros(0, 4),
                                                                                                     torch.zeros(0, 4)],
                     'trans': trans}
            # todo check the function
            [fim_id_r, online_tlwhs_r, online_ids_r], [fim_id_i, online_tlwhs_i,
                                                       online_ids_i] = tracker.dual_step_reidV3_pre_tracking_vit(batch)
            timer.toc()
            if show_image or save_dir is not None:
                online_im_r = vis.plot_tracking_gt(orig_img_r, online_tlwhs_r,
                                                online_ids_r, frame_id=fim_id_r,
                                                fps=1. / timer.average_time)
                online_im_i = vis.plot_tracking_gt(orig_img_i, online_tlwhs_i,
                                                online_ids_i, frame_id=fim_id_i,
                                                fps=1. / timer.average_time)
            # if show_image:
            #     cv2.imshow('online_im', np.concatenate(online_im_r, online_im_i), axis=1)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '00', im_name), online_im_r)
                cv2.imwrite(os.path.join(save_dir, '01', im_name), online_im_i)

        # save last results #
        output_dir_r = output_dir + "/00"
        output_dir_i = output_dir + "/01"
        if not (os.path.exists(output_dir + "/00/txt/" + pre_seq_name + '.txt') or os.path.exists(
                output_dir + "/01/txt/" + pre_seq_name + '.txt')) and idx != 0:
            # save results #
            [results_r, results_i] = tracker.get_results()
            print(f"r_Tracks found: {len(results_r)}")
            print(f"i_Tracks found: {len(results_i)}")
            print(f"Runtime for {pre_seq_name}: {time.time() - start :.2f} s.")
            write_results_pedestrian(results_r, output_dir_r, seq_name=pre_seq_name,
                                     frame_offset=frame_offset)
            write_results_pedestrian(results_i, output_dir_i, seq_name=pre_seq_name,
                                     frame_offset=frame_offset)

            evaluator_r = Evaluator(data_root_r, pre_seq_name, 'mot')
            evaluator_i = Evaluator(data_root_i, pre_seq_name, 'mot')
            accs_r.append(evaluator_r.eval_file(output_dir + "/00/txt/" + pre_seq_name + '.txt'))
            accs_i.append(evaluator_i.eval_file(output_dir + "/01/txt/" + pre_seq_name + '.txt'))

        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        seqs = [seq.strip() for seq in tracktor['seqs_str'].split()]
        summary_r = Evaluator.get_summary(accs_r, seqs, metrics)
        summary_i = Evaluator.get_summary(accs_i, seqs, metrics)
        strsummary_r = mm.io.render_summary(
            summary_r,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        strsummary_i = mm.io.render_summary(
            summary_i,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary_r)
        Evaluator.save_summary(summary_r, os.path.join(output_dir, 'summary_00_{}.xlsx'.format(main_args.exp_name)))
        print(strsummary_i)
        Evaluator.save_summary(summary_i, os.path.join(output_dir, 'summary_01_{}.xlsx'.format(main_args.exp_name)))









with open(curr_pth + '/cfgs/transcenter_cfg.yaml', 'r') as f:
    tracktor = yaml.safe_load(f)['tracktor']

main(tracktor)
