import copy
from torch import nn
from models.PVT_v2 import PyramidVisionTransformerV2
from functools import partial
from models.ut import MSDeformAttn
import torch
import torch.nn.functional as F
from .position_encoding import PositionEmbeddingSine
from .featurefusion_network import FeatureFusionNetwork
import numpy as np
from torch_geometric.nn import GATConv, GraphConv, GCNConv, AGNNConv, EdgeConv
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch
import util.misc as utils

class DeformableTransformer(nn.Module):
    """
    encoder:
        reference_points = images shape
        output=src, pos, reference_points, src, spatial_shapes, level_start_index, padding_mask

    decoder:
        reference_points = fc(query) 256-> 2
        output=tgt, query_pos, reference_points_input = reference_points*valid_ratio,
         src= memory = img feat maps +attention, src_spatial_shapes, src_level_start_index, src_padding_mask
    """

    def __init__(self,
                 d_model=(32, 64, 128, 256),
                 nhead=(1, 2, 8, 8),
                 num_encoder_layers=(2, 2, 2, 2),
                 num_decoder_layers=6,
                 dim_feedforward_ratio=(8, 8, 4, 4),
                 dropout=0.1,
                 activation="relu",
                 dec_n_points=4,
                 down_sample_ratio=(2, 2, 2, 2),
                 hidden_dim=256,
                 pretrained=None,
                 linear=False,
                 half=True,
                 gnn_layers_num=1):
        super().__init__()

        self.default_backbone_feature_resolution = [128, 160]
        # input proj for projecting all channel numbers to hidden_dim #

        self.project_r = nn.Sequential(nn.Conv2d(3, 3, 5, 1, 2),
                                       nn.LeakyReLU()
                                       )
        self.project_i = nn.Sequential(nn.Conv2d(3, 3, 5, 1, 2),
                                       nn.LeakyReLU()
                                       )
        input_proj_list = []
        for stage_idx in range(4):
            in_channels = d_model[stage_idx]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))

        self.input_proj = nn.ModuleList(input_proj_list)
        self.gnn_layers_num = gnn_layers_num
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, 512, dropout, 'relu',
                                                          n_levels=1, n_heads=8, n_points=dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers)

        # my_ffn
        self.linear1 = nn.Linear(hidden_dim, 512)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # self.ln1 = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.ln2 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)
        self.gnn = nn.ModuleList()
        for _ in range(gnn_layers_num):  # number of gnn
            self.gnn.append(GATConv(hidden_dim, hidden_dim // 8, heads=8))
        self._reset_parameters()

        # load_pvt #
        print(f"Loading: {pretrained}...")

        self.pvt_encoder = PyramidVisionTransformerV2(patch_size=4, embed_dims=d_model, num_heads=nhead,
                                                      mlp_ratios=dim_feedforward_ratio, qkv_bias=True,
                                                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                      depths=num_encoder_layers, sr_ratios=down_sample_ratio,
                                                      drop_rate=0.0, drop_path_rate=dropout, pretrained=pretrained,
                                                      linear=linear)

        self.half = half
        self.use_residual = True
        self.conv1 = nn.Conv2d(256, 128, 1)
        self.conv2 = nn.Conv2d(128, 1, 1)
        self.ReLU = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.center_feats_r = None
        self.center_feats_i = None
        self.gnn_pre_fea_i = None
        self.gnn_pre_fea_r = None
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    @staticmethod
    def build_edge_index_local(p_labels, box_length: int, fm_width: int, fm_height: int):
        # remove the (0, 0) point in p_labels
        vcenters = (p_labels.clone() * torch.tensor([[fm_width, fm_height]]).to(p_labels.device))

        centers = vcenters[(vcenters[:, 0] > 0) & (vcenters[:, 1] > 0)]

        lefttop = centers - torch.tensor([[box_length // 2, box_length // 2]]).to(p_labels.device)
        rightbottom = centers + torch.tensor([[box_length // 2, box_length // 2]]).to(p_labels.device)
        search_regions = torch.cat((lefttop, rightbottom), dim=1)
        search_regions[:, [0, 2]] = torch.clip(search_regions[:, [0, 2]], 0, fm_width - 1)
        search_regions[:, [1, 3]] = torch.clip(search_regions[:, [1, 3]], 0, fm_height - 1)

        default_boxes = torch.arange(box_length).repeat(1, box_length, 1).to(p_labels.device)
        row_offsets = (torch.arange(box_length) * fm_width)
        row_offsets = row_offsets.reshape(1, -1, 1).to(p_labels.device)
        default_boxes = default_boxes + row_offsets
        default_boxes = default_boxes.repeat(search_regions.shape[0], 1, 1)

        idx_offsets = (search_regions[:, 1] - 1) * fm_width + search_regions[:, 0]
        idx_offsets = idx_offsets.unsqueeze(-1).unsqueeze(-1)

        n_points_index = default_boxes + idx_offsets
        # n_points_index = n_points_index.flatten()
        # n_points_index = n_points_index[(n_points_index >= 0) & (n_points_index < fm_width * fm_height)]
        # n_points_index = torch.unique(n_points_index)
        n_points_index += len(p_labels)

        p_crop_index = torch.arange(len(centers)).to(p_labels.device)
        src = torch.tensor([]).to(p_labels.device)
        dst = torch.tensor([]).to(p_labels.device)
        for i in torch.arange(len(centers)):
            ts, td = torch.meshgrid(p_crop_index[i:i + 1], n_points_index[i].flatten().long())
            src = torch.cat([src, ts.flatten()])
            dst = torch.cat([dst, td.flatten()])
        # src, dst = torch.meshgrid(p_crop_index, n_points_index.long())
        edge_index_forward = torch.stack((src, dst))
        # edge_index_backward = torch.stack((dst, src))
        edge_index_forward = edge_index_forward.T[
            (edge_index_forward[1, :] >= 0) & (edge_index_forward[1, :] < fm_width * fm_height)].T
        edge_index = torch.cat((edge_index_forward, torch.stack((edge_index_forward[1, :], edge_index_forward[0, :]))),
                               dim=1)
        return edge_index

    @staticmethod
    def build_edge_index_sparse(pc, pd, box_length: int, fm_height: int, fm_width: int):
        centers = (pc.clone() * torch.tensor([[fm_height, fm_width]]).to(pc.device))
        d_center = (pd.clone() * torch.tensor([[fm_height, fm_width]]).to(pc.device))

        dist = torch.cdist(centers, d_center, p=2)
        edge_index = (dist < box_length).nonzero().T
        edge_index = edge_index.T[(centers[edge_index[0, :], 0] > 0) & (centers[edge_index[0, :], 1] > 0) & (
                d_center[edge_index[1, :], 0] > 0) & (d_center[edge_index[1, :], 1] > 0)].T
        edge_index[1] += centers.shape[0]
        edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=[0])], dim=1)
        return edge_index

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    # transform memory to a query embed
    def my_forward_ffn(self, memory):
        memory2 = self.linear2(self.dropout2(self.activation(self.linear1(memory))))
        return self.norm2(memory + self.dropout3(memory2))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_sum_h = torch.sum(~mask, 1, keepdim=True)
        valid_H, _ = torch.max(valid_sum_h, dim=2)
        valid_H.squeeze_(1)
        valid_sum_w = torch.sum(~mask, 2, keepdim=True)
        valid_W, _ = torch.max(valid_sum_w, dim=1)
        valid_W.squeeze_(1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio_h = torch.clamp(valid_ratio_h, min=1e-3, max=1.1)
        valid_ratio_w = torch.clamp(valid_ratio_w, min=1e-3, max=1.1)
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, src_r, pre_src_r, src_i, pre_src_i, pre_cts_r=None, pre_cts_i=None, no_pre_cts=False,
                tracking_r=None, tracking_i=None,
                pre_memories_r=None, pre_memories_i=None, masks_flatten=None):

        spatial_shapes = []
        memories_r = []
        memories_i = []
        hs_r = []
        hs_i = []
        gather_feat_r = None
        gather_feat_i = None
        pre_reference_points_r = []
        pre_reference_points_i = []

        if pre_memories_r is None:
            no_pre = True
        else:
            no_pre = False

        if masks_flatten is None:
            masks_flatten = []

        b, c, h, w = src_r.tensors.shape
        device = src_r.tensors.device

        h, w = h // 2, w // 2
        with torch.cuda.amp.autocast(self.half):
            outs_r = self.pvt_encoder(self.project_r(src_r.tensors))
            outs_i = self.pvt_encoder(self.project_i(src_i.tensors))
        if no_pre:
            with torch.no_grad():
                with torch.cuda.amp.autocast(self.half):
                    pre_outs_r = self.pvt_encoder(self.project_r(pre_src_r.tensors))
                    pre_outs_i = self.pvt_encoder(self.project_i(pre_src_i.tensors))

                pre_memories_r = []
                pre_memories_i = []

        for stage in range(4):
            # 1/(2**(stage+3))
            h, w = h // 2, w // 2

            # for detection memory we use 1/4 #
            with torch.cuda.amp.autocast(self.half):
                hs_r.append(self.input_proj[stage](outs_r[stage]))
                hs_i.append(self.input_proj[stage](outs_i[stage]))

            if stage == 0:
                # memories_r.shape = torch.Size([1, 20480, 256])
                memories_r.append(hs_r[-1].flatten(2).transpose(1, 2).detach().clone())
                memories_i.append(hs_i[-1].flatten(2).transpose(1, 2).detach().clone())

                spatial_shapes.append((w, h))
                # get memory with src #
                if isinstance(masks_flatten, list):
                    # todo can be optimized #
                    # get memory with src #
                    mask = src_r.mask.clone()
                    mask = F.interpolate(mask[None].float(), size=(w, h)).to(torch.bool)[0]
                    # for inference speed up
                    masks_flatten.append(mask.flatten(1))

                # get pre_memory with pre_src #
                if no_pre:
                    with torch.no_grad():
                        # # Prepare pre_mask, valid ratio, spatial shape #
                        with torch.cuda.amp.autocast(self.half):
                            pre_memory_r = self.input_proj[stage](pre_outs_r[stage]).detach()
                            pre_memory_i = self.input_proj[stage](pre_outs_i[stage]).detach()
                            # pre_memory = self.ln2(torch.cat([pre_memory_r, pre_memory_i], dim=1))

                else:
                    pre_memory_r = pre_memories_r[0]
                    pre_memory_i = pre_memories_i[0]

                if len(pre_memory_r.shape) == 3:
                    b, h_w, c = pre_memory_r.shape
                    # to bchw
                    pre_memory_r = pre_memory_r.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
                    pre_memory_i = pre_memory_i.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

                # gather pre_features and reference pts #
                # todo can use box and roi_aligned here.
                # Further interact feature with heterogeneous graph
                assert pre_memory_r.shape[2] == h and pre_memory_r.shape[3] == w

                # (x,y) to index
                pre_sample_r = pre_cts_r.clone()
                pre_sample_r[:, :, 0].clamp_(min=0, max=w - 1)
                pre_sample_r[:, :, 1].clamp_(min=0, max=h - 1)
                pre_sample_i = pre_cts_i.clone()
                pre_sample_i[:, :, 0].clamp_(min=0, max=w - 1)
                pre_sample_i[:, :, 1].clamp_(min=0, max=h - 1)

                pre_sample_r[:, :, 0] /= w
                pre_sample_r[:, :, 1] /= h
                pre_sample_i[:, :, 0] /= w
                pre_sample_i[:, :, 1] /= h

                pos = self.position_embedding(
                    utils.NestedTensor(pre_memory_r, torch.ones_like(pre_memory_r)[:, 0, :, :].bool()))

                pos_r = F.grid_sample(pos, (2.0 * pre_sample_r - 1.0).unsqueeze(1),
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0,
                                    :].transpose(1, 2)
                pos_i = F.grid_sample(pos, (2.0 * pre_sample_i - 1.0).unsqueeze(1),
                                      mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0,
                        :].transpose(1, 2)

                gather_pre_feat_r = F.grid_sample(pre_memory_r, (2.0 * pre_sample_r - 1.0).unsqueeze(1),
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0,
                                    :].transpose(1, 2)
                gather_pre_feat_i = F.grid_sample(pre_memory_i, (2.0 * pre_sample_i - 1.0).unsqueeze(1),
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0,
                                    :].transpose(1, 2)


                track_cts_r = pre_cts_r.clone() + tracking_r
                track_cts_r[:, :, 0].clamp_(min=0, max=w - 1)
                track_cts_r[:, :, 1].clamp_(min=0, max=h - 1)
                track_sample_r = track_cts_r.clone()
                track_sample_r[:, :, 0] = track_cts_r[:, :, 0] / w
                track_sample_r[:, :, 1] = track_cts_r[:, :, 1] / h
                track_cts_i = pre_cts_i.clone() + tracking_i
                track_cts_i[:, :, 0].clamp_(min=0, max=w - 1)
                track_cts_i[:, :, 1].clamp_(min=0, max=h - 1)
                track_sample_i = track_cts_i.clone()
                track_sample_i[:, :, 0] = track_cts_i[:, :, 0] / w
                track_sample_i[:, :, 1] = track_cts_i[:, :, 1] / h

                    # gather_feat_r = F.grid_sample(hs_r[-1].detach(), (2.0 * track_sample_r - 1.0).unsqueeze(1),
                    #                               mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0,
                    #                 :].transpose(1, 2)
                    # gather_feat_i = F.grid_sample(hs_i[-1].detach(), (2.0 * track_sample_i - 1.0).unsqueeze(1),
                    #                               mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0,
                    #                 :].transpose(1, 2)




                # make reference pts #
                pre_reference_points_r.append(pre_sample_r)
                pre_reference_points_i.append(pre_sample_i)

                # todo further check
                if not no_pre_cts:
                    pre_centers = torch.cat([pre_sample_r, pre_sample_i], dim=1)
                    ###########################################################################################################
                    # track_cts = torch.cat([track_cts_r, track_cts_i], dim=1)
                    gather_feat = torch.cat([gather_pre_feat_r, gather_pre_feat_i], dim=1)

                    data_list_r = []
                    data_list_i = []
                    center_inds = []
                    # edge_gt_r = []
                    # edge_gt_i = []
                    offset = 0
                    edge_inds_r = []
                    edge_inds_i = []
                    B, C, H, W = pre_memory_r.shape
                    for i in range(B):
                        edge_index1 = self.build_edge_index_local(
                            p_labels=pre_centers[i],
                            box_length=10,
                            fm_height=self.default_backbone_feature_resolution[0],
                            fm_width=self.default_backbone_feature_resolution[1]
                        )

                        edge_index2 = self.build_edge_index_sparse(pre_sample_r[i], pre_sample_i[i], 20,
                                                                   fm_height=self.default_backbone_feature_resolution[
                                                                       0],
                                                                   fm_width=self.default_backbone_feature_resolution[
                                                                       1])  # need to change



                        edge_index = torch.cat([edge_index1, edge_index2], dim=1).long()

                        graph_nodes_r = torch.cat((gather_feat[i], memories_r[0][i].contiguous()),
                                                  dim=0)
                        graph_nodes_i = torch.cat((gather_feat[i], memories_i[0][i].contiguous()),
                                                  dim=0)
                        data_list_r.append(gData(x=graph_nodes_r, edge_index=edge_index))
                        data_list_i.append(gData(x=graph_nodes_i, edge_index=edge_index))
                        center_ind = offset + len(gather_feat[i]) + torch.arange(len(pre_memory_r[i].reshape(C, -1).T))
                        center_inds.append(center_ind)


                        # if self.training:
                            # track_ct_int_i = torch.tensor(track_cts_i, dtype=torch.int32)
                            # track_ct_int_r = torch.tensor(track_cts_r, dtype=torch.int32)
                            # index_track_i = (track_ct_int_i[i, :, 1] * W + track_ct_int_i[i, :, 0] + gather_feat[i].shape[
                            #     0]).unsqueeze(0)
                            # index_track_r = (track_ct_int_r[i, :, 1] * W + track_ct_int_r[i, :, 0] + gather_feat[i].shape[
                            #     0]).unsqueeze(0)
                            # index_ct_r = torch.arange(pre_sample_r.shape[1]).unsqueeze(0).to(track_ct_int_i.device)
                            # index_ct_i = (torch.arange(pre_sample_i.shape[1]) + gather_feat_r[i].shape[0]).unsqueeze(0).to(
                            #     track_ct_int_i.device)

                            # edge_index_gt_r = torch.cat([index_ct_r, index_track_r], dim=0)
                            # edge_index_gt_i = torch.cat([index_ct_i, index_track_i], dim=0)
                            # edge_gt_r.append(edge_index_gt_r)
                            # edge_gt_i.append(edge_index_gt_i)
                        edge_ind_r = offset + torch.arange(pre_sample_r.shape[1])
                        edge_ind_i = offset + torch.arange(pre_sample_i.shape[1]) + gather_pre_feat_r[i].shape[0]
                        edge_inds_r.append(edge_ind_r)
                        edge_inds_i.append(edge_ind_i)
                        offset += len(graph_nodes_i)
                    graph_r = Batch.from_data_list(data_list_r)
                    graph_i = Batch.from_data_list(data_list_i)
                    center_inds = torch.cat(center_inds)
                    if self.training:
                        edge_inds_r = torch.cat(edge_inds_r)
                        edge_inds_i = torch.cat(edge_inds_i)
                    # pass through gnn
                    gnn_feat_r = graph_r.x
                    gnn_feat_i = graph_i.x

                    for gnn in self.gnn:
                        gnn_out_r = gnn(gnn_feat_r, graph_r.edge_index)
                        gnn_out_i = gnn(gnn_feat_i, graph_i.edge_index)
                        if self.use_residual:
                            gnn_feat_r = gnn_feat_r + gnn_out_r
                            gnn_feat_i = gnn_feat_i + gnn_out_i
                        else:
                            gnn_feat_r = gnn_out_r
                            gnn_feat_i = gnn_out_i
                        # if self.return_pre_gnn_layer_outputs:
                        #     cached_feats.append(gnn_feat[center_inds].reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous())
                    # slice the features corresponding to the centers of each image in the batch
                    self.center_feats_r = gnn_feat_r[center_inds].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                    self.center_feats_i = gnn_feat_i[center_inds].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                    gnn_pre_fea_r = gnn_feat_r[edge_inds_r].reshape(B, gather_pre_feat_r.shape[1], C).permute(0, 2,
                                                                                                              1).contiguous().unsqueeze(
                        -1)
                    gnn_pre_fea_i = gnn_feat_i[edge_inds_i].reshape(B, gather_pre_feat_i.shape[1], C).permute(0, 2,
                                                                                                              1).contiguous().unsqueeze(
                        -1)

                    tracks_feat_r = F.grid_sample(self.center_feats_r, (2.0 * track_sample_r - 1.0).unsqueeze(1),
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)[:,
                                    :, 0, :].unsqueeze(-2)
                    tracks_feat_i = F.grid_sample(self.center_feats_i, (2.0 * track_sample_i - 1.0).unsqueeze(1),
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)[:,
                                    :, 0, :].unsqueeze(-2)

                    result_edge_feature_r = gnn_pre_fea_r - tracks_feat_r
                    result_edge_feature_i = gnn_pre_fea_i - tracks_feat_i
                    cross_edge_feature = tracks_feat_r.permute(0, 1, -1, -2) - tracks_feat_i
                    if self.training:
                        edge_r = self.sig(self.conv2(self.dropout2(self.ReLU(self.conv1(result_edge_feature_r)))))
                        edge_i = self.sig(self.conv2(self.dropout2(self.ReLU(self.conv1(result_edge_feature_i)))))
                        edge_c = self.sig(self.conv2(self.dropout2(self.ReLU(self.conv1(cross_edge_feature)))))
                    else:
                        # self.gnn_pre_fea_r = gnn_feat_r[edge_inds_r].reshape(B, gather_pre_feat_r.shape[1], C).permute(0, 2,
                        #                                                                                       1).contiguous().unsqueeze(
                        #     -1)
                        # self.gnn_pre_fea_i = gnn_feat_i[edge_inds_i].reshape(B, gather_pre_feat_i.shape[1], C).permute(0, 2,
                        #                                                                                       1).contiguous().unsqueeze(
                        #     -1)
                        #
                        # tracks_feat_r = self.center_feats_r.clone().reshape(B, C, H * W).unsqueeze(-2)
                        # tracks_feat_i = self.center_feats_i.clone().reshape(B, C, H * W).unsqueeze(-2)
                        #
                        # result_edge_feature_r = self.gnn_pre_fea_r - tracks_feat_r
                        # result_edge_feature_i = self.gnn_pre_fea_i - tracks_feat_i
                        # cross_edge_feature = tracks_feat_r.permute(0, 1, -1, -2) - tracks_feat_i
                        edge_c = self.sig(self.conv2(self.ReLU(self.conv1(cross_edge_feature))))
                        edge_r = self.sig(self.conv2(self.ReLU(self.conv1(result_edge_feature_r))))
                        edge_i = self.sig(self.conv2(self.ReLU(self.conv1(result_edge_feature_i))))

                    hs_r[-1] = self.center_feats_r
                    hs_i[-1] = self.center_feats_i

                    memories_r[-1] = hs_r[-1].flatten(2).transpose(1, 2).detach().clone()
                    memories_i[-1] = hs_i[-1].flatten(2).transpose(1, 2).detach().clone()


                else:
                    edge_r, edge_i, edge_c = None, None, None


        if no_pre:
            del pre_outs_r, pre_outs_i
        del outs_r, outs_i
        # print(spatial_shapes)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        # level_start_indexes = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        level_start_indexes = spatial_shapes.new_zeros(1, )

        pre_query_embed_r = self.my_forward_ffn(gather_pre_feat_r)
        pre_query_embed_i = self.my_forward_ffn(gather_pre_feat_i)

        pre_reference_points_r = torch.stack(pre_reference_points_r, dim=2)
        pre_reference_points_i = torch.stack(pre_reference_points_i, dim=2)

        if isinstance(masks_flatten, list):
            masks_flatten = torch.cat(masks_flatten, 1)

        ################################################################################################################
        pre_hs_r = self.decoder(pre_tgt=pre_query_embed_r,
                                src_spatial_shapes=spatial_shapes, src_level_start_index=level_start_indexes,
                                pre_query_pos=pre_query_embed_r, src_padding_mask=masks_flatten,
                                src=torch.cat(memories_r, 1), pre_ref_pts=pre_reference_points_r)

        pre_hs_i = self.decoder(pre_tgt=pre_query_embed_i,
                                src_spatial_shapes=spatial_shapes, src_level_start_index=level_start_indexes,
                                pre_query_pos=pre_query_embed_i, src_padding_mask=masks_flatten,
                                src=torch.cat(memories_i, 1), pre_ref_pts=pre_reference_points_i)

        # for inference speed up #

        return [[hs_r, pre_hs_r]], [
                [hs_i, pre_hs_i]], [edge_r, edge_i, edge_c]



class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.norm2 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn pre
        self.pre_linear1 = nn.Linear(d_model, d_ffn)
        self.pre_dropout3 = nn.Dropout(dropout)
        self.pre_linear2 = nn.Linear(d_ffn, d_model)
        self.pre_dropout4 = nn.Dropout(dropout)
        self.pre_norm3 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn_pre(self, pre_tgt):
        pre_tgt2 = self.pre_linear2(self.pre_dropout3(self.activation(self.pre_linear1(pre_tgt))))
        pre_tgt = pre_tgt + self.pre_dropout4(pre_tgt2)
        pre_tgt = self.pre_norm3(pre_tgt)
        return pre_tgt

    def forward(self, pre_tgt, pre_query_pos, src_spatial_shapes,
                level_start_index, src_padding_mask=None, src=None, pre_ref_pts=None):
        # self attention #
        # print("pre tgt.shape", pre_tgt.shape)
        # q = k = self.with_pos_embed(pre_tgt, pre_query_pos)
        # pre_tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), pre_tgt.transpose(0, 1))[0].transpose(0, 1)
        # pre_tgt = self.norm2(pre_tgt + self.dropout2(pre_tgt2))

        # cross attention, find objects at t with queries at t-1 #
        pre_tgt = pre_tgt + self.dropout1(self.cross_attn(self.with_pos_embed(pre_tgt, pre_query_pos),
                                                          pre_ref_pts, src, src_spatial_shapes, level_start_index,
                                                          src_padding_mask))

        # ffn: 2 fc layers with dropout, 256 -> 1024-> 256
        return self.forward_ffn_pre(self.norm1(pre_tgt))


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    # xyh #
    def forward(self, pre_tgt, src_spatial_shapes, src_level_start_index,
                pre_query_pos=None, src_padding_mask=None, src=None, pre_ref_pts=None):
        pre_output = pre_tgt
        for lid, layer in enumerate(self.layers):
            pre_output = layer(pre_tgt=pre_output, pre_query_pos=pre_query_pos,
                               src_spatial_shapes=src_spatial_shapes,
                               level_start_index=src_level_start_index,
                               src_padding_mask=src_padding_mask,
                               src=src, pre_ref_pts=pre_ref_pts)
        return pre_output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.d_model,
        nhead=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward_ratio=args.dim_feedforward_ratio,
        dropout=args.dropout,
        activation="relu",
        dec_n_points=args.dec_n_points,
        down_sample_ratio=args.down_sample_ratio,
        hidden_dim=args.hidden_dim,
        pretrained=args.pretrained,
        linear=args.linear,
        half=args.half,
        gnn_layers_num=args.gnn_layer_num
    )
