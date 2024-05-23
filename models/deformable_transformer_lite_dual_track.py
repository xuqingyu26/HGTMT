import copy
from torch import nn
from models.PVT_v2 import PyramidVisionTransformerV2
from functools import partial
from models.ut import MSDeformAttn
import torch
import torch.nn.functional as F
from .position_encoding import PositionEmbeddingSine
from .featurefusion_network import FeatureFusionNetwork


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
                 pretrained="/scratch2/scorpio/yixu/Efficient_Transcenter/model_zoo/pvtv2_backbone/pvt_v2_b4.pth",
                 linear=False,
                 half=True):
        super().__init__()

        # input proj for projecting all channel numbers to hidden_dim #
        input_proj_list = []

        for stage_idx in range(4):
            in_channels = d_model[stage_idx]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))

        self.input_proj = nn.ModuleList(input_proj_list)
        # self.convert = nn.Conv2d(3, 3, kernel_size=5, padding=2, stride=1)
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
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.feature_exchange = FeatureFusionNetwork(d_model=256, nhead=4, num_featurefusion_layers=2,
                                                     dim_feedforward=512)

        self.inproj_r = nn.Conv2d(3, 256, kernel_size=1, stride=1)
        self.inproj_i = nn.Conv2d(3, 256, kernel_size=1, stride=1)
        self.in_r = nn.Conv2d(512, 3, kernel_size=1, stride=1)
        self.in_i = nn.Conv2d(512, 3, kernel_size=1, stride=1)
        self.ln1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln2 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)

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

    def forward(self, src_r, pre_src_r, src_i, pre_src_i, pre_cts=None, pre_memories=None, masks_flatten=None):

        spatial_shapes = []
        memories = []
        memories_r = []
        memories_i = []
        hs_r = []
        hs_i = []
        gather_feat_t = None
        pre_reference_points = []

        if pre_memories is None:
            no_pre = True
        else:
            no_pre = False

        if masks_flatten is None:
            masks_flatten = []

        b, c, h, w = src_r.tensors.shape
        device = src_r.tensors.device
        # pos_r = self.position_embedding(src_r)
        # pos_i = self.position_embedding(src_i)
        # mask_r = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        # mask_i = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        #
        # outs_r, outs_i = self.feature_exchange(self.inproj_r(src_r.tensors), mask_r, self.inproj_i(src_i.tensors),
        #                                      mask_i, pos_r, pos_i)

        h, w = h // 2, w // 2
        with torch.cuda.amp.autocast(self.half):
            # outs_r = self.pvt_encoder(self.in_r(outs_r))
            # outs_i = self.pvt_encoder(self.in_i(outs_i))
            outs_r = self.pvt_encoder(src_r.tensors)
            outs_i = self.pvt_encoder(src_i.tensors)
            # outs_i = self.pvt_encoder(self.convert(src_i.tensors))

        if no_pre:
            with torch.no_grad():
                with torch.cuda.amp.autocast(self.half):
                    pre_outs_r = self.pvt_encoder(pre_src_r.tensors)
                    pre_outs_i = self.pvt_encoder(pre_src_i.tensors)
                    # pre_outs_i = self.pvt_encoder(self.convert(pre_src_i.tensors))
                # pre info for the decoder
                pre_memories = []
                # pre_masks_flatten = []

        for stage in range(4):
            # 1/(2**(stage+3))
            h, w = h // 2, w // 2

            # for detection memory we use 1/4 #
            with torch.cuda.amp.autocast(self.half):
                hs_r.append(self.input_proj[stage](outs_r[stage]))
                hs_i.append(self.input_proj[stage](outs_i[stage]))

            if stage == 0:
                memories_r.append(hs_r[-1].flatten(2).transpose(1, 2).detach().clone())
                memories_i.append(hs_i[-1].flatten(2).transpose(1, 2).detach().clone())
                memories.append(self.ln1((torch.cat([memories_r[-1], memories_i[-1]], dim=-1))))
                spatial_shapes.append((h, w))
                # get memory with src #
                if isinstance(masks_flatten, list):
                    # todo can be optimized #
                    # get memory with src #
                    mask = src_r.mask.clone()
                    mask = F.interpolate(mask[None].float(), size=(h, w)).to(torch.bool)[0]
                    # for inference speed up
                    masks_flatten.append(mask.flatten(1))
                # get pre_memory with pre_src #
                if no_pre:
                    with torch.no_grad():
                        # # Prepare pre_mask, valid ratio, spatial shape #
                        with torch.cuda.amp.autocast(self.half):
                            pre_memory_r = self.input_proj[stage](pre_outs_r[stage]).detach()
                            pre_memory_i = self.input_proj[stage](pre_outs_i[stage]).detach()
                            pre_memory = self.ln2(torch.cat([pre_memory_r, pre_memory_i], dim=1))

                else:
                    pre_memory = pre_memories[0]

                if len(pre_memory.shape) == 3:
                    b, h_w, c = pre_memory.shape
                    # to bchw
                    pre_memory = pre_memory.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

                # gather pre_features and reference pts #
                # todo can use box and roi_aligned here.
                assert pre_memory.shape[2] == h and pre_memory.shape[3] == w
                # (x,y) to index
                pre_sample = pre_cts.clone()
                pre_sample[:, :, 0].clamp_(min=0, max=w - 1)
                pre_sample[:, :, 1].clamp_(min=0, max=h - 1)

                pre_sample[:, :, 0] /= w
                pre_sample[:, :, 1] /= h

                gather_feat_t = F.grid_sample(pre_memory, (2.0 * pre_sample - 1.0).unsqueeze(1),
                                              mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0,
                                :].transpose(1, 2)

                # make reference pts #
                pre_reference_points.append(pre_sample)

        if no_pre:
            del pre_outs_r, pre_outs_i
        del outs_r, outs_i
        # print(spatial_shapes)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_r.tensors.device)
        # level_start_indexes = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        level_start_indexes = spatial_shapes.new_zeros(1, )

        # transform to queries #
        pre_query_embed = self.my_forward_ffn(gather_feat_t)
        pre_reference_points = torch.stack(pre_reference_points, dim=2)
        if isinstance(masks_flatten, list):
            masks_flatten = torch.cat(masks_flatten, 1)
        # print(pre_reference_points.shape)

        # decoder
        # pre_query_embed is from sampled features of pre_memories,
        # pre_ref_pts are tracks pos at t-1, these are queries to question memory t
        # print([m.shape for m in masks_flatten])

        pre_hs_r = self.decoder(pre_tgt=pre_query_embed,
                                src_spatial_shapes=spatial_shapes, src_level_start_index=level_start_indexes,
                                pre_query_pos=pre_query_embed, src_padding_mask=masks_flatten,
                                src=torch.cat(memories_r, 1), pre_ref_pts=pre_reference_points)

        pre_hs_i = self.decoder(pre_tgt=pre_query_embed,
                                src_spatial_shapes=spatial_shapes, src_level_start_index=level_start_indexes,
                                pre_query_pos=pre_query_embed, src_padding_mask=masks_flatten,
                                src=torch.cat(memories_i, 1), pre_ref_pts=pre_reference_points)

        # for inference speed up #
        return [[hs_r, pre_hs_r]], [[hs_i, pre_hs_i]], memories, gather_feat_t, masks_flatten


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
        half=args.half
    )
