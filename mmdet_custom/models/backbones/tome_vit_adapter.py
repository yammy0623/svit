import torch
import torch.nn.functional as F
from typing import Tuple


from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r
from .vit_adapter import ViTAdapter
from .adapter_modules import InteractionBlock, deform_inputs
from .base.vit import Attention, Block


# class ToMeBlock(Block):
#     """
#     Modifications:
#      - Apply ToMe between the attention and mlp blocks
#      - Compute and propogate token size and potentially the token sources.
#     """

#     def _drop_path1(self, x):
#         return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

#     def _drop_path2(self, x):
#         return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Note: this is copied from timm.models.vision_transformer.Block with modifications.
#         attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
#         x_attn, metric = self.attn(self.norm1(x), attn_size)
#         x = x + self._drop_path1(x_attn)

#         r = self._tome_info["r"].pop(0)
#         if r > 0:
#             # Apply ToMe here
#             merge, _ = bipartite_soft_matching(
#                 metric,
#                 r,
#                 self._tome_info["class_token"],
#                 self._tome_info["distill_token"],
#             )
#             if self._tome_info["trace_source"]:
#                 self._tome_info["source"] = merge_source(
#                     merge, x, self._tome_info["source"]
#                 )
#             x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

#         x = x + self._drop_path2(self.mlp(self.norm2(x)))
#         return x

# class ToMeAttention(Attention):
#     """
#     Modifications:
#      - Apply proportional attention
#      - Return the mean of k over heads from attention
#     """

#     def forward(
#         self, x: torch.Tensor, size: torch.Tensor = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
#         B, N, C = x.shape
#         qkv = (
#             self.qkv(x)
#             .reshape(B, N, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = (
#             qkv[0],
#             qkv[1],
#             qkv[2],
#         )  # make torchscript happy (cannot use tensor as tuple)

#         attn = (q @ k.transpose(-2, -1)) * self.scale

#         # Apply proportional attention
#         if size is not None:
#             attn = attn + size.log()[:, None, None, :, 0]

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         # Return k as well here
#         return x, k.mean(1)

import math

import torch.nn as nn
# from .base.vit import TIMMVisionTransformer
from .base.tome_vit import ToMeVisionTransformer
from mmdet.models.builder import BACKBONES
from .adapter_modules import SpatialPriorModule, InteractionBlockWithToMeSelection, deform_inputs
from torch.nn.init import normal_
from timm.models.layers import trunc_normal_
from ops.modules import MSDeformAttn

# replace the attention to obtain k for metric
# pass the k out from the transformer block
@BACKBONES.register_module()
class ToMeViTAdapter(ToMeVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True, *args, **kwargs):

        super().__init__(num_heads=num_heads, *args, **kwargs)

        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlockWithToMeSelection(r=10, dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor))
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)            

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1], indexes,
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]