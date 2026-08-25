import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y, z):    # x: query, y: key, z: value
        batch_size, C, width, height = x.size()
        query_x = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key_y = self.key_conv(y).view(batch_size, -1, width * height)
        # Large spatial attention matrices can overflow in float16. Keep the
        # attention calculation in float32 while preserving AMP elsewhere.
        energy = torch.bmm(query_x.float(), key_y.float())
        attention = F.softmax(energy, dim=-1)
        value_y = self.value_conv(z).view(batch_size, -1, width * height).float()
        
        out = torch.bmm(value_y, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = (self.gamma.float() * out + x.float()).to(dtype=x.dtype)
        return out


class DPT(BaseModel):
    """Dual-branch dense-prediction backbone used by PSAFNet."""
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        use_pretrained_backbone=True,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        # branch 0 (subarea branch)
        self.pretrained0, self.scratch0 = _make_encoder(
            backbone,
            features,
            use_pretrained_backbone,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            use_vit_only=False,
            enable_attention_hooks=enable_attention_hooks,
        )
        # branch 1 (template branch)
        self.pretrained1, self.scratch1 = _make_encoder(
            backbone,
            features,
            use_pretrained_backbone,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            use_vit_only=False,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch0.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch0.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch0.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch0.refinenet4 = _make_fusion_block(features, use_bn)

        # Cross-attention fusion modules.
        self.crossatten1 = CrossAttention(256)
        self.crossatten2 = CrossAttention(512)
        self.crossatten3 = CrossAttention(768)
        self.crossatten4 = CrossAttention(768)
        

        self.scratch0.output_conv = head

        # Learnable embeddings for the 25%-FOV subareas.
        self.subarea_embedding = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 24, 22)) for i in range(25)]
        ) 

        # layer_1 size = (1,192,176)
        # layer_2 size = (1,96,88)
        # layer_3 size = (1,48,44)
        # layer_4 size = (1,24,22)
        

    def forward(self, x0, x1, label):
        if self.channels_last == True:
            x0.contiguous(memory_format=torch.channels_last)
            x1.contiguous(memory_format=torch.channels_last)
        
        # Subarea feature embedding.
        subarea_embedding = torch.stack([emb for emb in self.subarea_embedding])
        # encoder:
        # branch 0 (subarea branch)
        layer_1_b0, layer_2_b0, layer_3_b0, layer_4_b0 = forward_vit(self.pretrained0, x0, label, subarea_embedding)
        # branch 1 (template branch)
        layer_1_b1, layer_2_b1, layer_3_b1, layer_4_b1 = forward_vit(self.pretrained1, x1, label, subarea_embedding)

        # Patch-wise fusion blocks.
        layer_1 = self.crossatten1(layer_1_b0, layer_1_b1, layer_1_b1) + layer_1_b0
        layer_2 = self.crossatten2(layer_2_b0, layer_2_b1, layer_2_b1) + layer_2_b0
        layer_3 = self.crossatten3(layer_3_b0, layer_3_b1, layer_3_b1) + layer_3_b0
        layer_4 = self.crossatten4(layer_4_b0, layer_4_b1, layer_4_b1) + layer_4_b0


        layer_1_rn = self.scratch0.layer1_rn(layer_1)
        layer_2_rn = self.scratch0.layer2_rn(layer_2)
        layer_3_rn = self.scratch0.layer3_rn(layer_3)
        layer_4_rn = self.scratch0.layer4_rn(layer_4)

        path_4 = self.scratch0.refinenet4(layer_4_rn)
        path_3 = self.scratch0.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch0.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch0.refinenet1(path_2, layer_1_rn)

        
        
        # decoder
        out = self.scratch0.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x0, x1, label):
        inv_depth = super().forward(x0, x1, label).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        return inv_depth
