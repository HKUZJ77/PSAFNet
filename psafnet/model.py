"""PSAFNet model configured for the released POSDiag checkpoint."""

from dpt.models import DPTDepthModel


class PSAFNet(DPTDepthModel):
    """PSAFNet with a ResNet-50/ViT hybrid backbone and 24 x 22 embeddings."""

    def __init__(self, pretrained_backbone=True):
        super().__init__(
            path=None,
            scale=1.0,
            shift=0.0,
            invert=False,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
            use_pretrained_backbone=pretrained_backbone,
        )
