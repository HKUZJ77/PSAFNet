"""Loss functions used by the public PSAFNet training script."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG-19 feature loss for one- or three-channel OSM tensors."""

    def __init__(self, pretrained=True):
        super().__init__()
        if hasattr(models, "VGG19_Weights"):
            weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            self.vgg = models.vgg19(weights=weights).features
        else:
            self.vgg = models.vgg19(pretrained=pretrained).features

        self.vgg.eval()
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False

        self.feature_layers = {"3", "8", "15", "22"}

    def forward(self, prediction, target):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("prediction and target must have the same [B,C,H,W] shape")
        if prediction.shape[1] == 1:
            prediction = prediction.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        if prediction.shape[1] != 3:
            raise ValueError("perceptual loss expects one or three input channels")

        loss = prediction.new_zeros(())
        for name, module in self.vgg._modules.items():
            prediction = module(prediction)
            target = module(target)
            if name in self.feature_layers:
                loss = loss + F.mse_loss(prediction, target)
            if name == "22":
                break
        return loss


def batch_sinkhorn_loss(SRM_pred, SRM_gt, p=1, blur=0.01, reduction="mean"):
    """Legacy differentiable Sinkhorn loss retained for reference.

    This function is not used by the public PSAFNet training script. Install
    the optional ``geomloss`` package before calling it.
    """

    try:
        from geomloss import SamplesLoss
    except ImportError as exc:
        raise ImportError(
            "batch_sinkhorn_loss requires geomloss: pip install geomloss"
        ) from exc

    loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur)

    if SRM_pred.dim() == 4:
        batch_size, channels, _, _ = SRM_pred.shape
        SRM_pred = SRM_pred.reshape(batch_size, channels, -1)
        SRM_gt = SRM_gt.reshape(batch_size, channels, -1)
        channel_losses = torch.zeros(
            batch_size,
            channels,
            device=SRM_pred.device,
            dtype=SRM_pred.dtype,
        )

        for channel in range(channels):
            pred_flat = SRM_pred[:, channel, :] / (
                SRM_pred[:, channel, :].sum(dim=1, keepdim=True) + 1e-8
            )
            gt_flat = SRM_gt[:, channel, :] / (
                SRM_gt[:, channel, :].sum(dim=1, keepdim=True) + 1e-8
            )
            channel_losses[:, channel] = loss_fn(pred_flat, gt_flat)
        batch_losses = channel_losses
    else:
        batch_size, _, _ = SRM_pred.shape
        SRM_pred = SRM_pred.reshape(batch_size, -1)
        SRM_gt = SRM_gt.reshape(batch_size, -1)
        pred_flat = SRM_pred / (SRM_pred.sum(dim=1, keepdim=True) + 1e-8)
        gt_flat = SRM_gt / (SRM_gt.sum(dim=1, keepdim=True) + 1e-8)
        batch_losses = loss_fn(pred_flat, gt_flat)

    if reduction == "mean":
        return batch_losses.mean()
    if reduction == "sum":
        return batch_losses.sum()
    return batch_losses


class HistogramEMDLoss(nn.Module):
    """Differentiable 1D EMD between image-intensity histograms.

    Pixels are assigned linearly to adjacent bins. For ordered one-dimensional
    bins, EMD is the integral of the absolute difference between cumulative
    histograms, which is the exact optimal-transport solution.
    """

    def __init__(
        self,
        num_bins=256,
        value_min=0.0,
        value_max=1.0,
        reduction="mean",
        eps=1e-8,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if value_max <= value_min:
            raise ValueError("value_max must be greater than value_min")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be mean, sum, or none")

        self.num_bins = num_bins
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.reduction = reduction
        self.eps = eps

    def _soft_histogram(self, image):
        image = image.float().clamp(self.value_min, self.value_max)
        image = image.flatten(start_dim=2)

        scale = (self.num_bins - 1) / (self.value_max - self.value_min)
        positions = (image - self.value_min) * scale
        lower = positions.floor().long()
        upper = (lower + 1).clamp(max=self.num_bins - 1)

        upper_weight = positions - lower.to(positions.dtype)
        lower_weight = 1.0 - upper_weight

        histogram = image.new_zeros(image.shape[0], image.shape[1], self.num_bins)
        histogram.scatter_add_(2, lower, lower_weight)
        histogram.scatter_add_(2, upper, upper_weight)
        return histogram / histogram.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, prediction, target):
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("prediction and target must have shape [B,H,W] or [B,C,H,W]")
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target shapes differ: {} vs {}".format(
                    tuple(prediction.shape), tuple(target.shape)
                )
            )

        pred_cdf = self._soft_histogram(prediction).cumsum(dim=-1)
        target_cdf = self._soft_histogram(target).cumsum(dim=-1)
        bin_width = (self.value_max - self.value_min) / (self.num_bins - 1)
        per_channel = (pred_cdf - target_cdf).abs().sum(dim=-1) * bin_width
        per_sample = per_channel.mean(dim=1)

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample


def batch_histogram_emd_loss(
    prediction,
    target,
    num_bins=256,
    value_min=0.0,
    value_max=1.0,
    reduction="mean",
):
    """Functional interface for the histogram EMD used by PSAFNet."""

    return HistogramEMDLoss(
        num_bins=num_bins,
        value_min=value_min,
        value_max=value_max,
        reduction=reduction,
    )(prediction, target)


class PSAFNetLoss(nn.Module):
    """Smooth L1 plus weighted perceptual and histogram EMD losses."""

    def __init__(
        self,
        perceptual_weight=0.1,
        emd_weight=0.1,
        num_bins=256,
        pretrained_perceptual=True,
    ):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.perceptual = PerceptualLoss(pretrained=pretrained_perceptual)
        self.histogram_emd = HistogramEMDLoss(num_bins=num_bins)
        self.perceptual_weight = perceptual_weight
        self.emd_weight = emd_weight

    def forward(self, prediction, target):
        smooth_l1 = self.smooth_l1(prediction, target)
        perceptual = self.perceptual(prediction, target)
        emd = self.histogram_emd(prediction, target)
        total = smooth_l1 + self.perceptual_weight * perceptual + self.emd_weight * emd
        return total, {
            "smooth_l1": smooth_l1.detach(),
            "perceptual": perceptual.detach(),
            "emd": emd.detach(),
        }
