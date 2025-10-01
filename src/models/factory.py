"""Model factory centralizing segmentation model construction."""

from typing import Any, Dict
import torch.nn as nn


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build a segmentation model based on configuration."""
    model_config = config["model"]
    architecture = model_config["architecture"].lower()
    in_channels = model_config.get("in_channels", 3)
    out_channels = model_config.get("out_channels", 1)

    if architecture in {"unet", "unetplusplus", "deeplabv3", "deeplabv3plus", "fpn", "pspnet"}:
        import segmentation_models_pytorch as smp

        encoder = model_config.get("encoder", "resnet34")
        encoder_weights = model_config.get("encoder_weights", "imagenet")

        if architecture == "unet":
            return smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
            )
        if architecture == "unetplusplus":
            return smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
            )
        if architecture == "deeplabv3":
            return smp.DeepLabV3(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
            )
        if architecture == "deeplabv3plus":
            return smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
            )
        if architecture == "fpn":
            return smp.FPN(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
            )
        if architecture == "pspnet":
            return smp.PSPNet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
            )

    raise ValueError(f"Unknown architecture: {model_config['architecture']}")
