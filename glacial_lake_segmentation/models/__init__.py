import torch
import torch.nn as nn
import config
from models.unet import UNet
from models.simple_cnn import SimpleCNN
from models.aspp_segnet import ASPPSegNet

_REGISTRY = {
    "unet":        UNet,
    "simple_cnn":  SimpleCNN,
    "aspp_segnet": ASPPSegNet,
}


def get_model(model_name: str) -> nn.Module:
    if model_name not in _REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_REGISTRY.keys())}")
    model = _REGISTRY[model_name]()
    return model.to(config.DEVICE)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
