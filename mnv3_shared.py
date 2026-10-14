"""Shared MobileNetV3-Large definitions for Phase 4 (image baseline) and Phase 5 (stage-2).

Plan ground rule 10: ONE constructor and ONE preprocessing path for every MobileNetV3
model, identical at training and inference. two_stage_eval.build_records imports
build_mnv3_large + preprocess_uint8 from here so the stage-2 weights load everywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Head recipe (base paper Appendix A): GAP -> Dense 128 + ReLU -> Dropout 0.3 -> Dense 1
HEAD_DROPOUT = 0.3
HEAD_HIDDEN = 128


class MineClassifier(nn.Module):
    """MobileNetV3-Large features + paper head, single logit output."""

    def __init__(self, pretrained: bool = True, dropout: float = HEAD_DROPOUT):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = torchvision.models.mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = backbone.classifier[0].in_features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, HEAD_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(HEAD_HIDDEN, 1),
        )
        self._in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.avgpool(self.features(x)))


def build_mnv3_large(pretrained: bool = True, dropout: float = HEAD_DROPOUT) -> MineClassifier:
    return MineClassifier(pretrained=pretrained, dropout=dropout)


def unfreeze_last_layers(model: MineClassifier, max_leaf_modules: int = 50):
    """Stage B: unfreeze the deepest inverted-residual blocks whose leaf-module count
    stays <= max_leaf_modules (plan: 'last 50 layers' of the Keras model, mapped to the
    nearest block boundary in torchvision and documented).

    Returns (n_unfrozen_params, first_unfrozen_feature_index).
    """
    for p in model.parameters():
        p.requires_grad = False

    leaf_types = (nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, nn.Hardsigmoid, nn.Linear, nn.ReLU, nn.Dropout)
    running = 0
    start_idx = len(model.features)
    for i in range(len(model.features) - 1, -1, -1):
        block_leaves = sum(1 for m in model.features[i].modules()
                           if isinstance(m, leaf_types) and m is not model.features[i])
        if running + block_leaves > max_leaf_modules and running > 0:
            break
        running += block_leaves
        start_idx = i
    for i in range(start_idx, len(model.features)):
        for p in model.features[i].parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_train, start_idx


def preprocess_uint8(images_chw_uint8: torch.Tensor) -> torch.Tensor:
    """images: (N, 3, H, W) uint8 RGB -> float32 ImageNet-normalised.

    The single normalisation path for MobileNetV3 at train and inference.
    """
    x = images_chw_uint8.to(torch.float32).div(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _selftest() -> None:
    import numpy as np

    # deterministic known image: 64x64, red channel 255
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[..., 0] = 255
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    out = preprocess_uint8(t)
    expect_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert torch.allclose(out[0, 0], torch.full((64, 64), expect_r), atol=1e-5), out[0, 0, 0, 0]
    assert abs(float(out[0, 1, 0, 0]) - (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]) < 1e-5
    m = build_mnv3_large(pretrained=False)
    y = m(torch.zeros(2, 3, 224, 224))
    assert y.shape == (2, 1), y.shape
    n, idx = unfreeze_last_layers(m)
    assert n > 0 and idx < len(m.features)
    print(f"selftest OK: head in_features={m._in_features}, stageB unfreezes features[{idx}:] "
          f"({n:,} params)")


if __name__ == "__main__":
    _selftest()
