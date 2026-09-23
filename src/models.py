"""
Model definitions.

NormalizedModel  - wraps a network so it accepts [0,1] input (needed by torchattacks)
TrafficSignCNN   - the base classifier (Module 1)
DetectorCNN      - binary clean/adversarial detector (Module 2)
GradCAM          - explainability (Module 3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------- wrapper
class NormalizedModel(nn.Module):
    """Applies channel normalisation inside the forward pass.

    This lets attacks operate on raw [0,1] pixels while the underlying network
    still sees normalised inputs.
    """

    def __init__(self, model, mean=config.MEAN, std=config.STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


# ---------------------------------------------------------------- blocks
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


# ---------------------------------------------------------------- classifier
class TrafficSignCNN(nn.Module):
    """Compact 6-conv CNN. ~1.2M params, trains to >97% on GTSRB in ~15 epochs."""

    def __init__(self, num_classes=43):
        super().__init__()
        self.block1 = conv_block(3, 32)     # 32 -> 16
        self.block2 = conv_block(32, 64)    # 16 -> 8
        self.block3 = conv_block(64, 128)   # 8  -> 4
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def features(self, x):
        """Return the last conv feature map (used by Grad-CAM)."""
        x = self.block1(x)
        x = self.block2(x)
        return self.block3(x)

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------- detector
class DetectorCNN(nn.Module):
    """Binary classifier: 0 = clean, 1 = adversarial.

    Operates directly on the image. Adversarial perturbations leave
    high-frequency statistical traces that a CNN can learn to pick up.
    """

    def __init__(self):
        super().__init__()
        self.block1 = conv_block(3, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.block3(x)

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------- builders
def build_classifier(num_classes=None, device=None):
    num_classes = num_classes or config.NUM_CLASSES[config.DATASET]
    device = device or config.DEVICE
    return NormalizedModel(TrafficSignCNN(num_classes)).to(device)


def build_detector(device=None):
    device = device or config.DEVICE
    return NormalizedModel(DetectorCNN()).to(device)


def load_classifier(path=None, device=None):
    path = path or config.CLASSIFIER_CKPT
    device = device or config.DEVICE
    m = build_classifier(device=device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def load_detector(path=None, device=None):
    path = path or config.DETECTOR_CKPT
    device = device or config.DEVICE
    m = build_detector(device=device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


# ---------------------------------------------------------------- Grad-CAM
class GradCAM:
    """Minimal Grad-CAM. Hooks the last conv block of a NormalizedModel.

    Usage:
        cam = GradCAM(model)
        heatmap = cam(image_tensor)          # (H, W) numpy array in [0,1]
    """

    def __init__(self, normalized_model):
        self.model = normalized_model
        self.inner = normalized_model.model
        self.target_layer = self.inner.block3
        self.activations = None
        self.gradients = None
        self._register()

    def _register(self):
        def fwd_hook(_m, _i, out):
            self.activations = out.detach()

        def bwd_hook(_m, _gi, gout):
            self.gradients = gout[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W) tensor in [0,1]."""
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        # Global-average-pool the gradients to get channel weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # (1,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx
