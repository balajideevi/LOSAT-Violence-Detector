import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class ViolenceR3D18(nn.Module):
    """R3D-18 backbone with 2-class classification head."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models.video import R3D_18_Weights

            weights = R3D_18_Weights.DEFAULT if pretrained else None
            self.backbone = r3d_18(weights=weights)
        except ImportError:
            # Compatibility fallback for older torchvision versions.
            self.backbone = r3d_18(pretrained=pretrained)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_model(model_path: str | None = None, device: str = "cpu") -> nn.Module:
    """Load model for inference on CPU by default."""
    try:
        model = ViolenceR3D18(num_classes=2, pretrained=True)
    except Exception:
        # Fallback for offline environments where pretrained weights cannot be downloaded.
        model = ViolenceR3D18(num_classes=2, pretrained=False)
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
