from __future__ import annotations

import sys
import warnings

import torch
import torch.nn as nn

from .paths import FOUNDER_DIR, HUBERT_DIR, PULSEPPG_DIR


def load_founder_1lead(pth: str, device: torch.device, linear_prob: bool = True) -> nn.Module:
    if str(FOUNDER_DIR) not in sys.path:
        sys.path.insert(0, str(FOUNDER_DIR))
    from net1d import Net1D

    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=1,
    )
    ckpt = torch.load(pth, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {k: v for k, v in state.items() if not k.startswith("dense.")}
    model.load_state_dict(state, strict=False)
    model.dense = nn.Linear(model.dense.in_features, 1)
    if linear_prob:
        for name, param in model.named_parameters():
            if "dense" not in name:
                param.requires_grad = False
    return model.to(device)


def load_founder_hr(
    backbone_pth: str,
    hr_ckpt: str,
    device: torch.device,
    linear_prob: bool = True,
) -> nn.Module:
    """Backbone from ECGFounder .pth, then overlay trained HR head/state."""
    model = load_founder_1lead(backbone_pth, device, linear_prob=linear_prob)
    ckpt = torch.load(hr_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model


def load_hubert_hr(pretrained_id: str, device: torch.device, unfreeze_last: int = 2):
    if str(HUBERT_DIR) not in sys.path:
        sys.path.insert(0, str(HUBERT_DIR))
    import hubert_ecg  # noqa: F401  registers AutoModel
    from hubert_ecg import HuBERTECG, HuBERTECGClassificationConfig, HuBERTECGForClassification

    backbone = HuBERTECG.from_pretrained(pretrained_id)
    d = {k: v for k, v in backbone.config.to_dict().items() if k != "model_type"}
    d["mask_time_prob"] = 0.0
    d["mask_feature_prob"] = 0.0
    cls_config = HuBERTECGClassificationConfig(
        num_labels=1,
        classifier_hidden_size=None,
        use_label_embedding=False,
        task_type="regression",
        **d,
    )
    model = HuBERTECGForClassification(cls_config)
    prefixed = {"hubert_ecg." + k: v for k, v in backbone.state_dict().items()}
    model.load_state_dict(prefixed, strict=False)
    model.set_feature_extractor_trainable(False)
    n_layers = model.config.num_hidden_layers
    model.set_transformer_blocks_trainable(n_blocks=min(unfreeze_last, n_layers))
    return model.to(device)


def load_hubert_trained(ckpt_dir: str, device: torch.device):
    """Load save_pretrained HR checkpoint. Force num_labels=1 (config.json omits it; default is 2)."""
    if str(HUBERT_DIR) not in sys.path:
        sys.path.insert(0, str(HUBERT_DIR))
    import hubert_ecg  # noqa: F401
    from hubert_ecg import HuBERTECGForClassification

    model = HuBERTECGForClassification.from_pretrained(ckpt_dir, num_labels=1)
    return model.to(device)


class PulsePPGHR(nn.Module):
    """Pulse-PPG ResNet encoder plus a linear bpm head. Input (B, 1, T) at 50 Hz."""

    def __init__(self, encoder: nn.Module, embed_dim: int = 512):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def _pulse_encoder() -> nn.Module:
    if str(PULSEPPG_DIR) not in sys.path:
        sys.path.insert(0, str(PULSEPPG_DIR))
    # Upstream InstanceNorm is built with the last block's width and applied to 1-channel PPG.
    warnings.filterwarnings(
        "ignore",
        message="input's size at dim=1 does not match num_features",
    )
    from pulseppg.nets.ResNet1D.ResNet1D_Net import Net

    return Net(
        in_channels=1,
        base_filters=128,
        kernel_size=11,
        stride=2,
        groups=1,
        n_block=12,
        finalpool="max",
    )


def _encoder_state(blob: object) -> dict:
    if isinstance(blob, dict) and isinstance(blob.get("net"), dict):
        state = blob["net"]
    elif isinstance(blob, dict) and isinstance(blob.get("state_dict"), dict):
        state = blob["state_dict"]
    elif isinstance(blob, dict):
        state = blob
    else:
        raise TypeError(f"Unrecognised Pulse-PPG checkpoint type {type(blob)}")
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("encoder."):
            key = key[len("encoder.") :]
        if key.startswith("head."):
            continue
        cleaned[key] = value
    return cleaned


def load_pulseppg(weights: str, device: torch.device, linear_prob: bool = False) -> nn.Module:
    """Pretrained Pulse-PPG ResNet with a new bpm head."""
    encoder = _pulse_encoder()
    blob = torch.load(weights, map_location="cpu", weights_only=False)
    missing, unexpected = encoder.load_state_dict(_encoder_state(blob), strict=False)
    missing = [key for key in missing if "num_batches_tracked" not in key]
    if missing or unexpected:
        raise RuntimeError(
            f"Pulse-PPG weight mismatch in {weights}: "
            f"missing={missing[:8]} n_missing={len(missing)} "
            f"unexpected={unexpected[:8]} n_unexpected={len(unexpected)}"
        )
    model = PulsePPGHR(encoder)
    if linear_prob:
        for param in model.encoder.parameters():
            param.requires_grad = False
    return model.to(device)


def load_pulseppg_hr(
    weights: str,
    hr_ckpt: str,
    device: torch.device,
    linear_prob: bool = False,
) -> nn.Module:
    """Pretrained encoder, then the fine-tuned PPG-Sport HR checkpoint."""
    model = load_pulseppg(weights, device, linear_prob=linear_prob)
    ckpt = torch.load(hr_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model
