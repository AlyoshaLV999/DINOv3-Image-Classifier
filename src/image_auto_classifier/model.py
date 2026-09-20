"""DINOv3 backbone, exact LoRA targeting, metric head, and parameter EMA."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator, Mapping

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from .common import ProjectError
from .config import ModelConfig
from .losses import ArcMarginProduct


LORA_DROPOUT = 0.05
LORA_BLOCKS = frozenset({8, 9, 10, 11})  # Zero-based blocks 8..11 are human blocks 9..12.
LORA_LEAF_NAMES = ("qkv", "proj", "fc1", "fc2")
_LORA_TARGET_PATTERN = re.compile(
    r"(?:^|\.)layer\.(?P<block>\d+)\."
    r"(?:attention\.(?P<attention>q_proj|k_proj|v_proj|o_proj)|mlp\.(?P<mlp>up_proj|down_proj))$"
)


class ModelCompatibilityError(ProjectError):
    """The installed Transformers model topology does not match the locked architecture."""


def _lora_target_names(backbone: nn.Module) -> list[str]:
    """Return all native DINOv3 projections implementing logical targets in blocks 9–12."""
    logical_targets = {
        "qkv": ("q_proj", "k_proj", "v_proj"),
        "proj": ("o_proj",),
        "fc1": ("up_proj",),
        "fc2": ("down_proj",),
    }
    component_to_leaf = {
        component: leaf for leaf, components in logical_targets.items() for component in components
    }
    targets: dict[tuple[int, str], dict[str, str]] = {}
    for name, module in backbone.named_modules():
        match = _LORA_TARGET_PATTERN.search(name)
        if match is None:
            continue
        block = int(match.group("block"))
        if block not in LORA_BLOCKS:
            continue
        component = match.group("attention") or match.group("mlp")
        leaf = component_to_leaf[component]
        if not isinstance(module, nn.Linear):
            raise ModelCompatibilityError(f"Required DINOv3 LoRA target is not linear: {name}")
        key = (block, leaf)
        components = targets.setdefault(key, {})
        if component in components:
            raise ModelCompatibilityError(
                f"DINOv3 exposes duplicate LoRA targets for block {block + 1} component '{component}'"
            )
        components[component] = name

    expected = [(block, leaf) for block in sorted(LORA_BLOCKS) for leaf in LORA_LEAF_NAMES]
    missing = [
        f"block {block + 1}:{leaf}/{component}"
        for block, leaf in expected
        for component in logical_targets[leaf]
        if component not in targets.get((block, leaf), {})
    ]
    if missing:
        found = ", ".join(
            sorted(name for components in targets.values() for name in components.values())
        ) or "none"
        raise ModelCompatibilityError(
            "DINOv3 module topology is incompatible with the locked LoRA plan. "
            "Expected DINOv3 native q_proj/k_proj/v_proj/o_proj/up_proj/down_proj modules "
            "for logical qkv/proj/fc1/fc2 targets in blocks 9–12 "
            f"(24 physical targets); missing {', '.join(missing)}; found: {found}"
        )
    return [
        targets[(block, leaf)][component]
        for block, leaf in expected
        for component in logical_targets[leaf]
    ]



class MetricArcFaceModel(nn.Module):
    """Frozen DINOv3 + LoRA, 768→512→256 L2 metric head, and ArcFace classifier."""

    def __init__(
        self,
        model_config: ModelConfig,
        num_classes: int,
        *,
        local_files_only: bool = False,
        load_pretrained_weights: bool = True,
    ) -> None:
        super().__init__()
        try:
            if load_pretrained_weights:
                raw_backbone = AutoModel.from_pretrained(model_config.name, local_files_only=local_files_only)
            else:
                from transformers import AutoConfig

                backbone_config = AutoConfig.from_pretrained(model_config.name, local_files_only=local_files_only)
                raw_backbone = AutoModel.from_config(backbone_config)
        except Exception as error:
            location = "the local Hugging Face cache" if local_files_only else "Hugging Face"
            raise ProjectError(
                f"Unable to load {model_config.name} from {location}. The DINOv3 repository is access-gated: "
                "sign in to https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m, "
                "accept its access terms, then authenticate this machine with `hf auth login` "
                "or set the HF_TOKEN environment variable and run training once to populate the local cache. "
                "The project uses Hugging Face's default endpoint unless you explicitly set HF_ENDPOINT."
            ) from error

        hidden_size = getattr(raw_backbone.config, "hidden_size", None)
        if hidden_size != 768:
            raise ModelCompatibilityError(
                f"Expected DINOv3 ViT-B/16 hidden size 768, but loaded model exposes {hidden_size!r}"
            )
        target_names = _lora_target_names(raw_backbone)
        lora_config = LoraConfig(
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=LORA_DROPOUT,
            target_modules=target_names,
            bias="none",
        )
        self.backbone = get_peft_model(raw_backbone, lora_config)
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_("lora_" in name)
        self.metric_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256, bias=False),
        )
        self.arcface = ArcMarginProduct(256, num_classes, scale=30.0, margin=0.35)
        self.lora_target_names = tuple(target_names)

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=pixel_values, return_dict=True)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None or hidden.ndim != 3:
                raise ModelCompatibilityError("DINOv3 model output lacks pooler_output and usable last_hidden_state")
            pooled = hidden[:, 0]
        return F.normalize(self.metric_head(pooled), dim=1)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(pixel_values)
        return features, self.arcface(features, labels)

    def expand_arcface(self, num_classes: int) -> None:
        """Expand class rows while retaining all existing classifier weights exactly."""
        old = self.arcface
        if num_classes < old.weight.shape[0]:
            raise ProjectError("Refusing to remove ArcFace classes during resume")
        if num_classes == old.weight.shape[0]:
            return
        replacement = ArcMarginProduct(256, num_classes, old.scale, old.margin).to(old.weight.device)
        with torch.no_grad():
            replacement.weight[: old.weight.shape[0]].copy_(old.weight)
        self.arcface = replacement

    def set_lora_trainable(self, enabled: bool) -> None:
        for name, parameter in self.backbone.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(enabled)


class ParameterEMA:
    """EMA over every trainable head/LoRA parameter, leaving frozen DINO weights untouched."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            shadow.mul_(self.decay).add_(parameters[name].detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": {name: value.cpu() for name, value in self.shadow.items()}}

    def load_state_dict(self, state: Mapping[str, object], model: nn.Module, old_class_count: int | None = None) -> None:
        if not isinstance(state.get("shadow"), Mapping):
            raise ProjectError("Checkpoint EMA state is invalid")
        self.decay = float(state.get("decay", self.decay))
        incoming = state["shadow"]
        parameters = dict(model.named_parameters())
        loaded: dict[str, torch.Tensor] = {}
        for name, current in parameters.items():
            if name not in self.shadow:
                continue
            previous = incoming.get(name)
            if not isinstance(previous, torch.Tensor):
                raise ProjectError(f"Checkpoint EMA is missing parameter '{name}'")
            if previous.shape == current.shape:
                loaded[name] = previous.to(device=current.device, dtype=current.dtype).clone()
            elif name == "arcface.weight" and old_class_count is not None and previous.shape[0] == old_class_count:
                expanded = current.detach().clone()
                expanded[:old_class_count].copy_(previous.to(device=current.device, dtype=current.dtype))
                loaded[name] = expanded
            else:
                raise ProjectError(f"Checkpoint EMA shape mismatch for '{name}'")
        self.shadow = loaded

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        parameters = dict(model.named_parameters())
        originals = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            for name, shadow in self.shadow.items():
                parameters[name].data.copy_(shadow.to(device=parameters[name].device))
            yield
        finally:
            for name, original in originals.items():
                parameters[name].data.copy_(original)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            parameters[name].copy_(shadow.to(device=parameters[name].device))


