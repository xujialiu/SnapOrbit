"""
model_vit.py - Minimal ViT wrapper using timm

Single-scale Vision Transformer with configurable backbone (dinov3, dinov2,
retfound, visionfm). Uses timm's built-in classifier replacement, supports
LoRA fine-tuning, and optional per-target MLP heads.

-------------------------------------------------------------------------------
Full config example (YAML, only fields this script reads)
-------------------------------------------------------------------------------
Fields below are the complete set of keys that ``create_model`` touches.
Any other keys in the project's model YAMLs (e.g. ``model.module``,
``model.input_size``) are consumed elsewhere and are ignored here.

    dataset:
      nb_classes: 1                  # number of targets/classes

    model:
      name: dinov3_large             # "<backbone>_<size>"; see backbone table below
      drop_rate: 0.0                 # timm classifier-input dropout
      drop_path_rate: 0.0            # timm stochastic depth
      pretrained_ckpt: null          # required only for retfound / visionfm
      enable_fulltune: false         # mutually exclusive with enable_lora
      enable_lora: true              # mutually exclusive with enable_fulltune
      enable_last_vit: false         # optional; replaces default pooling with LaSt-ViT

      lora:                          # read only when enable_lora: true
        rank: 8
        alpha: 16
        dropout: 0.1
        bias: lora_only              # "none" | "all" | "lora_only"

      per_target_head:               # optional block; omit or enable=false to disable
        enable: false
        num_layers: 1                # 1 -> Linear only; >=2 -> MLP with (num_layers-1) hidden layers
        hidden_dim: 512              # ignored when num_layers == 1
        dropout: 0.1                 # ignored when num_layers == 1

      last_vit:                      # read only when enable_last_vit: true
        top_k: 49                    # # patches kept per channel
        sigma: 10.0                  # Gaussian low-pass width (in channel-FFT bins)
        eps: 1.0e-6                  # stability-score denominator floor
        score_variant: code          # "code" (default, matches ref. impl.) | "paper" (Eq. 5)

-------------------------------------------------------------------------------
Configuration reference for ``create_model(cfg)``
-------------------------------------------------------------------------------

Required
~~~~~~~~
cfg.model.name : str
    Format ``"<backbone>_<size>"``. ``_parse_model_name`` handles multi-part
    sizes (e.g. ``"huge_plus"``).
        - ``dinov3_<size>``  -> ``vit_<size>_patch16_dinov3.lvd1689m``
                                (pretrained via timm)
        - ``dinov2_<size>``  -> ``vit_<size>_patch14_dinov2.lvd142m``
                                (pretrained via timm)
        - ``retfound_<size>`` -> hardcoded ``vit_large_patch16_224.mae``
                                 (size suffix required but ignored; use
                                 ``retfound_large``; weights loaded from
                                 ``cfg.model.pretrained_ckpt``)
        - ``visionfm_<size>`` -> hardcoded ``vit_base_patch16_224.mae``
                                 (use ``visionfm_base``; weights loaded from
                                 ``cfg.model.pretrained_ckpt``)
    Backbone whitelist is the keys of ``MODEL_CONFIGS``.

cfg.dataset.nb_classes : int
    Number of output targets / classes. Drives both the timm classifier
    dimension and the number of per-target heads (when enabled).

Regularization (all have defaults, forwarded to timm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.model.drop_rate : float, default 0.0
    Dropout on the classifier input.
cfg.model.drop_path_rate : float, default 0.0
    Stochastic depth rate across transformer blocks.

Note: ``dynamic_img_size=True`` is always passed to ``timm.create_model``,
so the wrapped model accepts variable input resolutions.

Pretrained weights (conditional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.model.pretrained_ckpt : str
    Local checkpoint path. Required for ``retfound`` / ``visionfm`` (their
    ``MODEL_CONFIGS[...]["pretrained"]`` is ``False``); ignored for
    ``dinov3`` / ``dinov2`` which download weights through timm.
    Accepted checkpoint layouts (auto-detected by ``_load_pretrained_weights``):
        - ``{"model": state_dict, ...}``
        - ``{"state_dict": state_dict, ...}``
        - raw ``state_dict``
    Leading ``"model."`` prefix on keys is stripped automatically, and the
    classifier is reset after loading (pretrained classifier is discarded).

Training mode (mutually exclusive)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.model.enable_fulltune : bool
    If ``True`` -> all backbone parameters trainable (full fine-tune).
cfg.model.enable_lora : bool
    If ``True`` -> backbone frozen, LoRA adapters + classifier/head trainable.
Combination rules:
    - ``fulltune=True,  lora=False`` -> full fine-tune.
    - ``fulltune=False, lora=True``  -> LoRA + head trainable.
    - ``fulltune=False, lora=False`` -> linear probe (head only).
    - ``fulltune=True,  lora=True``  -> raises ``ValueError``.

LoRA config (read only when ``enable_lora=True``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.model.lora.rank : int
    LoRA rank ``r``.
cfg.model.lora.alpha : int
    LoRA alpha (scaling factor).
cfg.model.lora.dropout : float
    Dropout applied inside LoRA adapters.
cfg.model.lora.bias : str
    PEFT bias mode, one of ``{"none", "all", "lora_only"}``.
Note: ``target_modules`` is hardcoded to ``["qkv"]`` — adapters are injected
into attention QKV projections only.

Per-target heads (optional section; absent = disabled)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If this section is missing on ``cfg.model`` the behaviour is identical to
``enable=False`` (backwards compatible).

cfg.model.per_target_head.enable : bool, default False
    If ``True``, the timm classifier is replaced with ``nb_classes``
    independent MLP heads (one scalar per target). Concatenated output has
    shape ``(B, nb_classes)``. The timm model is built with
    ``num_classes=0`` so ``num_features`` is exposed directly.
cfg.model.per_target_head.num_layers : int, default 1
    - ``1``  -> ``Linear(feat_dim, 1)`` per target.
    - ``>=2`` -> ``(Linear -> GELU -> Dropout) * (num_layers-1) -> Linear(_, 1)``.
cfg.model.per_target_head.hidden_dim : int, default 512
    MLP hidden dimension, used only when ``num_layers >= 2``.
cfg.model.per_target_head.dropout : float, default 0.1
    Dropout between MLP layers (ignored when ``num_layers == 1``).

Per-target heads are always trainable regardless of the fulltune/LoRA mode
(their ``requires_grad`` is explicitly set to ``True`` after the mode is
applied to the backbone).

LaSt-ViT aggregation (optional section; absent = disabled)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Frequency-Aware Selective Aggregation from Shi et al., "Vision Transformers
Need More Than Registers" (CVPR 2026, arXiv:2602.22394). Replaces the
backbone's default pooling (CLS token or global average) with a channel-wise
top-K selection of patches judged most "stable" under a 1D Gaussian low-pass
filter applied along the channel dimension. No new trainable parameters are
introduced; the module is orthogonal to fulltune / LoRA / linear-probe modes
and to per-target heads.

cfg.model.enable_last_vit : bool, default False
    If ``True``, patch tokens are passed through ``LaStAggregator`` in place
    of the timm backbone's default pooling; ``fc_norm`` and ``head_drop`` are
    re-applied manually so the rest of the head path is unchanged.
cfg.model.last_vit.top_k : int, default 49
    Number of patches selected per channel (capped at ``N`` patches).
cfg.model.last_vit.sigma : float, default 10.0
    Std-dev of the 1D Gaussian low-pass filter applied in the channel-FFT
    domain. Larger ``sigma`` = wider passband = less aggressive smoothing.
cfg.model.last_vit.eps : float, default 1.0e-6
    Stability-score denominator floor to avoid division by zero.
cfg.model.last_vit.score_variant : str, default "code"
    Which stability-score numerator to use:
        - ``"code"``  -> ``S = x / (|x̂ - x| + eps)`` (matches the official
                         repo ``cls_pretrain/conf.py``; consistent with Eq. 7
                         which aggregates original ``x`` values; recommended
                         for fine-tuning downstream tasks).
        - ``"paper"`` -> ``S = x̂ / (|x̂ - x| + eps)`` (paper Eq. 5, with ``x̂``
                         = low-pass-filtered; use for paper-faithful experiments).
    Ranking usually agrees on clearly-stable and clearly-noisy patches;
    choice only matters in the mid-stability regime.

-------------------------------------------------------------------------------
ViT wrapper behaviour
-------------------------------------------------------------------------------
- ``ViT.forward``: if ``heads`` is ``None`` returns the timm model output
  directly; otherwise returns ``cat([h(feat) for h in heads], dim=1)``.
  When ``last_aggregator`` is set, ``forward`` calls ``forward_features`` on
  the (possibly PEFT-wrapped) timm backbone, strips ``num_prefix_tokens``
  (CLS + any register tokens), feeds the remaining patch tokens through
  ``LaStAggregator``, then re-applies ``fc_norm`` + ``head_drop`` + ``head``
  (or ``heads``).
- ``ViT.no_weight_decay``: excludes ``model.cls_token`` and ``model.pos_embed``.
- ``ViT.get_param_groups_lrd(weight_decay, layer_decay)``: builds layer-wise
  LR-decay param groups; compatible with raw timm models and PEFT-wrapped
  models (prefixes ``base_model.model.`` / ``model.base_model.model.`` are
  normalised).
"""

import torch
import torch.nn as nn
import timm

from peft import LoraConfig, get_peft_model


class LaStAggregator(nn.Module):
    """Frequency-Aware Selective Aggregation (LaSt-ViT).

    Replaces the backbone's default pooling with a channel-wise top-K
    selection of patches judged most "stable" under a 1D Gaussian low-pass
    filter applied along the channel dimension. Follows the formulation in
    Shi, Yu, Yang. "Vision Transformers Need More Than Registers."
    CVPR 2026, arXiv:2602.22394 (ref. impl.: github.com/ChengShiest/LAST-ViT).

    The module has no learnable parameters.
    """

    def __init__(
        self,
        top_k: int = 49,
        sigma: float = 10.0,
        eps: float = 1e-6,
        score_variant: str = "code",
    ):
        super().__init__()
        if score_variant not in ("code", "paper"):
            raise ValueError(
                f"score_variant must be 'code' or 'paper', got {score_variant!r}"
            )
        self.top_k = int(top_k)
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.score_variant = score_variant

    @staticmethod
    def _gaussian_kernel_1d(
        size: int, sigma: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Length-``size`` Gaussian aligned with ``fftshift`` DC (index ``size//2``)."""
        idx = torch.arange(size, device=device, dtype=dtype) - (size // 2)
        k = torch.exp(-(idx ** 2) / (2.0 * sigma ** 2))
        return k / k.max()

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, D) — patch tokens only (no CLS / register).
        Returns:
            (B, D) — aggregated CLS-like feature (paper Eq. 7).
        """
        B, N, D = patch_tokens.shape
        x = patch_tokens

        # Channel-wise 1D FFT + centered Gaussian low-pass + inverse FFT (Eq. 3-4).
        x_f = torch.fft.fft(x, dim=-1)
        x_f = torch.fft.fftshift(x_f, dim=-1)
        gk = self._gaussian_kernel_1d(D, self.sigma, x.device, x.real.dtype)
        x_f = x_f * gk  # broadcasts (D,) across (B, N, D)
        x_f = torch.fft.ifftshift(x_f, dim=-1)
        x_lp = torch.fft.ifft(x_f, dim=-1).real  # x̂_patch, (B, N, D)

        # Stability score. "paper" = Eq. 5 (numerator x̂); "code" = ref. impl.
        # (numerator x). Denominator |x̂ − x| + eps is identical.
        numer = x_lp if self.score_variant == "paper" else x
        score = numer / (torch.abs(x_lp - x) + self.eps)

        # Top-K per channel (dim=1 = patch dim) → indices I_K(j) (Eq. 6).
        k = min(self.top_k, N)
        _, idx = torch.topk(score, k=k, dim=1, largest=True)  # (B, k, D)

        # Q_CLS[j] = mean_{i ∈ I_K(j)} x_patch[i, j]  (Eq. 7).
        selected = torch.gather(x, dim=1, index=idx)  # (B, k, D)
        return selected.mean(dim=1)  # (B, D)


class ViT(nn.Module):
    """Wrapper around timm ViT for consistent interface with MultiScale."""

    def __init__(
        self,
        backbone: nn.Module,
        heads: nn.ModuleList | None = None,
        last_aggregator: nn.Module | None = None,
    ):
        super().__init__()
        self.model = backbone
        self.heads = heads
        self.last_aggregator = last_aggregator

    def _unwrap_timm(self) -> nn.Module:
        """Return the raw timm ViT (handles PEFT wrapping)."""
        base = getattr(self.model, "base_model", self.model)
        return getattr(base, "model", base)

    def forward(self, x):
        if self.last_aggregator is None:
            if self.heads is None:
                return self.model(x)
            feat = self.model(x)
            return torch.cat([h(feat) for h in self.heads], dim=1)

        # LaSt-ViT path: custom pooling, then re-apply timm head sub-modules.
        timm_model = self._unwrap_timm()
        all_tokens = timm_model.forward_features(x)  # (B, N+prefix, D)
        num_prefix = getattr(timm_model, "num_prefix_tokens", 1)
        patch_tokens = all_tokens[:, num_prefix:, :]
        feat = self.last_aggregator(patch_tokens)  # (B, D)
        feat = timm_model.fc_norm(feat)
        feat = timm_model.head_drop(feat)
        if self.heads is None:
            return timm_model.head(feat)
        return torch.cat([h(feat) for h in self.heads], dim=1)

    @property
    def blocks(self):
        """Access blocks, handling both raw timm and PEFT-wrapped models."""
        base = getattr(self.model, "base_model", self.model)
        model = getattr(base, "model", base)
        return model.blocks

    def no_weight_decay(self) -> set:
        """Parameters that should not have weight decay."""
        return {"model.cls_token", "model.pos_embed"}

    def _get_layer_id(self, name: str, num_layers: int) -> int:
        """Map parameter name to layer index for timm ViT."""
        # Handle PEFT wrapper prefix
        name = name.replace("model.base_model.model.", "model.")
        name = name.replace("base_model.model.", "model.")

        if name in ["model.cls_token", "model.pos_embed"]:
            return 0
        elif name.startswith("model.patch_embed"):
            return 0
        elif name.startswith("model.blocks"):
            # model.blocks.0.attn... -> extract block index
            return int(name.split(".")[2]) + 1
        else:  # norm, head, per-target heads
            return num_layers

    def get_param_groups_lrd(
        self, weight_decay: float, layer_decay: float
    ) -> list[dict]:
        """
        Parameter groups with layer-wise learning rate decay.

        Returns:
            List of dicts with keys: 'params', 'weight_decay', 'lr_scale'
        """
        param_groups = {}
        no_wd_list = self.no_weight_decay()

        num_layers = len(self.blocks) + 1
        layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Weight decay: skip 1D params and no_weight_decay list
            this_decay = 0.0 if param.ndim == 1 or name in no_wd_list else weight_decay

            layer_id = self._get_layer_id(name, num_layers)
            group_name = (
                f"layer_{layer_id}_{'no_decay' if this_decay == 0 else 'decay'}"
            )

            if group_name not in param_groups:
                param_groups[group_name] = {
                    "lr_scale": layer_scales[layer_id],
                    "weight_decay": this_decay,
                    "params": [],
                }
            param_groups[group_name]["params"].append(param)

        return list(param_groups.values())


MODEL_CONFIGS = {
    "dinov3": {
        "timm_name": "vit_{size}_patch16_dinov3.lvd1689m",
        "patch_size": 16,
        "pretrained": True,
    },
    "dinov2": {
        "timm_name": "vit_{size}_patch14_dinov2.lvd142m",
        "patch_size": 14,
        "pretrained": True,
    },
    "retfound": {
        "timm_name": "vit_large_patch16_224.mae",
        "patch_size": 16,
        "pretrained": False,
    },
    "visionfm": {
        "timm_name": "vit_base_patch16_224.mae",
        "patch_size": 16,
        "pretrained": False,
    },
}


def _get_timm_model_name(backbone: str, size: str) -> str:
    """Resolve timm model name from backbone and size."""
    config = MODEL_CONFIGS[backbone]
    return config["timm_name"].format(size=size)


def _build_per_target_heads(
    feat_dim: int,
    num_targets: int,
    num_layers: int,
    hidden_dim: int,
    dropout: float,
) -> nn.ModuleList:
    """Build an independent head per target.

    num_layers=1 -> Linear(feat_dim, 1) per target
    num_layers>=2 -> (Linear -> GELU -> Dropout) * (num_layers-1) -> Linear(_, 1)
    """
    heads = []
    for _ in range(num_targets):
        if num_layers <= 1:
            heads.append(nn.Linear(feat_dim, 1))
            continue
        layers: list[nn.Module] = []
        in_dim = feat_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        heads.append(nn.Sequential(*layers))
    return nn.ModuleList(heads)


def _load_pretrained_weights(model: nn.Module, ckpt_path: str) -> None:
    """Load pretrained weights from local checkpoint."""
    print(f"Loading pretrained weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'model.' prefix if present (common in some checkpoints)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded: {msg}")


def _parse_model_name(name: str) -> tuple[str, str]:
    """Parse model name in format 'backbone_size' (e.g., 'dinov3_base', 'dinov3_huge_plus').

    Matches against known backbone keys in MODEL_CONFIGS to handle
    multi-part sizes like 'huge_plus'.

    Returns:
        tuple: (backbone, size)
    """
    for key in sorted(MODEL_CONFIGS.keys(), key=len, reverse=True):
        prefix = key + "_"
        if name.startswith(prefix) and len(name) > len(prefix):
            return key, name[len(prefix):]
    raise ValueError(
        f"Invalid model name format: {name}. "
        f"Expected 'backbone_size' where backbone is one of {list(MODEL_CONFIGS.keys())} "
        "(e.g., 'dinov3_base', 'retfound_large')"
    )


def create_model(cfg):
    """
    Create single-scale ViT model with configurable backbone.

    Training modes (cfg.model.enable_fulltune / cfg.model.enable_lora):
        - fulltune=True, lora=False: all parameters trainable
        - fulltune=False, lora=True: LoRA + head trainable
        - fulltune=False, lora=False: head only
        - fulltune=True, lora=True: raises ValueError
    """
    backbone, size = _parse_model_name(cfg.model.name)
    num_classes = cfg.dataset.nb_classes

    if backbone not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown backbone: {backbone}. Choose from {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[backbone]
    timm_name = _get_timm_model_name(backbone, size)

    # Per-target head config (backwards compatible: missing = disabled)
    pth_cfg = getattr(cfg.model, "per_target_head", None)
    enable_pth = bool(pth_cfg and getattr(pth_cfg, "enable", False))
    effective_num_classes = 0 if enable_pth else num_classes

    # Create model with timm
    print(f"Creating model: {timm_name}")
    model = timm.create_model(
        timm_name,
        pretrained=config["pretrained"],
        num_classes=effective_num_classes,
        drop_rate=getattr(cfg.model, "drop_rate", 0.0),
        drop_path_rate=getattr(cfg.model, "drop_path_rate", 0.0),
        dynamic_img_size=True,
    )
    feat_dim = model.num_features

    # Load local checkpoint if needed (for retfound/visionfm)
    if not config["pretrained"] and cfg.model.pretrained_ckpt:
        _load_pretrained_weights(model, cfg.model.pretrained_ckpt)
        # Reset classifier after loading weights (weights were for pretraining)
        model.reset_classifier(effective_num_classes)

    # Build per-target heads (random init, not loaded from ckpt)
    heads = None
    if enable_pth:
        h_layers = int(getattr(pth_cfg, "num_layers", 1))
        h_hidden = int(getattr(pth_cfg, "hidden_dim", 512))
        h_dropout = float(getattr(pth_cfg, "dropout", 0.1))
        heads = _build_per_target_heads(
            feat_dim=feat_dim,
            num_targets=num_classes,
            num_layers=h_layers,
            hidden_dim=h_hidden,
            dropout=h_dropout,
        )
        print(
            f"Per-target heads: num_targets={num_classes}, num_layers={h_layers}, "
            f"hidden_dim={h_hidden}, dropout={h_dropout}, feat_dim={feat_dim}"
        )

    # Validate flags
    enable_fulltune = cfg.model.enable_fulltune
    enable_lora = cfg.model.enable_lora

    if enable_fulltune and enable_lora:
        raise ValueError("enable_fulltune and enable_lora cannot both be True")

    # Apply LoRA if enabled
    if enable_lora:
        lora_config = LoraConfig(
            r=cfg.model.lora.rank,
            lora_alpha=cfg.model.lora.alpha,
            target_modules=["qkv"],
            lora_dropout=cfg.model.lora.dropout,
            bias=cfg.model.lora.bias,
        )
        model = get_peft_model(model, lora_config)

    # Set requires_grad based on mode
    if enable_fulltune:
        print("Full finetune: all parameters trainable")
        for param in model.parameters():
            param.requires_grad = True
    elif enable_lora:
        # LoRA + head trainable, backbone frozen
        for name, param in model.named_parameters():
            if "lora" in name or "head" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # Head only
        for name, param in model.named_parameters():
            if "head" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Per-target heads (if any) are always trainable
    if heads is not None:
        for param in heads.parameters():
            param.requires_grad = True

    # Optional LaSt-ViT aggregator (backwards compatible: default disabled)
    last_aggregator = None
    if bool(getattr(cfg.model, "enable_last_vit", False)):
        last_cfg = getattr(cfg.model, "last_vit", None)
        top_k = int(getattr(last_cfg, "top_k", 49)) if last_cfg is not None else 49
        sigma = float(getattr(last_cfg, "sigma", 10.0)) if last_cfg is not None else 10.0
        eps = float(getattr(last_cfg, "eps", 1e-6)) if last_cfg is not None else 1e-6
        score_variant = (
            str(getattr(last_cfg, "score_variant", "code")) if last_cfg is not None else "code"
        )
        last_aggregator = LaStAggregator(
            top_k=top_k, sigma=sigma, eps=eps, score_variant=score_variant
        )
        print(
            f"LaSt-ViT enabled: top_k={top_k}, sigma={sigma}, eps={eps}, "
            f"score_variant={score_variant}"
        )

    # Wrap in ViT class for consistent interface
    wrapped = ViT(model, heads=heads, last_aggregator=last_aggregator)

    # Print trainable parameters summary
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    print(
        f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    # Print trainable layers
    print("Trainable layers:")
    for name, param in wrapped.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    return wrapped
