"""
explainability_classification.py - Generate ViT explainability visualizations

For every test-fold image, runs the trained classification model and produces a
3-panel PNG (Original | Heatmap | Combined) saved under
``<result_root>/<result_name>/<paths.explainability>/<basename>.png``.

Methods (selected via ``cfg.explainability.method``):
    smoothgrad           - input-gradient saliency averaged over noisy copies
    attention_rollout    - Abnar & Zuidema (2020) rollout across all blocks
    attention_last_layer - last-block CLS-to-patch attention only

-------------------------------------------------------------------------------
Config schema (only fields read by THIS script; model/dataset/aug come from the
shared training config and are loaded the same way as train_classification.py)
-------------------------------------------------------------------------------

    paths:
      explainability: explainability     # subdir under <result_root>/<result_name>
      ckpt_path: /abs/path/to/checkpoint_best_X.pth

    explainability:
      method: smoothgrad                 # smoothgrad | attention_rollout | attention_last_layer
      alpha: 0.5                         # combined = alpha*heatmap + (1-alpha)*original
      explain_class: prediction          # prediction | ground_truth
      cmap: jet                          # any matplotlib colormap name
      other_params:                      # optional; per-method, defaults applied if absent
        n_samples: 25                    # smoothgrad
        noise_std: 0.15                  # smoothgrad
        discard_ratio: 0.9               # attention_rollout
        head_fusion: mean                # attention_rollout / attention_last_layer (mean | max)

CLI:
    conda run -n dinov3 python explainability_classification.py \\
        -c configs/<group>/<run>/config_explainability.yaml
"""

import importlib
import types
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from util.config import parse_args_and_config

# Allow CJK characters (e.g. Chinese filenames) to render in figure titles.
# Noto Sans CJK JP shares glyphs with the SC/TC variants and also covers ASCII,
# so it works as a single primary font without needing per-glyph fallback.
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Attention capture (shared by rollout / last-layer methods)
# ---------------------------------------------------------------------------

def _capturing_attn_forward(self, x, rope=None, attn_mask=None):
    """Drop-in replacement for timm Attention.forward / EvaAttention.forward
    that stores the softmax attention weights into ``self._capture_dict``.

    Handles both ``timm.models.vision_transformer.Attention`` (standard) and
    ``timm.models.eva.EvaAttention`` (used by DINOv3). Requires
    ``self.fused_attn = False`` so the explicit-softmax branch runs.
    """
    B, N, C = x.shape

    # qkv computation (standard `qkv` path or Eva `q_proj/k_proj/v_proj`)
    if getattr(self, "qkv", None) is not None:
        if getattr(self, "q_bias", None) is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if getattr(self, "qkv_bias_separate", False):
                qkv = self.qkv(x)
                qkv = qkv + qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    else:
        q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

    if hasattr(self, "q_norm"):
        q = self.q_norm(q)
        k = self.k_norm(k)

    if rope is not None:
        from timm.layers import apply_rot_embed_cat
        npt = self.num_prefix_tokens
        half = getattr(self, "rotate_half", False)
        q = torch.cat(
            [q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)],
            dim=2,
        ).type_as(v)
        k = torch.cat(
            [k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)],
            dim=2,
        ).type_as(v)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if attn_mask is not None:
        from timm.layers import maybe_add_mask
        attn = maybe_add_mask(attn, attn_mask)
    attn = attn.softmax(dim=-1)

    self._capture_dict[self._capture_idx] = attn

    attn = self.attn_drop(attn)
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    if hasattr(self, "norm"):
        x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _unwrap_timm(model):
    """Get the raw timm ViT, peeling ViT wrapper + optional PEFT wrapper."""
    if hasattr(model, "_unwrap_timm"):
        return model._unwrap_timm()
    inner = getattr(model, "model", model)
    base = getattr(inner, "base_model", inner)
    return getattr(base, "model", base)


def _get_patch_size(timm_model):
    pe = timm_model.patch_embed
    if hasattr(pe, "patch_size"):
        ps = pe.patch_size
        return ps[0] if isinstance(ps, (tuple, list)) else int(ps)
    return 16


def _normalize(arr):
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Explainers
# ---------------------------------------------------------------------------

class SmoothGradExplainer:
    """SmoothGrad: average |gradient| over noisy input copies (Smilkov 2017)."""

    def __init__(self, model, n_samples=25, noise_std=0.15):
        self.model = model
        self.n_samples = int(n_samples)
        self.noise_std = float(noise_std)

    def explain(self, x, target_class):
        H, W = x.shape[-2:]
        accum = torch.zeros(H, W, device=x.device)
        for _ in range(self.n_samples):
            noise = torch.randn_like(x) * self.noise_std
            xn = (x + noise).detach().requires_grad_(True)
            logits = self.model(xn)
            score = logits[0, target_class]
            grad = torch.autograd.grad(score, xn)[0]
            sal = grad.abs().squeeze(0).max(dim=0)[0]
            accum = accum + sal.detach()
        sal = accum / self.n_samples
        return _normalize(sal.cpu().numpy())


class AttentionRolloutExplainer:
    """Attention Rollout across all transformer blocks (Abnar & Zuidema 2020)."""

    def __init__(self, model, discard_ratio=0.9, head_fusion="mean"):
        if head_fusion not in ("mean", "max"):
            raise ValueError(f"head_fusion must be mean|max, got {head_fusion!r}")
        self.model = model
        self.discard_ratio = float(discard_ratio)
        self.head_fusion = head_fusion
        self.attentions = {}
        timm_model = _unwrap_timm(model)
        for i, block in enumerate(timm_model.blocks):
            attn = block.attn
            attn.fused_attn = False
            attn._capture_idx = i
            attn._capture_dict = self.attentions
            attn.forward = types.MethodType(_capturing_attn_forward, attn)

    def explain(self, x, target_class):
        self.attentions.clear()
        with torch.no_grad():
            _ = self.model(x)

        rollout = None
        for i in sorted(self.attentions.keys()):
            attn = self.attentions[i]                                # (1, H, N, N)
            attn = attn.mean(dim=1) if self.head_fusion == "mean" else attn.max(dim=1)[0]
            # Per-batch flatten + drop the lowest-attention positions
            B, Nt, _ = attn.shape
            flat = attn.reshape(B, -1)
            k = int(flat.size(-1) * self.discard_ratio)
            if k > 0:
                _, idx = flat.topk(k, dim=-1, largest=False)
                flat.scatter_(-1, idx, 0.0)
            attn = flat.reshape(B, Nt, Nt)
            # Add residual identity, row-normalize
            I = torch.eye(Nt, device=attn.device).unsqueeze(0)
            a = attn + I
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            rollout = a if rollout is None else a @ rollout

        timm_model = _unwrap_timm(self.model)
        num_prefix = getattr(timm_model, "num_prefix_tokens", 1)
        cls_attn = rollout[0, 0, num_prefix:]                        # (num_patches,)

        H, W = x.shape[-2:]
        ps = _get_patch_size(timm_model)
        Hp, Wp = H // ps, W // ps
        sal = cls_attn.reshape(Hp, Wp)
        sal = F.interpolate(
            sal[None, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
        return _normalize(sal.cpu().numpy())


class AttentionLastLayerExplainer:
    """CLS-to-patch attention from the final transformer block."""

    def __init__(self, model, head_fusion="mean"):
        if head_fusion not in ("mean", "max"):
            raise ValueError(f"head_fusion must be mean|max, got {head_fusion!r}")
        self.model = model
        self.head_fusion = head_fusion
        self.attentions = {}
        timm_model = _unwrap_timm(model)
        attn = timm_model.blocks[-1].attn
        attn.fused_attn = False
        attn._capture_idx = "last"
        attn._capture_dict = self.attentions
        attn.forward = types.MethodType(_capturing_attn_forward, attn)

    def explain(self, x, target_class):
        self.attentions.clear()
        with torch.no_grad():
            _ = self.model(x)
        attn = self.attentions["last"]                               # (1, H, N, N)
        attn = attn.mean(dim=1) if self.head_fusion == "mean" else attn.max(dim=1)[0]

        timm_model = _unwrap_timm(self.model)
        num_prefix = getattr(timm_model, "num_prefix_tokens", 1)
        cls_attn = attn[0, 0, num_prefix:]                           # (num_patches,)

        H, W = x.shape[-2:]
        ps = _get_patch_size(timm_model)
        Hp, Wp = H // ps, W // ps
        sal = cls_attn.reshape(Hp, Wp)
        sal = F.interpolate(
            sal[None, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
        return _normalize(sal.cpu().numpy())


def build_explainer(model, cfg):
    method = cfg.explainability.method
    raw = cfg.explainability.get("other_params", None)
    p = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    if method == "smoothgrad":
        return SmoothGradExplainer(
            model,
            n_samples=p.get("n_samples", 25),
            noise_std=p.get("noise_std", 0.15),
        )
    if method == "attention_rollout":
        return AttentionRolloutExplainer(
            model,
            discard_ratio=p.get("discard_ratio", 0.9),
            head_fusion=p.get("head_fusion", "mean"),
        )
    if method == "attention_last_layer":
        return AttentionLastLayerExplainer(
            model,
            head_fusion=p.get("head_fusion", "mean"),
        )
    raise ValueError(
        f"Unknown explainability method: {method!r}. "
        "Choose smoothgrad | attention_rollout | attention_last_layer."
    )


# ---------------------------------------------------------------------------
# Image / saving helpers
# ---------------------------------------------------------------------------

def denormalize(x, mean, std):
    """(1, C, H, W) normalized tensor -> (H, W, 3) uint8 RGB."""
    m = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    img = (x * s + m).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def saliency_to_heatmap(saliency, cmap_name):
    cmap_fn = matplotlib.colormaps[cmap_name]
    rgba = cmap_fn(saliency)
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay(original, heatmap, alpha):
    return (
        alpha * heatmap.astype(np.float32) + (1 - alpha) * original.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)


def save_three_panel(out_path, original, heatmap, combined, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, img, title in zip(
        axes, [original, heatmap, combined], ["Original", "Heatmap", "Combined"]
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    assert list(cfg.augmentation.image_size) == list(cfg.model.input_size), (
        f"Mismatch: augmentation.image_size={list(cfg.augmentation.image_size)} "
        f"!= model.input_size={list(cfg.model.input_size)}"
    )

    result_path = Path(cfg.paths.result_root_path) / cfg.paths.result_name
    subdir = cfg.paths.get("explainability", "explainability")
    output_dir = result_path / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # --- Dataset (test split, fall back to val if test is empty) ----------
    dataset_module = importlib.import_module(cfg.dataset.module)
    dataset_test = dataset_module.build_dataset(is_train="test", cfg=cfg)
    if len(dataset_test) == 0:
        print("WARNING: test split empty - falling back to val split.")
        dataset_test = dataset_module.build_dataset(is_train="val", cfg=cfg)
    print(f"Test dataset size: {len(dataset_test)}")

    # --- Model + checkpoint ----------------------------------------------
    model_module = importlib.import_module(cfg.model.module)
    model = model_module.create_model(cfg)
    print(f"Loading checkpoint: {cfg.paths.ckpt_path}")
    ckpt = torch.load(cfg.paths.ckpt_path, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f"Load message: {msg}")
    model.to(device)
    model.eval()

    # --- Explainer + render config ----------------------------------------
    method = cfg.explainability.method
    alpha = float(cfg.explainability.get("alpha", 0.5))
    cmap_name = cfg.explainability.get("cmap", "jet")
    explain_class_mode = cfg.explainability.get("explain_class", "prediction")
    if explain_class_mode not in ("prediction", "ground_truth"):
        raise ValueError(
            f"explain_class must be 'prediction' or 'ground_truth', "
            f"got {explain_class_mode!r}"
        )
    mean = list(cfg.augmentation.mean)
    std = list(cfg.augmentation.std)
    explainer = build_explainer(model, cfg)
    print(
        f"method={method}  alpha={alpha}  cmap={cmap_name}  "
        f"explain_class={explain_class_mode}"
    )

    # --- Loop -------------------------------------------------------------
    n_skipped = 0
    for idx in tqdm(range(len(dataset_test)), desc="Explaining"):
        img_tensor, img_path, label = dataset_test[idx]
        # Single-eye dataset appends "::R"/"::L"; Path.stem would otherwise
        # treat ".jpg::R" as one suffix and collide both eyes onto one file.
        if "::" in img_path:
            base, eye = img_path.rsplit("::", 1)
            out_path = output_dir / f"{Path(base).stem}_{eye}.png"
        else:
            out_path = output_dir / (Path(img_path).stem + ".png")
        if out_path.exists():
            n_skipped += 1
            continue

        x = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)[0]
            pred = int(probs.argmax().item())

        gt = int(label)
        target = pred if explain_class_mode == "prediction" else gt

        saliency = explainer.explain(x, target)
        original = denormalize(x, mean, std)
        heatmap = saliency_to_heatmap(saliency, cmap_name)
        combined = overlay(original, heatmap, alpha)

        suptitle = (
            f"{Path(img_path).name}   "
            f"pred={pred} (p={probs[pred].item():.3f})   "
            f"gt={gt}   "
            f"explained={target}   method={method}"
        )
        save_three_panel(out_path, original, heatmap, combined, suptitle)

    print(f"Done. Saved to: {output_dir}  (skipped {n_skipped} pre-existing)")


if __name__ == "__main__":
    cfg = parse_args_and_config()
    main(cfg)
