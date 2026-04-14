import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoAttentionExtractor(nn.Module):
    """Frozen DINO wrapper that returns a CLS-to-patch attention map."""

    def __init__(self, model_name="dino_vitb8", image_size=224):
        super().__init__()
        self.image_size = int(image_size)
        self.model = torch.hub.load("facebookresearch/dino:main", model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self._last_attention = None
        self._hook = None
        self._register_hook()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

    def _register_hook(self):
        if not hasattr(self.model, "blocks") or len(self.model.blocks) == 0:
            raise RuntimeError("Loaded DINO model does not expose transformer blocks.")
        self._hook = self.model.blocks[-1].attn.register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module, _inputs, output):
        if isinstance(output, tuple):
            attn = output[1] if len(output) > 1 else output[0]
        else:
            attn = output
        if torch.is_tensor(attn):
            self._last_attention = attn

    def _preprocess(self, images):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = images.clamp(-1.0, 1.0)
        images = (images + 1.0) * 0.5
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (images - self.pixel_mean) / self.pixel_std

    def forward(self, images):
        self._last_attention = None
        proc = self._preprocess(images)
        _ = self.model(proc)
        if self._last_attention is None:
            raise RuntimeError("DINO attention hook did not capture any tensor.")

        attn = self._last_attention
        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        if attn.dim() != 3:
            raise RuntimeError(f"Unexpected DINO attention shape: {tuple(attn.shape)}")

        cls_to_patch = attn[:, 0, 1:]
        grid = int(math.sqrt(cls_to_patch.shape[-1]))
        if grid * grid != cls_to_patch.shape[-1]:
            raise RuntimeError(
                f"CLS-to-patch tokens ({cls_to_patch.shape[-1]}) do not form a square grid."
            )
        attn_map = cls_to_patch.view(images.shape[0], 1, grid, grid)
        attn_map = attn_map - attn_map.amin(dim=(2, 3), keepdim=True)
        attn_map = attn_map / attn_map.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return attn_map

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
