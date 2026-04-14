import math

import torch
import torch.nn.functional as F


class LastBlockRawAttention:
    """
    Hook helper that captures raw attention weights from the last transformer block.
    Expected shape per batch element: [heads, N, N].
    """

    def __init__(self, model):
        self.model = model
        self.attention = None
        self._hook = None
        self._register_hooks()

    def _register_hooks(self):
        if not hasattr(self.model, "blocks") or len(self.model.blocks) == 0:
            raise RuntimeError("Model does not expose transformer blocks.")
        self._hook = self.model.blocks[-1].attn.register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module, _input, output):
        # Some implementations return (x, attn), others only attn.
        if isinstance(output, tuple):
            attn = output[1] if len(output) > 1 else output[0]
        else:
            attn = output
        if torch.is_tensor(attn):
            self.attention = attn.detach()

    def __call__(self, input_tensor):
        self.attention = None
        with torch.no_grad():
            _ = self.model(input_tensor)
        if self.attention is None:
            raise RuntimeError("Attention hook did not capture any tensor.")
        return self.attention

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def _to_matrix(attn):
    """
    Convert attention to [B, N, N].
    Accepts [N, N], [B, N, N], [B, H, N, N].
    """
    if attn.dim() == 2:
        attn = attn.unsqueeze(0)
    elif attn.dim() == 4:
        attn = attn.mean(dim=1)
    if attn.dim() != 3:
        raise ValueError(f"Unsupported attention shape: {tuple(attn.shape)}")
    return attn


def extract_shared_structure_batch(attn_a, attn_b, out_size=None, eps=1e-12):
    """
    Shared-structure extraction with fused attention Laplacian (Fiedler map).

    Args:
        attn_a, attn_b:
            Attention matrices shaped [B, N, N] or [B, H, N, N] or [N, N].
            N can be either pure patch tokens (N = P^2) or include CLS (N = P^2 + 1).
        out_size:
            Optional output size (H, W). If provided, map is resized by bilinear interpolation.
        eps:
            Numerical stability epsilon.

    Returns:
        Tensor [B, 1, H, W] if out_size is set; otherwise [B, 1, P, P].
    """
    attn_a = _to_matrix(attn_a).float()
    attn_b = _to_matrix(attn_b).float()
    if attn_a.shape != attn_b.shape:
        raise ValueError(f"Shape mismatch: {tuple(attn_a.shape)} vs {tuple(attn_b.shape)}")

    bsz, n_tokens, _ = attn_a.shape
    grid_no_cls = int(math.sqrt(n_tokens))
    grid_with_cls = int(math.sqrt(max(0, n_tokens - 1)))

    if grid_no_cls * grid_no_cls == n_tokens:
        patch_tokens = n_tokens
        a_patch = attn_a
        b_patch = attn_b
    elif grid_with_cls * grid_with_cls == (n_tokens - 1):
        patch_tokens = n_tokens - 1
        # Remove CLS and keep only patch-to-patch attention.
        a_patch = attn_a[:, 1:, 1:]
        b_patch = attn_b[:, 1:, 1:]
    else:
        raise ValueError(
            f"Token count {n_tokens} is not compatible with square patch grid (with or without CLS)."
        )

    grid = int(math.sqrt(patch_tokens))

    # Symmetrize patch attention.
    a_sym = 0.5 * (a_patch + a_patch.transpose(-1, -2))
    b_sym = 0.5 * (b_patch + b_patch.transpose(-1, -2))

    # Geometric fusion.
    shared = torch.sqrt(torch.clamp(a_sym, min=0.0) * torch.clamp(b_sym, min=0.0) + eps)
    shared = shared - torch.diag_embed(torch.diagonal(shared, dim1=-2, dim2=-1))

    # Normalized graph Laplacian.
    degree = shared.sum(dim=-1).clamp_min(eps)
    d_inv_sqrt = degree.pow(-0.5)
    d_mat = torch.diag_embed(d_inv_sqrt)
    eye = torch.eye(patch_tokens, device=shared.device, dtype=shared.dtype).unsqueeze(0).expand(bsz, -1, -1)
    lap = eye - torch.bmm(d_mat, torch.bmm(shared, d_mat))

    # Fiedler vector = 2nd smallest eigenvector.
    _, eigenvectors = torch.linalg.eigh(lap)
    fiedler = eigenvectors[:, :, 1].reshape(bsz, 1, grid, grid)

    if out_size is not None and tuple(out_size) != (grid, grid):
        fiedler = F.interpolate(fiedler, size=out_size, mode="bilinear", align_corners=False)

    # Normalize to [-1, 1] per sample for stable downstream usage.
    fiedler = fiedler - fiedler.mean(dim=(2, 3), keepdim=True)
    fiedler = fiedler / fiedler.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return fiedler.clamp(-1.0, 1.0)
