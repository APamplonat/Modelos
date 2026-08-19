from typing import Optional, Tuple, Callable
import torch
from torch.autograd import Function

from .xformer_embeddingbag_legacy import embedding_bag_bw_rev_indices


# -------------------------------------------------------
# 1) Double-backward wrapper around grad_weight
#    Exposes the VJP:
#       (grad_output, per_sample_weights, indices) -> grad_weight
# -------------------------------------------------------
class EmbeddingBagBW(Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,  # [B, bag], long
        per_sample_weights: Optional[torch.Tensor],  # [B, bag] (float) or None
        grad_output: torch.Tensor,  # [B, D]
        num_embeddings: int,  # K
        embedding_dim: int,  # D
        fast_grad_weight: torch.Tensor,  # [K, D], from fused kernel
    ) -> torch.Tensor:
        # Save inputs needed to express the VJP in backward
        if per_sample_weights is None:
            ctx.has_psw = False
            # save a placeholder to keep ordering consistent
            ctx.save_for_backward(
                indices, torch.empty(0, device=grad_output.device, dtype=grad_output.dtype), grad_output
            )
        else:
            ctx.has_psw = True
            ctx.save_for_backward(indices, per_sample_weights, grad_output)

        ctx.num_embeddings = num_embeddings
        ctx.embedding_dim = embedding_dim

        # Return the terminal gradient tensor (first-backward output).
        # The graph connection enabling double-backward is created by this Function node.
        return fast_grad_weight

    @staticmethod
    def backward(ctx, g_bar: torch.Tensor):
        """
        g_bar: upstream gradient w.r.t. output (i.e., dL/d(grad_weight)), shape [K, D]

        We produce gradients w.r.t. the *inputs* of forward():
          indices -> None (ints)
          per_sample_weights -> d_a [B, bag] or None
          grad_output -> d_grad_output [B, D]
          num_embeddings -> None
          embedding_dim -> None
          fast_grad_weight -> None
        """
        idx, a_saved, g = ctx.saved_tensors
        B, bag = idx.shape
        D = ctx.embedding_dim

        # Gather g_bar rows for each occurrence in idx: [B, bag, D]
        gathered = g_bar.index_select(0, idx.reshape(-1)).view(B, bag, D)

        if ctx.has_psw:
            a = a_saved
            # VJP formulas:
            # dL/d(grad_output_b) = sum_j a_bj * g_bar[idx_bj]
            d_grad_output = (a.unsqueeze(-1) * gathered).sum(dim=1)  # [B, D]
            # dL/d(a_bj) = <g_bar[idx_bj], grad_output_b>
            d_a = (gathered * g.unsqueeze(1)).sum(dim=-1)  # [B, bag]
        else:
            d_grad_output = gathered.sum(dim=1)  # [B, D]
            d_a = None

        # Return grads in the exact order of forward() inputs
        return (None, d_a, d_grad_output, None, None, None)


# -------------------------------------------------------
# 2) Main op: forward + backward
#    - Forward: sum_j a_bj * W[idx_bj]
#    - Backward: call your fast kernel to get grads,
#                then WRAP weight_grad via EmbeddingBagBW
# -------------------------------------------------------
class _XEmbeddingBagFn(Function):
    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,  # [K, D], learnable
        indices: torch.Tensor,  # [B, bag], long
        per_sample_weights: Optional[torch.Tensor],  # [B, bag] or None
        bw_kernel: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> torch.Tensor:
        assert indices.dtype == torch.long
        K, D = weight.shape
        B, bag = indices.shape

        # Gather rows and compute weighted sum per bag
        rows = weight.index_select(0, indices.reshape(-1)).view(B, bag, D)
        if per_sample_weights is None:
            out = rows.sum(dim=1)  # [B, D]
        else:
            out = (rows * per_sample_weights.unsqueeze(-1)).sum(dim=1)

        # Save everything we need for the first backward
        ctx.save_for_backward(
            weight,
            indices,
            per_sample_weights
            if per_sample_weights is not None
            else torch.empty(0, device=weight.device, dtype=weight.dtype),
        )
        ctx.has_psw = per_sample_weights is not None
        ctx.K, ctx.D = K, D
        ctx.bw_kernel = bw_kernel
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, indices, a_saved = ctx.saved_tensors
        K, D = ctx.K, ctx.D
        B, bag = indices.shape

        # Resolve optional per-sample weights
        a = a_saved if ctx.has_psw else None

        # ----- Call your fast fused backward kernel (first-order) -----
        # It returns (weight_grad, per_sample_weights_grad)
        # NOTE: This kernel is allowed to be in no_grad and non-differentiable.
        #       The double-backward path is provided by EmbeddingBagBW below.
        weight_grad_fast, psw_grad_fast = ctx.bw_kernel(indices, weight, a, grad_output)

        # ----- Wrap weight_grad to expose double-backward VJP -----
        grad_weight = EmbeddingBagBW.apply(
            indices,
            a if ctx.has_psw else None,
            grad_output,
            K,
            D,
            weight_grad_fast,
        )

        # Per-sample weights gradient (first-order). If you prefer the autograd
        # expression, you can compute:
        #   psw_grad = <W[idx], grad_output>
        # but since your kernel already returns it, just use it.
        grad_psw = psw_grad_fast if ctx.has_psw else None

        # No grad for indices (integer) and bw_kernel (callable)
        grad_indices = None
        grad_bw_kernel = None

        # Return grads for inputs of forward() in order:
        #   weight,       indices,     per_sample_weights,  bw_kernel
        return grad_weight, grad_indices, grad_psw, grad_bw_kernel


def xformers_embedding_bag(
    indices: torch.Tensor,
    weight: torch.Tensor,
    per_sample_weights: torch.Tensor,
) -> torch.Tensor:
    return _XEmbeddingBagFn.apply(weight, indices, per_sample_weights, embedding_bag_bw_rev_indices)
