from collections import defaultdict, OrderedDict
from logging import getLogger
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .xformer_embeddingbag_grad_wrapper import xformers_embedding_bag


logger = getLogger()


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class FastWeightProductKeyMemory(nn.Module):
    def __init__(
        self,
        # PKM parameters
        mem_k_dim=512,  # Memory keys dimension
        mem_v_dim=512,  # Memory values dimension
        mem_heads=1,  # Number of memory reading heads
        mem_topk=8,  # Number of memory slots to read / update per head
        mem_n_subkeys=512,  # Number of keys per subspace (total memory size will be mem_n_subkeys^2)
        # target value
        lookahead=1,
        # score
        qk_score_type="idw",  # "dot_product" | "idw"
        score_nonlinear="softmax",  # "softmax" | "silu" | "relu"
        score_temperature=1.0,
        # optim
        optimizer_type="sgd",
        learning_rate=1.0,
        weight_decay=0.0,
        loss_type="mse",
        mem_grad_to_values_only=True,
        grad_clip=False,
        addr_loss="me",  # None | "me"
        addr_loss_weight=1.0,
        # misc
        fp32_fw=True,
    ):
        # Check parameters
        # even number of key dimensions for product quantization
        assert mem_k_dim >= 2
        assert mem_k_dim % 2 == 0

        # initialize
        super().__init__()

        # PKM parameters
        self.subsize = mem_n_subkeys
        self.size = mem_n_subkeys**2
        self.k_dim = mem_k_dim
        self.v_dim = mem_v_dim
        self.heads = mem_heads
        self.topk = mem_topk
        # target value
        self.lookahead = lookahead
        # score
        self.qk_score_type = qk_score_type
        self.score_nonlinear = score_nonlinear
        self.score_temperature = score_temperature
        # optim
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.mem_grad_to_values_only = mem_grad_to_values_only
        self.grad_clip = grad_clip
        self.addr_loss = addr_loss
        self.addr_loss_weight = addr_loss_weight
        # misc
        self.fp32_fw = fp32_fw

        # For observing memory content
        self.ob_mode = None
        self.ob_idx2log = defaultdict(list)

        # Fast-weight parameters
        self.keys = nn.Parameter(torch.zeros(2 * self.heads * self.subsize, self.k_dim // 2))
        self.values = nn.Parameter(torch.zeros(self.size, self.v_dim))

    def reset_parameters(self):
        # Keys
        bound = 1 / math.sqrt(self.k_dim)
        nn.init.uniform_(self.keys, a=-bound, b=bound)
        # Values
        nn.init.normal_(self.values, mean=0, std=self.v_dim**-0.5)

    def get_fw_named_params(self):
        fw_named_params = OrderedDict()
        for n, p in self.named_parameters():
            if self.fp32_fw:
                p = p.float()
            fw_named_params[n] = p
        return fw_named_params

    def score_transform(self, scores):
        scores = scores / self.score_temperature
        if self.score_nonlinear == "softmax":
            normed_scores = F.softmax(scores, dim=-1).type_as(scores)
        elif self.score_nonlinear == "silu":
            normed_scores = F.silu(scores).type_as(scores)
        elif self.score_nonlinear == "relu":
            normed_scores = F.relu(scores).type_as(scores)
        else:
            raise NotImplementedError(f"Non-linear {self.score_nonlinear} not implemented for scores.")
        return normed_scores

    def compute_marginal_entropy(self, M, epsilon=1e-8):
        """
        M: Real-valued score matrix of size [N, K] (Logits)
        epsilon: Small value to prevent log(0)
        """
        input_dtype = M.dtype

        if self.addr_loss_weight == 0.0:
            marginal_entropy = torch.tensor(0.0, device=M.device, dtype=input_dtype)
        else:
            # Keep only topk highest scores per row
            topk_values, _ = torch.topk(M, self.topk, dim=1)
            threshold = topk_values[:, -1].unsqueeze(1)
            M = torch.where(M >= threshold, M, torch.full_like(M, float("-inf")))

            M = M.float()

            # Convert scores (logits) to probabilities (Softmax)
            # P_ij represents probability of item i belonging to class j
            P = F.softmax(M, dim=1)

            # Calculate Marginal Distribution
            # P_bar_j is the average probability of class j across the batch

            # Calculate Marginal Entropy (Maximize this -> Minimize negative)
            # We want the global distribution to be uniform (high entropy)
            # H(Y) = - sum(P_bar * log(P_bar))
            P_bar = torch.mean(P, dim=0)
            marginal_entropy = -torch.sum(P_bar * torch.log(P_bar + epsilon))
            marginal_entropy = marginal_entropy.to(input_dtype)
        return marginal_entropy

    def retrieve_values(
        self,
        query: torch.Tensor,
        fw_named_params: OrderedDict,
    ):
        """
        Retrieve values from current fast weight memory given queries.
        Args:
            query: [B, T, input_dim] Query tensor.
            fw_named_params: OrderedDict of fast weight parameters.
        Returns:
            A dict.
                retireved_values: [B * T, output_dim] Retrieved values.
                scores: [B * T, heads, topk] TopK Query-key scores.
                indices: [B * T, heads, topk] Retrieved topK key indices.
                sub_indices1: [B * T, heads, topk] TopK sub-key indices part 1.
                sub_indices2: [B * T, heads, topk] TopK sub-key indices part 2.
                all_sub_scores1: [B * T, heads, subsize] Sub-key scores part 1.
                all_sub_scores2: [B * T, heads, subsize] Sub-key scores part 2.
        """
        B, T = query.shape[:2]

        # Get indices and scores
        # All [B * T, heads, *]
        scores, indices, indices1, indices2, all_scores1, all_scores2 = self.get_indices(
            query.view(B * T, self.heads, self.k_dim), fw_named_params["keys"]
        )

        # Normalize score on the heads+topK dimension
        # [B * T, heads * topk]
        normed_scores = self.score_transform(scores.view(B * T, self.heads * self.topk))

        # Get values
        # [B * T, v_dim]
        vals = xformers_embedding_bag(
            indices=indices.view(B * T, self.heads * self.topk),  # sum over heads+topK dimension
            weight=fw_named_params["values"],
            per_sample_weights=normed_scores,
        )

        return {
            "retireved_values": vals,
            "scores": scores,
            "normed_scores": normed_scores,
            "indices": indices,
            "sub_indices1": indices1,
            "sub_indices2": indices2,
            "all_sub_scores1": all_scores1,
            "all_sub_scores2": all_scores2,
        }

    def compute_mem_loss(
        self,
        hyp_values: torch.Tensor,
        ref_values: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
    ):
        """
        Compute memorization loss from hyp_values and ref_values.
        Args:
            hyp_values: [BT, v_dim] Retrieved values from memory.
            ref_values: [BT, v_dim] Reference target values.
            loss_mask: [BT] Optional mask for memorization loss.
            loss_weights: [BT] Optional weights for memorization loss.
        Returns:
            mem_losses: [BT] Memorization losses per example.
        """
        BT, D = hyp_values.shape

        # Compute memorization loss
        if self.loss_type == "mse":
            mem_losses = F.mse_loss(hyp_values, ref_values, reduction="none").mean(dim=-1)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        # Apply loss mask and weights
        if loss_mask is not None:
            mem_losses = mem_losses * loss_mask
        if loss_weights is not None:
            mem_losses = mem_losses * loss_weights

        return mem_losses

    def compute_addr_loss(
        self,
        sub_indices1: torch.Tensor,
        sub_indices2: torch.Tensor,
        all_sub_scores1: torch.Tensor,
        all_sub_scores2: torch.Tensor,
    ):
        """
        Compute addressing loss from key indices and scores.
        Args:
            sub_indices1: [BT, heads, topk] Sub-key indices part 1.
            sub_indices2: [BT, heads, topk] Sub-key indices part 2.
            all_sub_scores1: [BT, heads, subsize] Sub-key scores part 1.
            all_sub_scores2: [BT, heads, subsize] Sub-key scores part 2.
        Returns:
            addr_loss: Scalar addressing loss.
        """
        sub_indices1 = sub_indices1.view(-1, self.topk)  # [BT * heads, topk]
        sub_indices2 = sub_indices2.view(-1, self.topk)  # [BT * heads, topk]
        all_sub_scores1 = all_sub_scores1.view(-1, self.subsize)  # [BT * heads, subsize]
        all_sub_scores2 = all_sub_scores2.view(-1, self.subsize)  # [BT * heads, subsize]

        if self.addr_loss is not None:
            if self.addr_loss == "me":
                addr_loss1 = -1.0 * self.compute_marginal_entropy(all_sub_scores1)
                addr_loss2 = -1.0 * self.compute_marginal_entropy(all_sub_scores2)
                addr_loss = self.addr_loss_weight * (addr_loss1 + addr_loss2)
            else:
                raise NotImplementedError(f"Addressing loss {self.addr_loss} not implemented.")
        else:
            addr_loss = None
        return addr_loss

    def update_fw_param(self, p, g, lr, wd):
        if self.optimizer_type == "sgd":
            dp = lr * (g + wd * p)
            updated_p = p - dp
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_type} not implemented.")
        return updated_p

    def update_fw(
        self,
        fw_named_params: OrderedDict,
        num_samples: int,
        indices: torch.Tensor,
        mem_loss: torch.Tensor,
        addr_loss: Optional[torch.Tensor] = None,
    ):
        """
        Update fast weight parameters using memorization loss and addressing loss.
        Args:
            fw_named_params: OrderedDict of current fast weight parameters.
            num_samples: Number of samples used in the current update.
            indices: Retrieved 1D key indices for the current batch.
            mem_loss: Scalar memorization loss.
            addr_loss: Optional scalar addressing loss.
        Returns:
            A dict.
                updated_fw_named_params: OrderedDict of updated fast weight parameters.
                grads: List of unclipped gradients for each fast weight parameter.
        """
        # Adjust mem loss
        mem_loss = mem_loss * 0.5

        if self.mem_grad_to_values_only:
            if addr_loss is not None:
                addr_grads = torch.autograd.grad(
                    addr_loss,
                    fw_named_params.values(),
                    create_graph=self.training,
                    allow_unused=True,
                )
            else:
                addr_grads = [None] * len(fw_named_params)
            mem_grads = torch.autograd.grad(
                mem_loss,
                fw_named_params["values"],
                create_graph=self.training,
            )
            grads = []
            for i, n in enumerate(fw_named_params.keys()):
                if n == "values":
                    grads.append(mem_grads[0])
                else:
                    grads.append(addr_grads[i])
        else:
            # Total loss
            loss = mem_loss
            if addr_loss is not None:
                loss = loss + addr_loss

            # Grads
            fw_params = tuple(fw_named_params.values())
            grads = torch.autograd.grad(
                loss,
                fw_params,
                create_graph=self.training,
            )

        # Adjust grad scale for values
        scaled_grads = []
        for n, g in zip(fw_named_params.keys(), grads):
            if n == "values":
                # Summing instead of averaging over samples and features
                scale = num_samples * self.v_dim
                # Scale by top-k and heads
                scale = scale * self.topk * self.heads
                g = g * scale
                # Scale by slot accesses
                idx_binc = torch.bincount(indices.view(-1), minlength=self.size)
                binc_scale = idx_binc.to(g.dtype)
                binc_scale[binc_scale == 0] = 1.0
                g = g / binc_scale.unsqueeze(-1)
                scaled_grads.append(g)
            else:
                scaled_grads.append(g)
        grads = scaled_grads

        # Grad clipping
        if self.grad_clip:
            clipped_grads = [torch.clamp(g, min=-1.0, max=1.0) for g in grads]
        else:
            clipped_grads = grads

        # Apply update
        updated_params = []
        fw_params = tuple(fw_named_params.values())
        for i, n in enumerate(fw_named_params.keys()):
            updated_params.append(
                self.update_fw_param(fw_params[i], clipped_grads[i], self.learning_rate, self.weight_decay)
            )
        updated_fw_named_params = OrderedDict({n: updated_params[i] for i, n in enumerate(fw_named_params.keys())})

        return updated_fw_named_params, grads

    def write_ob_log(self, token_ids, indices, scores):
        """
        Record each slot's (q_token_id, v_token_id, v_prev_ctx_token_ids, v_next_ctx_token_ids, score)
        token_ids: [B, T]
        indices: [B, T, heads*topk]
        scores: [B, T, heads*topk]
        """
        assert self.ob_mode

        B, T = token_ids.shape
        T_mem = T - self.lookahead
        token_ids = token_ids.tolist()
        for b in range(B):
            for t in range(T_mem):
                q_token_id = token_ids[b][t]
                v_token_id = token_ids[b][t + self.lookahead]
                v_ctx_begin = max(0, t + self.lookahead - 10)
                v_ctx_end = min(T, t + self.lookahead + 10)
                v_prev_ctx_token_ids = token_ids[b][v_ctx_begin : t + self.lookahead]
                v_next_ctx_token_ids = token_ids[b][t + self.lookahead + 1 : v_ctx_end]
                for htk in range(self.heads * self.topk):
                    score = scores[b, t, htk].item()
                    idx = indices[b, t, htk].item()
                    log_entry = (q_token_id, v_token_id, v_prev_ctx_token_ids, v_next_ctx_token_ids, score)
                    self.ob_idx2log[idx].append(log_entry)

    def forward_wo_past(
        self,
        q,
        ref_v,
        chunk_size,
        gates=None,
        loss_mask=None,
        token_ids=None,
    ):
        B, T = q.shape[:2]

        # Dtype conversion
        input_dtype = q.dtype
        if self.fp32_fw:
            q = q.float()
            ref_v = ref_v.float()

        # Compute chunk boundaries
        chunk_boundaries = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_boundaries.append((start, end))

        all_grads = []
        output = []
        losses = []
        addr_losses = [] if self.addr_loss is not None else None
        indices = []
        scores = []
        with torch.enable_grad():
            # Initial fast weight parameters
            fw_named_params_list = []
            fw_named_params_list.append(self.get_fw_named_params())

            for chunk_start, chunk_end in chunk_boundaries:
                # Get chunk inputs/outputs
                chunk_T = chunk_end - chunk_start
                # [B, chunk_T, in_dim]
                chunk_q = q[:, chunk_start:chunk_end, :].contiguous()
                # [B, chunk_T, out_dim]
                chunk_ref_v = ref_v[:, chunk_start:chunk_end, :].contiguous()
                # [B, chunk_T]
                chunk_gates = gates[:, chunk_start:chunk_end].contiguous() if gates is not None else None
                # [B, chunk_T]
                chunk_loss_mask = loss_mask[:, chunk_start:chunk_end].contiguous() if loss_mask is not None else None
                # [B, chunk_T]
                chunk_token_ids = token_ids[:, chunk_start:chunk_end].contiguous() if token_ids is not None else None

                # Latest fast weights
                cur_fw_named_params = fw_named_params_list[-1]

                ####### Forward #######
                # All [B * chunk_T, *]
                retrieval_output_dict = self.retrieve_values(chunk_q, cur_fw_named_params)
                chunk_hyp_v = retrieval_output_dict["retireved_values"]
                chunk_indices = retrieval_output_dict["indices"]
                chunk_scores = retrieval_output_dict["normed_scores"]
                chunk_sub_indices1 = retrieval_output_dict["sub_indices1"]
                chunk_sub_indices2 = retrieval_output_dict["sub_indices2"]
                chunk_all_sub_scores1 = retrieval_output_dict["all_sub_scores1"]
                chunk_all_sub_scores2 = retrieval_output_dict["all_sub_scores2"]
                # Take out chunk dim [B, chunk_T, v_dim]
                chunk_hyp_v = chunk_hyp_v.view(B, chunk_T, -1)

                ####### Forward outputs #######
                output.append(chunk_hyp_v)  # [B, chunk_T, out_dim]
                indices.append(chunk_indices.view(B, chunk_T, -1))  # [B, chunk_T, heads * topk]
                scores.append(chunk_scores.view(B, chunk_T, -1))  # [B, chunk_T, heads * topk]

                ###### Construct FW update inputs #######
                # Use local chunk instead of queue to ignore corner cases that cause grad issues
                update_end_idx = None if self.lookahead == 0 else -self.lookahead
                update_hyp = chunk_hyp_v[:, :update_end_idx]
                update_weights = chunk_gates[:, :update_end_idx] if chunk_gates is not None else None
                update_mask = chunk_loss_mask[:, :update_end_idx] if chunk_loss_mask is not None else None
                update_ref_src = chunk_ref_v

                # Special process of ref_v
                if self.lookahead == 0:
                    update_ref = update_ref_src
                else:
                    update_ref = []
                    for i in range(self.lookahead + 1):
                        lookahead_idx = (i, -(self.lookahead - i)) if (self.lookahead - i) > 0 else (i, None)
                        update_ref.append(update_ref_src[:, lookahead_idx[0] : lookahead_idx[1]])
                    update_ref = update_ref[-1]

                # Contiguous
                update_hyp = update_hyp.contiguous()
                update_ref = update_ref.contiguous()
                update_weights = update_weights.contiguous() if update_weights is not None else None
                update_mask = update_mask.contiguous() if update_mask is not None else None

                ###### FW update #######
                # Compute losses
                mem_losses = self.compute_mem_loss(
                    hyp_values=update_hyp.view(-1, update_hyp.size(-1)),
                    ref_values=update_ref.view(-1, update_ref.size(-1)),
                    loss_mask=update_mask.view(-1) if update_mask is not None else None,
                    loss_weights=update_weights.view(-1) if update_weights is not None else None,
                )
                addr_loss = self.compute_addr_loss(
                    sub_indices1=chunk_sub_indices1,
                    sub_indices2=chunk_sub_indices2,
                    all_sub_scores1=chunk_all_sub_scores1,
                    all_sub_scores2=chunk_all_sub_scores2,
                )
                num_samples = (update_mask > 0).sum().item() if update_mask is not None else B * chunk_T
                updated_fw_named_params, grads = self.update_fw(
                    fw_named_params=cur_fw_named_params,
                    num_samples=num_samples,
                    indices=chunk_indices,
                    mem_loss=mem_losses.mean(),
                    addr_loss=addr_loss,
                )
                if self.ob_mode:
                    self.write_ob_log(
                        token_ids=chunk_token_ids.view(B, chunk_T),
                        indices=chunk_indices.view(B, chunk_T, -1),
                        scores=chunk_gates.view(B, chunk_T, -1),
                    )

                # Backward outputs
                all_grads.append([g.detach() for g in grads])
                fw_named_params_list.append(updated_fw_named_params)
                losses.append(mem_losses.view(B, -1).detach())  # [B, chunk_T-lookahead]
                if self.addr_loss is not None:
                    addr_loss = addr_loss.detach()
                    addr_losses.append(addr_loss)

        output = torch.cat(output, dim=1)  # [B, T, out_dim]
        indices = torch.cat(indices, dim=1).view(B, T, self.heads, self.topk)  # [B, T, heads, topk]
        scores = torch.cat(scores, dim=1).view(B, T, self.heads, self.topk)  # [B, T, heads, topk]

        # Losses
        losses = torch.cat(losses, dim=1)  # [B, T]
        addr_loss = torch.stack(addr_losses).mean() if self.addr_loss is not None else None

        # Grad norms
        with torch.no_grad():
            grad_norms = []
            for i in range(len(fw_named_params_list[0])):
                grad_norm = torch.stack([g[i].norm() for g in all_grads]).mean()
                grad_norms.append(grad_norm)

        # Update fast weight variables with the latest params, in original dtype
        with torch.no_grad():
            final_fw_named_params = fw_named_params_list[-1]
            for n, p in self.named_parameters():
                if n in final_fw_named_params:
                    p.data.copy_(final_fw_named_params[n].data.to(p.dtype))

        # Dtype reversion
        output = output.to(input_dtype)

        return {
            "output": output,
            "indices": indices,
            "scores": scores,
            "losses": losses,
            "addr_loss": addr_loss if self.addr_loss is not None else None,
            "grad_norms": grad_norms,
            "past_key_values": None,
        }

    def forward_w_past(
        self,
        q,
        ref_v,
        gates,
        chunk_size,
        loss_mask=None,
        past_key_values=None,
        token_ids=None,
    ):
        B, T = q.shape[:2]

        # Dtype conversion
        input_dtype = q.dtype
        if self.fp32_fw:
            q = q.float()
            ref_v = ref_v.float()

        # Queue of signals for updating FW
        q_queue = torch.zeros(B, 0, q.size(-1), device=ref_v.device, dtype=ref_v.dtype)
        hyp_v_queue = torch.zeros(B, 0, ref_v.size(-1), device=ref_v.device, dtype=ref_v.dtype)
        ref_v_queue = torch.zeros(B, 0, ref_v.size(-1), device=ref_v.device, dtype=ref_v.dtype)
        gates_queue = torch.zeros(B, 0, device=ref_v.device, dtype=ref_v.dtype) if gates is not None else None
        mask_queue = torch.zeros(B, 0, device=ref_v.device, dtype=ref_v.dtype) if loss_mask is not None else None
        ids_queue = torch.zeros(B, 0, device=ref_v.device, dtype=torch.long) if token_ids is not None else None

        # Unprocessed info in past key values
        if past_key_values is not None:
            past_q, past_hyp_v, past_ref_v, past_gates, past_mask, past_ids = past_key_values
            if past_hyp_v is not None:
                past_T = past_hyp_v.size(1)
                assert past_T < chunk_size + self.lookahead, (
                    f"Past key values length {past_T} longer than "
                    f"chunk size + lookahead ({chunk_size} + {self.lookahead})."
                )
                q_queue = torch.cat([past_q, q_queue], dim=1)
                hyp_v_queue = torch.cat([past_hyp_v, hyp_v_queue], dim=1)
                ref_v_queue = torch.cat([past_ref_v, ref_v_queue], dim=1)
                gates_queue = torch.cat([past_gates, gates_queue], dim=1) if gates is not None else None
                mask_queue = torch.cat([past_mask, mask_queue], dim=1) if loss_mask is not None else None
                ids_queue = torch.cat([past_ids, ids_queue], dim=1) if token_ids is not None else None

        # Compute chunk boundaries
        chunk_boundaries = []
        # First chunk follows past key values
        first_end = min(chunk_size + self.lookahead - ref_v_queue.size(1), T)
        chunk_boundaries.append((0, first_end))
        # Remaining chunks
        for start in range(first_end, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_boundaries.append((start, end))

        # Update flags
        updated = False

        # Initial fast weight parameters
        cur_fw_named_params = self.get_fw_named_params()

        output = []
        losses = []
        addr_losses = [] if self.addr_loss is not None else None
        indices = []
        scores = []
        for chunk_start, chunk_end in chunk_boundaries:
            # Get chunk inputs/outputs
            chunk_T = chunk_end - chunk_start
            # [B, chunk_T, in_dim]
            chunk_q = q[:, chunk_start:chunk_end, :].contiguous()
            # [B, chunk_T, out_dim]
            chunk_ref_v = ref_v[:, chunk_start:chunk_end, :].contiguous()
            # [B, chunk_T]
            chunk_gates = gates[:, chunk_start:chunk_end].contiguous() if gates is not None else None
            # [B, chunk_T]
            chunk_loss_mask = loss_mask[:, chunk_start:chunk_end].contiguous() if loss_mask is not None else None
            # [B, chunk_T]
            chunk_token_ids = token_ids[:, chunk_start:chunk_end].contiguous() if token_ids is not None else None

            ####### Forward #######
            # All [B * chunk_T, *]
            retrieval_output_dict = self.retrieve_values(chunk_q, cur_fw_named_params)
            chunk_hyp_v = retrieval_output_dict["retireved_values"]
            chunk_scores = retrieval_output_dict["normed_scores"]
            chunk_indices = retrieval_output_dict["indices"]
            # Take out chunk dim [B, chunk_T, v_dim]
            chunk_hyp_v = chunk_hyp_v.view(B, chunk_T, -1)

            # Forward outputs
            output.append(chunk_hyp_v)  # [B, chunk_T, out_dim]
            indices.append(chunk_indices.view(B, chunk_T, -1))  # [B, chunk_T, heads * topk]
            scores.append(chunk_scores.view(B, chunk_T, -1))  # [B, chunk_T, heads * topk]

            ####### Update signal queues #######
            # Append to queues
            q_queue = torch.cat([q_queue, chunk_q], dim=1)
            hyp_v_queue = torch.cat([hyp_v_queue, chunk_hyp_v], dim=1)
            ref_v_queue = torch.cat([ref_v_queue, chunk_ref_v], dim=1)
            gates_queue = torch.cat([gates_queue, chunk_gates], dim=1) if gates is not None else None
            mask_queue = torch.cat([mask_queue, chunk_loss_mask], dim=1) if loss_mask is not None else None
            ids_queue = torch.cat([ids_queue, chunk_token_ids], dim=1) if token_ids is not None else None

            ####### Update FW only if enough signal is in the queue #######
            if past_key_values and hyp_v_queue.size(1) < chunk_size:
                continue

            updated = True
            # print(f"Updating FW at chunk {chunk_start}: {chunk_end} with queue size {hyp_v_queue.size()}")

            with torch.enable_grad():
                cur_fw_named_params = {n: p.detach() for n, p in cur_fw_named_params.items()}
                for p in cur_fw_named_params.values():
                    p.requires_grad = True

                ###### Forward with accumulated q #######
                queue_end_idx = None if self.lookahead == 0 else -self.lookahead
                update_q = q_queue[:, :queue_end_idx].contiguous()  # [B, queue_T - lookahead, in_dim]
                retrieval_output_dict = self.retrieve_values(update_q, cur_fw_named_params)
                chunk_hyp_v_bwd = retrieval_output_dict["retireved_values"]  # [B * (queue_T - lookahead), out_dim]
                chunk_scores_bwd = retrieval_output_dict["scores"]
                chunk_indices_bwd = retrieval_output_dict["indices"]
                chunk_sub_indices1_bwd = retrieval_output_dict["sub_indices1"]
                chunk_sub_indices2_bwd = retrieval_output_dict["sub_indices2"]
                chunk_all_sub_scores1_bwd = retrieval_output_dict["all_sub_scores1"]
                chunk_all_sub_scores2_bwd = retrieval_output_dict["all_sub_scores2"]

                ###### Construct FW update inputs from signal queues #######
                update_hyp = chunk_hyp_v_bwd  # [B, queue_T - lookahead, out_dim]
                update_weights = gates_queue[:, :queue_end_idx] if gates_queue is not None else None
                update_mask = mask_queue[:, :queue_end_idx] if mask_queue is not None else None
                # Construct lookahead target values from `ref_v_queue`
                # update_ref: [B, queue_T - lookahead, out_dim]
                if self.lookahead == 0:
                    update_ref = ref_v_queue
                else:
                    update_ref = []
                    for i in range(self.lookahead + 1):
                        lookahead_idx = (i, -(self.lookahead - i)) if (self.lookahead - i) > 0 else (i, None)
                        update_ref.append(ref_v_queue[:, lookahead_idx[0] : lookahead_idx[1]])
                    update_ref = update_ref[-1]
                update_hyp = update_hyp.contiguous()  # [B, queue_T - lookahead, out_dim]
                update_ref = update_ref.contiguous()  # [B, queue_T - lookahead, out_dim]
                update_weights = update_weights.contiguous() if update_weights is not None else None
                update_mask = update_mask.contiguous() if update_mask is not None else None

                ###### FW update #######
                # Compute losses
                mem_losses = self.compute_mem_loss(
                    hyp_values=update_hyp.view(-1, update_hyp.size(-1)),
                    ref_values=update_ref.view(-1, update_ref.size(-1)),
                    loss_mask=update_mask.view(-1) if update_mask is not None else None,
                    loss_weights=update_weights.view(-1) if update_weights is not None else None,
                )
                addr_loss = self.compute_addr_loss(
                    sub_indices1=chunk_sub_indices1_bwd,
                    sub_indices2=chunk_sub_indices2_bwd,
                    all_sub_scores1=chunk_all_sub_scores1_bwd,
                    all_sub_scores2=chunk_all_sub_scores2_bwd,
                )
                num_samples = (update_mask > 0).sum().item() if update_mask is not None else B * chunk_T

                cur_fw_named_params, grads = self.update_fw(
                    fw_named_params=cur_fw_named_params,
                    num_samples=num_samples,
                    indices=chunk_indices,
                    mem_loss=mem_losses.mean(),
                    addr_loss=addr_loss,
                )
                if self.ob_mode:
                    chunk_indices_bwd_padded = torch.cat(
                        [
                            chunk_indices_bwd.view(B, ids_queue.size(1) - self.lookahead, -1),
                            torch.zeros(
                                ids_queue.size(0),
                                self.lookahead,
                                self.heads * self.topk,
                                device=ids_queue.device,
                                dtype=chunk_indices_bwd.dtype,
                            ),
                        ],
                        dim=1,
                    )
                    chunk_scores_bwd_padded = torch.cat(
                        [
                            chunk_scores_bwd.view(B, ids_queue.size(1) - self.lookahead, -1),
                            torch.zeros(
                                ids_queue.size(0),
                                self.lookahead,
                                self.heads * self.topk,
                                device=ids_queue.device,
                                dtype=chunk_scores_bwd.dtype,
                            ),
                        ],
                        dim=1,
                    )
                    self.write_ob_log(
                        token_ids=ids_queue,
                        indices=chunk_indices_bwd_padded,
                        scores=chunk_scores_bwd_padded,
                    )

            ####### Pop out used signals from queues #######
            if self.lookahead == 0:
                q_queue = torch.zeros(B, 0, q.size(-1), device=ref_v.device, dtype=ref_v.dtype)
                hyp_v_queue = torch.zeros(B, 0, ref_v.size(-1), device=ref_v.device, dtype=ref_v.dtype)
                ref_v_queue = torch.zeros(B, 0, ref_v.size(-1), device=ref_v.device, dtype=ref_v.dtype)
                gates_queue = torch.zeros(B, 0, device=ref_v.device, dtype=ref_v.dtype) if gates is not None else None
                mask_queue = (
                    torch.zeros(B, 0, device=ref_v.device, dtype=ref_v.dtype) if loss_mask is not None else None
                )
                ids_queue = torch.zeros(B, 0, device=ref_v.device, dtype=torch.long)
            else:
                q_queue = q_queue[:, -self.lookahead :].contiguous()  # [B, lookahead, in_dim]
                hyp_v_queue = hyp_v_queue[:, -self.lookahead :].contiguous()  # [B, lookahead, out_dim]
                ref_v_queue = ref_v_queue[:, -self.lookahead :].contiguous()  # [B, lookahead, out_dim]
                gates_queue = gates_queue[:, -self.lookahead :].contiguous() if gates is not None else None
                mask_queue = mask_queue[:, -self.lookahead :].contiguous() if update_mask is not None else None
                ids_queue = ids_queue[:, -self.lookahead :].contiguous()  # [B, lookahead]

            ####### Backward outputs #######
            losses.append(mem_losses.view(B, -1).detach())  # [B, chunk_T-lookahead]
            if self.addr_loss is not None:
                addr_loss = addr_loss.detach()
                addr_losses.append(addr_loss)

        output = torch.cat(output, dim=1)  # [B, T, out_dim]
        indices = torch.cat(indices, dim=1).view(B, T, self.heads, self.topk)  # [B, T, heads, topk]
        scores = torch.cat(scores, dim=1).view(B, T, self.heads, self.topk)  # [B, T, heads, topk]

        # Update past_key_values
        past_key_values = (q_queue, hyp_v_queue, ref_v_queue, gates_queue, mask_queue, ids_queue)

        # Collect update statistics if update happened
        if updated:
            # Mem loss
            losses = torch.cat(losses, dim=1)  # [B, T]
            # Addr loss
            if self.addr_loss is not None:
                addr_loss = torch.stack(addr_losses).mean()
        else:
            losses = None
            addr_loss = None

        # Update fast weight variables with the latest params, in original dtype
        if updated:
            with torch.no_grad():
                final_fw_named_params = cur_fw_named_params
                for n, p in self.named_parameters():
                    if n in final_fw_named_params:
                        p.data.copy_(final_fw_named_params[n].data.to(p.dtype))

        # Dtype reversion
        output = output.to(input_dtype)

        return {
            "output": output,
            "indices": indices,
            "scores": scores,
            "losses": losses,
            "addr_loss": addr_loss if self.addr_loss is not None else None,
            "grad_norms": None,
            "past_key_values": past_key_values,
        }

    @torch.compiler.disable(recursive=True)
    def forward(
        self,
        q,
        ref_v,
        gates,
        chunk_size,
        loss_mask=None,
        past_key_values=None,
        token_ids=None,
    ):
        if past_key_values is None:
            return self.forward_wo_past(
                q=q,
                ref_v=ref_v,
                gates=gates,
                chunk_size=chunk_size,
                loss_mask=loss_mask,
                token_ids=token_ids,
            )
        else:
            return self.forward_w_past(
                q=q,
                ref_v=ref_v,
                gates=gates,
                chunk_size=chunk_size,
                loss_mask=loss_mask,
                past_key_values=past_key_values,
                token_ids=token_ids,
            )

    def compute_qk_scores(self, q, k):
        """Compute query-key scores.
        q: (batch_size, heads, k_dim)
        k: (heads, n_keys, k_dim)
        Returns:
            scores: (batch_size, heads, n_keys)
        """
        if self.qk_score_type == "dot_product":
            scores = torch.einsum("bhd,hnd->bhn", q, k)  # (batch_size, heads, n_keys)
        elif self.qk_score_type == "idw":
            q_perm = q.permute(1, 0, 2)  # (heads, batch_size, k_dim)
            dists = torch.cdist(q_perm, k, p=2, compute_mode="use_mm_for_euclid_dist")  # (heads, batch_size, n_keys)
            dists = dists.permute(1, 0, 2)  # (batch_size, heads, n_keys)
            scores = -torch.log(1e-3 + dists.pow(2))
        else:
            raise NotImplementedError(f"QK score type {self.qk_score_type} not implemented.")
        return scores

    def get_indices(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ):
        """
        Get top-k indices and scores from product keys.
        Args:
            query: (BT, heads, k_dim)
            keys: (heads * 2 * subsize, k_dim // 2)
        Returns:
            scores: (BT, heads, topk)
            indices: (BT, heads, topk)
            topk_indices1: (BT, heads, topk)
            topk_indices2: (BT, heads, topk)
            scores1: (BT, heads, subsize)
            scores2: (BT, heads, subsize)
        """
        topk = self.topk
        BT = query.size(0)
        query = query.view(BT, self.heads, self.k_dim)
        half = self.k_dim // 2
        # keys : (heads, 2, n_keys, half)
        # keys1 : (heads, n_keys, half)
        keys = keys.view(self.heads, 2, -1, half)
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]

        # split query for product quantization
        q1 = query[:, :, :half]  # (BT, heads, half)
        q2 = query[:, :, half:]  # (BT, heads, half)

        # compute indices with associated scores
        scores1 = self.compute_qk_scores(q1, keys1)  # (BT, heads, subsize)
        scores2 = self.compute_qk_scores(q2, keys2)  # (BT, heads, subsize)

        # select top-k
        topk_scores1, topk_indices1 = scores1.topk(topk, dim=2, largest=True)  # (BT, heads, topk)
        topk_scores2, topk_indices2 = scores2.topk(topk, dim=2, largest=True)  # (BT, heads, topk)

        # cartesian product on best candidate keys
        all_scores = (
            topk_scores1.view(BT, self.heads, topk, 1).expand(BT, self.heads, topk, topk)
            + topk_scores2.view(BT, self.heads, 1, topk).expand(BT, self.heads, topk, topk)
        ).view(BT, self.heads, topk**2)  # (BT, heads, topk ** 2)
        all_indices = (
            topk_indices1.view(BT, self.heads, topk, 1).expand(BT, self.heads, topk, topk) * self.subsize
            + topk_indices2.view(BT, self.heads, 1, topk).expand(BT, self.heads, topk, topk)
        ).view(BT, self.heads, topk**2)  # (BT, heads, topk ** 2)

        # select overall best scores and indices
        scores, best_indices = torch.topk(all_scores, k=topk, dim=2, largest=True, sorted=True)  # (BT, heads, topk)
        indices = all_indices.gather(2, best_indices)  # (BT, heads, topk)

        # return scores with indices
        assert scores.shape == indices.shape == (BT, self.heads, topk)
        return (
            scores.contiguous(),
            indices.contiguous(),
            topk_indices1.contiguous(),
            topk_indices2.contiguous(),
            scores1.contiguous(),
            scores2.contiguous(),
        )
