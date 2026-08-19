from collections import OrderedDict
from logging import getLogger
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

logger = getLogger()


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class FastWeightMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        size,
        # target
        lookahead=0,
        lookahead_strategy="mean",  # "mean" | "last"
        # optim
        optimizer_type="sgd",
        learning_rate=0.1,
        weight_decay=0.0,
        loss_type="mse",  # "mse" | "mae"
        grad_clip=True,
        # misc
        fp32_fw=True,
    ):
        # initialize
        super().__init__()

        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.size = size
        # target
        self.lookahead = lookahead
        self.lookahead_strategy = lookahead_strategy
        # optim
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.grad_clip = grad_clip
        # misc
        self.fp32_fw = fp32_fw

        # params
        self.w_up = nn.Parameter(torch.ones(self.size, self.input_dim))
        self.b_up = nn.Parameter(torch.zeros(self.size))
        self.w_gate = nn.Parameter(torch.ones(self.size, self.input_dim))
        self.b_gate = nn.Parameter(torch.zeros(self.size))
        self.w_down = nn.Parameter(torch.ones(self.output_dim, self.size))
        self.b_down = nn.Parameter(torch.zeros(self.output_dim))

    def reset_parameters(self):
        nn.init.normal_(self.w_up, mean=0, std=1 / math.sqrt(self.input_dim))
        nn.init.zeros_(self.b_up)
        nn.init.normal_(self.w_gate, mean=0, std=1 / math.sqrt(self.input_dim))
        nn.init.zeros_(self.b_gate)
        nn.init.normal_(self.w_down, mean=0, std=1 / math.sqrt(self.size))
        nn.init.zeros_(self.b_down)

    def get_fw_named_params(self):
        fw_named_params = OrderedDict()
        for n, p in self.named_parameters():
            if self.fp32_fw:
                p = p.float()
            fw_named_params[n] = p
        return fw_named_params

    def update_fw_param(self, p, g, lr, wd):
        if self.optimizer_type == "sgd":
            dp = lr * (g + wd * p)
            updated_p = p - dp
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_type} not implemented for fast weights.")
        return updated_p

    def compute_mem_loss(
        self,
        hyp_values: torch.Tensor,
        ref_values: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        """
        Compute memorization loss from hyp_values and ref_values.
        Args:
            hyp_values: [BT, v_dim] Retrieved values from memory.
            ref_values: [BT, v_dim] Reference target values.
            loss_mask: [BT] Optional mask for memorization loss.
        Returns:
            mem_losses: [BT] Memorization losses per example.
        """
        BT, D = hyp_values.shape

        # Compute memorization loss
        if self.loss_type == "mse":
            mem_losses = F.mse_loss(hyp_values, ref_values, reduction="none").mean(dim=-1)
        elif self.loss_type == "mae":
            mem_losses = (hyp_values - ref_values).abs().mean(dim=-1)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        # Apply loss mask
        if loss_mask is not None:
            mem_losses = mem_losses * loss_mask

        return mem_losses

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
        """
        B, T = query.shape[:2]
        query = query.view(B * T, self.input_dim)  # [B * T, input_dim]

        # Forward
        # up: [B * T, size]
        up = F.linear(query, fw_named_params["w_up"], fw_named_params["b_up"])
        # gate: [B * T, size]
        gate = ACT2FN["silu"](F.linear(query, fw_named_params["w_gate"], fw_named_params["b_gate"]))
        # vals: [B * T, size]
        vals = up * gate
        # out: [B * T, output_dim]
        out = F.linear(vals, fw_named_params["w_down"], fw_named_params["b_down"])

        return {
            "retireved_values": out,
        }

    def update_fw(
        self,
        fw_named_params: OrderedDict,
        loss: torch.Tensor,
    ):
        """
        Update fast weight parameters using memorization loss.
        Args:
            fw_named_params: OrderedDict of current fast weight parameters.
            loss: Scalar memorization loss.
        Returns:
            A dict.
                updated_fw_named_params: OrderedDict of updated fast weight parameters.
                grads: List of unclipped gradients for each fast weight parameter.
        """
        # Grads
        fw_params = tuple(fw_named_params.values())
        grads = torch.autograd.grad(
            loss,
            fw_params,
            create_graph=self.training,
        )

        # Grad clipping
        if self.grad_clip:
            clipped_grads = [torch.clamp(g, min=-1.0, max=1.0) for g in grads]
        else:
            clipped_grads = grads

        # Apply update
        updated_params = []
        fw_params = tuple(fw_named_params.values())
        for i, n in enumerate(fw_named_params.keys()):
            lr = self.learning_rate
            wd = self.weight_decay
            updated_params.append(self.update_fw_param(fw_params[i], clipped_grads[i], lr, wd))

        updated_fw_named_params = OrderedDict({n: updated_params[i] for i, n in enumerate(fw_named_params.keys())})

        return updated_fw_named_params, grads

    def forward_wo_past(
        self,
        q,
        ref_v,
        chunk_size,
        loss_mask=None,
    ):
        B, T = q.shape[:2]

        # Dtype conversion
        input_dtype = q.dtype
        if self.fp32_fw:
            q = q.float()
            ref_v = ref_v.float()

        # Queue of signals for updating FW
        # hyp_v_queue = torch.zeros(B, 0, ref_v.size(-1), device=ref_v.device, dtype=ref_v.dtype)
        # ref_v_queue = torch.zeros(B, 0, ref_v.size(-1), device=ref_v.device, dtype=ref_v.dtype)
        # mask_queue = torch.zeros(B, 0, device=ref_v.device, dtype=ref_v.dtype) if loss_mask is not None else None

        # Compute chunk boundaries
        chunk_boundaries = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_boundaries.append((start, end))

        all_grads = []
        output = []
        losses = []
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
                chunk_loss_mask = loss_mask[:, chunk_start:chunk_end].contiguous() if loss_mask is not None else None

                # Latest fast weights
                cur_fw_named_params = fw_named_params_list[-1]

                ####### Forward #######
                # All [B * chunk_T, *]
                retrieval_output_dict = self.retrieve_values(chunk_q, cur_fw_named_params)
                chunk_hyp_v = retrieval_output_dict["retireved_values"]
                # Take out chunk dim [B, chunk_T, v_dim]
                chunk_hyp_v = chunk_hyp_v.view(B, chunk_T, -1)

                ####### Forward outputs #######
                # Collected before the first update to prevent information leak
                output.append(chunk_hyp_v)  # [B, chunk_T, out_dim]

                # Use local chunk instead of queue to ignore corner cases that cause grad issues
                update_end_idx = None if self.lookahead == 0 else -self.lookahead
                update_hyp = chunk_hyp_v[:, :update_end_idx]
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
                    if self.lookahead_strategy == "mean":
                        update_ref = torch.stack(update_ref, dim=0).mean(dim=0)
                    elif self.lookahead_strategy == "last":
                        update_ref = update_ref[-1]
                    else:
                        raise NotImplementedError(f"Lookahead strategy {self.lookahead_strategy} not implemented.")
                update_hyp = update_hyp.contiguous()
                update_ref = update_ref.contiguous()
                update_mask = update_mask.contiguous() if update_mask is not None else None

                ###### FW update #######
                # Compute losses
                mem_losses = self.compute_mem_loss(
                    hyp_values=update_hyp.view(-1, update_hyp.size(-1)),
                    ref_values=update_ref.view(-1, update_ref.size(-1)),
                    loss_mask=update_mask.view(-1) if update_mask is not None else None,
                )
                updated_fw_named_params, grads = self.update_fw(
                    fw_named_params=cur_fw_named_params,
                    loss=mem_losses.mean(),
                )

                # Backward outputs
                all_grads.append([g.detach() for g in grads])
                fw_named_params_list.append(updated_fw_named_params)
                losses.append(mem_losses.view(B, -1).detach())  # [B, chunk_T-lookahead]

        output = torch.cat(output, dim=1)  # [B, T, out_dim]

        # Losses
        losses = torch.cat(losses, dim=1)  # [B, T]

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
            "losses": losses,
            "grad_norms": grad_norms,
            "past_key_values": None,
        }

    def forward_w_past(
        self,
        q,
        ref_v,
        chunk_size,
        loss_mask=None,
        past_key_values=None,
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
        mask_queue = torch.zeros(B, 0, device=ref_v.device, dtype=ref_v.dtype) if loss_mask is not None else None

        # Unprocessed info in past key values
        if past_key_values is not None:
            past_q, past_hyp_v, past_ref_v, _, past_mask, _ = past_key_values
            if past_hyp_v is not None:
                past_T = past_hyp_v.size(1)
                assert past_T < chunk_size + self.lookahead, (
                    f"Past key values length {past_T} longer than "
                    f"chunk size + lookahead ({chunk_size} + {self.lookahead})."
                )
                q_queue = torch.cat([past_q, q_queue], dim=1)
                hyp_v_queue = torch.cat([past_hyp_v, hyp_v_queue], dim=1)
                ref_v_queue = torch.cat([past_ref_v, ref_v_queue], dim=1)
                mask_queue = torch.cat([past_mask, mask_queue], dim=1) if loss_mask is not None else None

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

        all_grads = []
        output = []
        losses = []
        for chunk_start, chunk_end in chunk_boundaries:
            # Get chunk inputs/outputs
            chunk_T = chunk_end - chunk_start
            # [B, chunk_T, in_dim]
            chunk_q = q[:, chunk_start:chunk_end, :].contiguous()
            # [B, chunk_T, out_dim]
            chunk_ref_v = ref_v[:, chunk_start:chunk_end, :].contiguous()
            # [B, chunk_T]
            chunk_loss_mask = loss_mask[:, chunk_start:chunk_end].contiguous() if loss_mask is not None else None

            ####### Forward #######
            # All [B * chunk_T, *]
            retrieval_output_dict = self.retrieve_values(chunk_q, cur_fw_named_params)
            chunk_hyp_v = retrieval_output_dict["retireved_values"]
            # Take out chunk dim [B, chunk_T, v_dim]
            chunk_hyp_v = chunk_hyp_v.view(B, chunk_T, -1)

            # Forward outputs
            output.append(chunk_hyp_v)  # [B, chunk_T, out_dim]

            ####### Update signal queues #######
            # Append to queues
            q_queue = torch.cat([q_queue, chunk_q], dim=1)
            hyp_v_queue = torch.cat([hyp_v_queue, chunk_hyp_v], dim=1)
            ref_v_queue = torch.cat([ref_v_queue, chunk_ref_v], dim=1)
            mask_queue = torch.cat([mask_queue, chunk_loss_mask], dim=1) if loss_mask is not None else None

            ####### Update FW only if enough signal is in the queue #######
            if past_key_values and hyp_v_queue.size(1) < chunk_size:
                continue

            updated = True

            with torch.enable_grad():
                cur_fw_named_params = {n: p.detach() for n, p in cur_fw_named_params.items()}
                for p in cur_fw_named_params.values():
                    p.requires_grad = True

                ###### Forward with accumulated q #######
                update_q = q_queue[:, : -self.lookahead].contiguous()  # [B, queue_T - lookahead, in_dim]
                retrieval_output_dict = self.retrieve_values(update_q, cur_fw_named_params)
                chunk_hyp_v_bwd = retrieval_output_dict["retireved_values"]  # [B * (queue_T - lookahead), out_dim]

                ###### Construct FW update inputs from signal queues #######
                update_hyp = chunk_hyp_v_bwd  # [B, queue_T - lookahead, out_dim]
                update_mask = mask_queue[:, : -self.lookahead] if mask_queue is not None else None
                # Construct lookahead target values from `ref_v_queue`
                # update_ref: [B, queue_T - lookahead, out_dim]
                if self.lookahead == 0:
                    update_ref = ref_v_queue
                else:
                    update_ref = []
                    for i in range(self.lookahead + 1):
                        lookahead_idx = (i, -(self.lookahead - i)) if (self.lookahead - i) > 0 else (i, None)
                        update_ref.append(ref_v_queue[:, lookahead_idx[0] : lookahead_idx[1]])
                    if self.lookahead_strategy == "mean":
                        update_ref = torch.stack(update_ref, dim=0).mean(dim=0)
                    elif self.lookahead_strategy == "last":
                        update_ref = update_ref[-1]
                    else:
                        raise NotImplementedError(f"Lookahead strategy {self.lookahead_strategy} not implemented.")
                update_hyp = update_hyp.contiguous()  # [B, queue_T - lookahead, out_dim]
                update_ref = update_ref.contiguous()  # [B, queue_T - lookahead, out_dim]
                update_mask = update_mask.contiguous() if update_mask is not None else None

                ###### FW update #######
                # Compute losses
                mem_losses = self.compute_mem_loss(
                    hyp_values=update_hyp.view(-1, update_hyp.size(-1)),
                    ref_values=update_ref.view(-1, update_ref.size(-1)),
                    loss_mask=update_mask.view(-1) if update_mask is not None else None,
                )
                cur_fw_named_params, grads = self.update_fw(
                    fw_named_params=cur_fw_named_params,
                    loss=mem_losses.mean(),
                )

            ####### Pop out used signals from queues #######
            q_queue = q_queue[:, -self.lookahead :].contiguous()  # [B, lookahead, in_dim]
            hyp_v_queue = hyp_v_queue[:, -self.lookahead :].contiguous()  # [B, lookahead, out_dim]
            ref_v_queue = ref_v_queue[:, -self.lookahead :].contiguous()  # [B, lookahead, out_dim]
            mask_queue = mask_queue[:, -self.lookahead :].contiguous() if update_mask is not None else None

            ####### Backward outputs #######
            all_grads.append([g.detach() for g in grads])
            losses.append(mem_losses.view(B, -1).detach())  # [B, chunk_T-lookahead]

        output = torch.cat(output, dim=1)  # [B, T, out_dim]

        # Update past_key_values
        past_key_values = (q_queue, hyp_v_queue, ref_v_queue, None, mask_queue, None)

        # Collect update statistics if update happened
        if updated > 0:
            # Mem loss
            losses = torch.cat(losses, dim=1)  # [B, T]
            # Grad norms
            with torch.no_grad():
                grad_norms = []
                for i in range(len(cur_fw_named_params)):
                    grad_norm = torch.stack([g[i].norm() for g in all_grads]).mean()
                    grad_norms.append(grad_norm)
        else:
            losses = None
            grad_norms = None

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
            "losses": losses,
            "grad_norms": grad_norms,
            "past_key_values": past_key_values,
        }

    @torch.compiler.disable(recursive=True)
    def forward(
        self,
        q,
        ref_v,
        chunk_size,
        loss_mask=None,
        past_key_values=None,
    ):
        if past_key_values is None:
            return self.forward_wo_past(
                q,
                ref_v,
                chunk_size,
                loss_mask=loss_mask,
            )
        else:
            return self.forward_w_past(
                q,
                ref_v,
                chunk_size,
                loss_mask=loss_mask,
                past_key_values=past_key_values,
            )
