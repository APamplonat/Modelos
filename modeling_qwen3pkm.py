"""Qwen3 + Product Key Memory (slow weights) + Fast-weight PKM (FwPKM)"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import logging

from .configuration_qwen3pkm import Qwen3PKMConfig

from .pkm.memory import HashingMemory
from .fwpkm.fwpkm import FastWeightProductKeyMemory, l2norm
from .fwpkm.fwmlp import FastWeightMLP

logger = logging.get_logger(__name__)


# Standard Qwen3 
class Qwen3PKMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3PKMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Qwen3PKMRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3PKMConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            self.rope_type = rope_parameters["rope_type"]
        elif getattr(config, "rope_scaling", None) is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
        else:
            self.rope_type = "default"
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config=None, device=None, seq_len=None):
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            base = rope_parameters["rope_theta"]
        else:
            base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3PKMAttention(nn.Module):
    """Standard Qwen3 attention """

    def __init__(self, config: Qwen3PKMConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if getattr(config, "layer_types", None) else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3PKMRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3PKMRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



##########################################################################################
##########################################################################################

class Qwen3PKMDynamicCache(DynamicCache):
    def __init__(self, config: Qwen3PKMConfig, **kwargs):
        try:
            super().__init__(config=config, **kwargs)
        except TypeError: 
            super().__init__(**kwargs)
        _init_fwpkm_cache(self, config)

def _init_fwpkm_cache(cache, config: Qwen3PKMConfig):
    # (q, hyp_v, ref_v, gates, mask, ids) per layer
    cache.fwpkm_cache = [(None, None, None, None, None, None) for _ in range(config.num_hidden_layers)]
    cache.first_fwpkm_layer = config.fwpkm_layers[0] if len(config.fwpkm_layers) > 0 else None


def _get_fwpkm_cache_length(cache) -> Optional[int]:
    if cache is None or getattr(cache, "first_fwpkm_layer", None) is None:
        return None
    fwpkm_cache = cache.fwpkm_cache[cache.first_fwpkm_layer]
    if fwpkm_cache[0] is None:
        return 0
    return fwpkm_cache[0].shape[1]

def _reset_fwpkm_cache(cache):
    if hasattr(cache, "fwpkm_cache"):
        for layer_idx in range(len(cache.fwpkm_cache)):
            cache.fwpkm_cache[layer_idx] = (None, None, None, None, None, None)


##########################################################################################
##########################################################################################

@dataclass
class Qwen3PKMModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    all_pkm_idcs: Optional[torch.LongTensor] = None
    all_fwpkm_losses: Optional[torch.FloatTensor] = None
    all_fwpkm_idcs: Optional[torch.LongTensor] = None
    all_fwpkm_scores: Optional[torch.FloatTensor] = None
    all_fwpkm_addr_stats: Optional[list] = None
    all_fwpkm_grad_norms: Optional[list] = None
    all_fwpkm_gates: Optional[torch.FloatTensor] = None


@dataclass
class Qwen3PKMCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    all_pkm_idcs: Optional[torch.LongTensor] = None
    all_fwpkm_losses: Optional[torch.FloatTensor] = None
    all_fwpkm_idcs: Optional[torch.LongTensor] = None
    all_fwpkm_scores: Optional[torch.FloatTensor] = None
    all_fwpkm_addr_stats: Optional[list] = None
    all_fwpkm_grad_norms: Optional[list] = None
    all_fwpkm_gates: Optional[torch.FloatTensor] = None


#########################################################################
# Decoder layer
#########################################################################
class Qwen3PKMDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3PKMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.fwpkm_layer_idx = config.fwpkm_layers.index(layer_idx) if layer_idx in config.fwpkm_layers else None

        # token mixer: standard Qwen3 attention
        self.layer_type = config.layer_types[layer_idx] if getattr(config, "layer_types", None) else "full_attention"
        self.self_attn = Qwen3PKMAttention(config=config, layer_idx=layer_idx)

        self.input_layernorm = Qwen3PKMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3PKMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # PKM (slow weights)
        if layer_idx in config.pkm_layers:
            self.pkm = HashingMemory(
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                mem_k_dim=config.pkm_k_dim,
                mem_v_dim=config.pkm_v_dim,
                mem_heads=config.pkm_heads,
                mem_knn=config.pkm_topk,
                mem_n_keys=config.pkm_n_subkeys,
                mem_share_values=False,
                mem_query_rmsnorm=config.pkm_query_rmsnorm,
            )
            self.mlp = None
        else:
            self.pkm = None
            self.mlp = Qwen3PKMMLP(config)

        # FPKM (fast weights)
        if layer_idx in config.fwpkm_layers:
            # Query
            fwpkm_query_src_dim = self.get_src_state_size(config.fwpkm_query_src)
            self.fw_q_norm = Qwen3PKMRMSNorm(fwpkm_query_src_dim, eps=config.rms_norm_eps)
            self.fw_q_proj = nn.Linear(fwpkm_query_src_dim, config.fwpkm_k_dim * config.fwpkm_heads)

            # Value
            fwpkm_value_src_dim = self.get_src_state_size(config.fwpkm_value_src)
            self.fw_v_norm = Qwen3PKMRMSNorm(fwpkm_value_src_dim, eps=config.rms_norm_eps)
            self.fw_v_proj = nn.Linear(fwpkm_value_src_dim, config.fwpkm_v_dim)

            if config.fwpkm_variant == "pkm":
                self.fwpkm = FastWeightProductKeyMemory(
                    mem_k_dim=config.fwpkm_k_dim,
                    mem_v_dim=config.fwpkm_v_dim,
                    mem_heads=config.fwpkm_heads,
                    mem_topk=config.fwpkm_topk,
                    mem_n_subkeys=config.fwpkm_n_subkeys,
                    lookahead=config.fwpkm_target_value_lookahead,
                    qk_score_type=config.fwpkm_qk_score_type,
                    optimizer_type=config.fwpkm_optimizer_type,
                    learning_rate=config.fwpkm_optimizer_lr,
                    weight_decay=config.fwpkm_optimizer_weight_decay,
                    loss_type=config.fwpkm_loss_type,
                    mem_grad_to_values_only=config.fwpkm_mem_grad_to_values_only,
                    grad_clip=config.fwpkm_grad_clip,
                    addr_loss=config.fwpkm_addr_loss,
                    addr_loss_weight=config.fwpkm_addr_loss_weight,
                    fp32_fw=config.fwpkm_fp32_fw,
                    score_nonlinear=config.fwpkm_score_nonlinear,
                    score_temperature=config.fwpkm_score_temperature,
                )
            elif config.fwpkm_variant == "mlp":
                assert config.fwpkm_heads == 1
                self.fwpkm = FastWeightMLP(
                    input_dim=config.fwpkm_k_dim,
                    output_dim=config.fwpkm_v_dim,
                    size=config.fwpkm_n_subkeys,
                    lookahead=config.fwpkm_target_value_lookahead,
                    optimizer_type=config.fwpkm_optimizer_type,
                    learning_rate=config.fwpkm_optimizer_lr,
                    weight_decay=config.fwpkm_optimizer_weight_decay,
                    grad_clip=config.fwpkm_grad_clip,
                )
            else:
                raise NotImplementedError(f"FwPKM variant {config.fwpkm_variant} not implemented")

            if config.fwpkm_out_fuse_gate:
                self.fw_out_gate_norm = Qwen3PKMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.fw_out_gate = nn.Linear(config.hidden_size, 8)

            self.fw_out_norm = Qwen3PKMRMSNorm(config.fwpkm_v_dim, eps=config.rms_norm_eps)
            self.fw_out_proj = nn.Linear(config.fwpkm_v_dim, config.hidden_size)
        else:
            self.fwpkm = None

    def get_src_state_size(self, state_src: str) -> int:
        # Standard Qwen3: only full/sliding softmax attention layers exist.
        hidden_state_size = self.config.hidden_size
        if state_src in ["hidden", "output"]:
            return hidden_state_size
        else:
            raise NotImplementedError(
                f"FwPKM state source {state_src} not implemented in the Qwen3 port (only 'hidden')"
            )

    def fwpkm_construct_states(
        self,
        hidden_states: torch.Tensor,
        query_states: Optional[torch.Tensor],
        key_states: Optional[torch.Tensor],
        value_states: Optional[torch.Tensor],
        raw_attn_output_states: Optional[torch.Tensor],
        attn_output_states: Optional[torch.Tensor],
        state_src: str,
    ) -> torch.Tensor:
        if state_src == "hidden":
            states = hidden_states
        else:
            raise NotImplementedError(
                f"FwPKM state source {state_src} not implemented (only 'hidden')"
            )
        return states

    def compute_gates(self, gate_input: torch.Tensor) -> torch.Tensor:
        gate_input = self.fw_out_gate_norm(gate_input)
        gate_values = self.fw_out_gate(gate_input).mean(dim=-1, keepdim=True)
        gate_values = torch.sigmoid(gate_values)
        return gate_values

    def compress_fwpkm_inputs(self, type, x, two_parts=False):
        def _compress(type, x_part):
            if type == "l2norm":
                return l2norm(x_part)
            elif type == "zero_mean":
                mean = x_part.mean(dim=-1, keepdim=True)
                std = x_part.std(dim=-1, keepdim=True) + 1e-6
                return (x_part - mean) / std
            else:
                raise NotImplementedError(f"FwPKM input compression type {type} not implemented")

        if two_parts:
            half_dim = x.size(-1) // 2
            x_part1 = x[:, :, :half_dim]
            x_part2 = x[:, :, half_dim:]
            x_part1 = _compress(type, x_part1)
            x_part2 = _compress(type, x_part2)
            return torch.cat([x_part1, x_part2], dim=-1)
        else:
            return _compress(type, x)

    @torch.no_grad()
    @torch.compiler.disable(recursive=True)
    def compute_addr_stats(
        self,
        idcs: torch.LongTensor,
        addr_loss: Optional[torch.FloatTensor] = None,
    ) -> dict[str, float]:
        def _compute_addr_stats(self, idcs: torch.LongTensor) -> dict[str, float]:
            addr_stats = {}
            idx_counter = torch.bincount(idcs.view(-1), minlength=self.config.fwpkm_n_subkeys**2)
            addr_stats["collision_ratio"] = idx_counter[idx_counter > 1].sum().item() / (idcs.numel() + 1e-6)
            addr_stats["coverage_ratio"] = (idx_counter > 0).sum().item() / self.config.fwpkm_n_subkeys**2
            addr_stats["kld"] = torch.distributions.kl.kl_divergence(
                torch.distributions.Categorical(probs=idx_counter.float() / (idx_counter.sum() + 1e-6)),
                torch.distributions.Categorical(probs=torch.ones_like(idx_counter).float() / idx_counter.numel()),
            ).item()
            return addr_stats

        full_addr_stats = {}
        addr_stats = _compute_addr_stats(self, idcs)
        for k, v in addr_stats.items():
            full_addr_stats[f"{k}"] = v
        if addr_loss is not None:
            full_addr_stats[f"addressing_loss/{self.config.fwpkm_addr_loss}"] = addr_loss.mean().item()
        return full_addr_stats

    #Proyecta q/v desde hidden, comprime, calcula el gate, llama a la memoria, rapida (con el fwpkm_cache de la capa), aplica gating con el value residual,
    #normaliza, proyecta y suma al residual.
    def fwpkm_forward(
        self,
        hidden_states: torch.Tensor,
        query_states: Optional[torch.Tensor],
        key_states: Optional[torch.Tensor],
        value_states: Optional[torch.Tensor],
        raw_attn_output_states: Optional[torch.Tensor],
        attn_output_states: Optional[torch.Tensor],
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        residual = hidden_states

        # Query projection
        fwpkm_query_states = self.fwpkm_construct_states(
            hidden_states=hidden_states,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            raw_attn_output_states=raw_attn_output_states,
            attn_output_states=attn_output_states,
            state_src=self.config.fwpkm_query_src,
        )
        fwpkm_query_states = self.fw_q_norm(fwpkm_query_states)
        fwpkm_query_states = self.fw_q_proj(fwpkm_query_states)

        # Value projection
        fwpkm_value_states = self.fwpkm_construct_states(
            hidden_states=hidden_states,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            raw_attn_output_states=raw_attn_output_states,
            attn_output_states=attn_output_states,
            state_src=self.config.fwpkm_value_src,
        )
        fwpkm_value_states = self.fw_v_norm(fwpkm_value_states)
        fwpkm_value_states = self.fw_v_proj(fwpkm_value_states)

        # Compression
        if self.config.fwpkm_compress_query is not None:
            fwpkm_query_states = self.compress_fwpkm_inputs(
                self.config.fwpkm_compress_query,
                fwpkm_query_states,
                two_parts=True,
            )
        if self.config.fwpkm_compress_value is not None:
            fwpkm_value_states = self.compress_fwpkm_inputs(
                self.config.fwpkm_compress_value,
                fwpkm_value_states,
                two_parts=False,
            )

        # Gating projection
        gate_values = None
        if self.config.fwpkm_out_fuse_gate:
            gate_values = self.compute_gates(hidden_states)  # [batch, seq_len, 1]

        # attention mask
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
            attention_mask = attention_mask[:, 0, -1, :]
        elif attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            attention_mask = None 

        # FwPKM forward
        if self.config.fwpkm_variant == "pkm":
            fwpkm_output_dict = self.fwpkm(
                q=fwpkm_query_states,
                ref_v=fwpkm_value_states,
                gates=gate_values.squeeze(-1) if self.config.fwpkm_weight_loss_with_gates else None,
                chunk_size=self.config.fwpkm_update_chunk_size,
                loss_mask=attention_mask,
                past_key_values=past_key_values.fwpkm_cache[self.layer_idx] if past_key_values is not None else None,
                token_ids=input_ids,
            )
            fwpkm_output = fwpkm_output_dict["output"]
            fwpkm_idcs = fwpkm_output_dict["indices"]
            fwpkm_scores = fwpkm_output_dict["scores"]
            fwpkm_losses = fwpkm_output_dict["losses"]
            fwpkm_grad_norms = fwpkm_output_dict["grad_norms"]
            fwpkm_addr_loss = fwpkm_output_dict["addr_loss"]
            fwpkm_past_key_values = fwpkm_output_dict["past_key_values"]
        elif self.config.fwpkm_variant == "mlp":
            fwpkm_output_dict = self.fwpkm(
                q=fwpkm_query_states,
                ref_v=fwpkm_value_states,
                chunk_size=self.config.fwpkm_update_chunk_size,
                loss_mask=attention_mask,
                past_key_values=past_key_values.fwpkm_cache[self.layer_idx] if past_key_values is not None else None,
            )
            fwpkm_output = fwpkm_output_dict["output"]
            fwpkm_idcs = None
            fwpkm_scores = None
            fwpkm_losses = fwpkm_output_dict["losses"]
            fwpkm_grad_norms = fwpkm_output_dict["grad_norms"]
            fwpkm_addr_loss = None
            fwpkm_past_key_values = fwpkm_output_dict["past_key_values"]
        else:
            raise NotImplementedError(f"FwPKM variant {self.config.fwpkm_variant} not implemented")

        # Update past key values
        if past_key_values is not None:
            past_key_values.fwpkm_cache[self.layer_idx] = fwpkm_past_key_values

        # Addressing stats
        if fwpkm_idcs is not None:
            fwpkm_addr_stats = self.compute_addr_stats(fwpkm_idcs, fwpkm_addr_loss)
        else:
            fwpkm_addr_stats = None

        # Gating
        if self.config.fwpkm_out_fuse_gate:
            fwpkm_output = fwpkm_output * gate_values

        # Residual value
        if self.config.fwpkm_out_fuse_gate:
            fwpkm_value_states = fwpkm_value_states * (1.0 - gate_values)
        fwpkm_output = fwpkm_output + fwpkm_value_states

        # Output projection
        fwpkm_output = self.fw_out_norm(fwpkm_output)
        fwpkm_output = self.fw_out_proj(fwpkm_output)

        # Final residual connection
        hidden_states = residual + fwpkm_output

        return (
            hidden_states,
            fwpkm_output,
            fwpkm_idcs,
            fwpkm_scores,
            fwpkm_addr_stats,
            fwpkm_losses,
            fwpkm_grad_norms,
            fwpkm_past_key_values,
            gate_values,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> dict:

        # FwPKM outputs
        fwpkm_output = None
        fwpkm_idcs = None
        fwpkm_scores = None
        fwpkm_addr_stats = None
        fwpkm_losses = None
        fwpkm_grad_norms = None
        fwpkm_past_key_values = None
        gate_values = None

        # FwPKM (before attention)
        if self.fwpkm is not None and self.config.fwpkm_before_attn:
            assert self.config.fwpkm_query_src in ("hidden")
            assert self.config.fwpkm_value_src in ("hidden")
            (
                hidden_states,
                fwpkm_output,
                fwpkm_idcs,
                fwpkm_scores,
                fwpkm_addr_stats,
                fwpkm_losses,
                fwpkm_grad_norms,
                fwpkm_past_key_values,
                gate_values,
            ) = self.fwpkm_forward(
                hidden_states=hidden_states,
                query_states=None,
                key_states=None,
                value_states=None,
                raw_attn_output_states=None,
                attn_output_states=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + attn_output_states

        if self.fwpkm is not None and not self.config.fwpkm_before_attn:
            (
                hidden_states,
                fwpkm_output,
                fwpkm_idcs,
                fwpkm_scores,
                fwpkm_addr_stats,
                fwpkm_losses,
                fwpkm_grad_norms,
                fwpkm_past_key_values,
                gate_values,
            ) = self.fwpkm_forward(
                hidden_states=hidden_states,
                query_states=None,
                key_states=None,
                value_states=None,
                raw_attn_output_states=None,
                attn_output_states=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.pkm is not None:
            hidden_states, pkm_idcs = self.pkm(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            pkm_idcs = None

        hidden_states = residual + hidden_states

        return {
            "hidden_states": hidden_states,
            "pkm_idcs": pkm_idcs,
            "fwpkm_losses": fwpkm_losses,
            "fwpkm_idcs": fwpkm_idcs,
            "fwpkm_scores": fwpkm_scores,
            "fwpkm_addr_stats": fwpkm_addr_stats,
            "fwpkm_grad_norms": fwpkm_grad_norms,
            "fwpkm_gates": gate_values.detach() if gate_values is not None else None,
            "fwpkm_past_key_values": fwpkm_past_key_values,
        }

class Qwen3PKMPreTrainedModel(PreTrainedModel):
    config_class = Qwen3PKMConfig
    config: Qwen3PKMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3PKMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _is_stateful = True

    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, (FastWeightProductKeyMemory, HashingMemory, FastWeightMLP)):
            module.reset_parameters()


class Qwen3PKMModel(Qwen3PKMPreTrainedModel):
    def __init__(self, config: Qwen3PKMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3PKMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3PKMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3PKMRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Qwen3PKMModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3PKMDynamicCache(config=self.config)
        # Caches created externally (e.g. by `generate`) may be plain
        # DynamicCache objects: attach the FwPKM state lazily.
        if past_key_values is not None and not hasattr(past_key_values, "fwpkm_cache"):
            _init_fwpkm_cache(past_key_values, self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Standard Qwen3 mask preparation
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # ---- collection lists: verbatim from Qwen3NextMemModel.forward ----
        all_pkm_idcs = []
        all_fwpkm_losses = []
        all_fwpkm_idcs = []
        all_fwpkm_scores = []
        all_fwpkm_addr_stats = []
        all_fwpkm_grad_norms = []
        all_fwpkm_gates = []

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_output_dict = decoder_layer(
                hidden_states,
                input_ids=input_ids,
                attention_mask=causal_mask_mapping["full_attention"],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_output_dict["hidden_states"]
            pkm_idcs = layer_output_dict["pkm_idcs"]
            fwpkm_losses = layer_output_dict["fwpkm_losses"]
            fwpkm_idcs = layer_output_dict["fwpkm_idcs"]
            fwpkm_scores = layer_output_dict["fwpkm_scores"]
            fwpkm_addr_stats = layer_output_dict["fwpkm_addr_stats"]
            fwpkm_grad_norms = layer_output_dict["fwpkm_grad_norms"]
            fwpkm_gates = layer_output_dict["fwpkm_gates"]

            if pkm_idcs is not None:
                all_pkm_idcs.append(pkm_idcs)
            if fwpkm_losses is not None:
                all_fwpkm_losses.append(fwpkm_losses)
            if fwpkm_idcs is not None:
                all_fwpkm_idcs.append(fwpkm_idcs)
            if fwpkm_scores is not None:
                all_fwpkm_scores.append(fwpkm_scores)
            if fwpkm_addr_stats is not None:
                all_fwpkm_addr_stats.append(fwpkm_addr_stats)
            if fwpkm_grad_norms is not None:
                all_fwpkm_grad_norms.append(fwpkm_grad_norms)
            if fwpkm_gates is not None:
                all_fwpkm_gates.append(fwpkm_gates)

        all_pkm_idcs = (
            torch.stack(all_pkm_idcs, dim=2) if all_pkm_idcs else None
        )  # (batch_size, seq_len, num_pkm_layers, knn)
        all_fwpkm_losses = (
            torch.stack(all_fwpkm_losses, dim=2) if all_fwpkm_losses else None
        )  # (batch_size, seq_len, num_fwpkm_layers, ...)
        all_fwpkm_idcs = torch.stack(all_fwpkm_idcs, dim=2) if all_fwpkm_idcs else None
        all_fwpkm_scores = torch.stack(all_fwpkm_scores, dim=2) if all_fwpkm_scores else None
        all_fwpkm_gates = torch.stack(all_fwpkm_gates, dim=2) if all_fwpkm_gates else None

        hidden_states = self.norm(hidden_states)

        return Qwen3PKMModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            all_pkm_idcs=all_pkm_idcs,
            all_fwpkm_losses=all_fwpkm_losses,
            all_fwpkm_idcs=all_fwpkm_idcs,
            all_fwpkm_scores=all_fwpkm_scores,
            all_fwpkm_addr_stats=all_fwpkm_addr_stats if all_fwpkm_addr_stats else None,
            all_fwpkm_grad_norms=all_fwpkm_grad_norms if all_fwpkm_grad_norms else None,
            all_fwpkm_gates=all_fwpkm_gates,
        )


class Qwen3PKMForCausalLM(Qwen3PKMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3PKMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

 
    def adjust_fwpkm_update_chunksize(self, input_len: int, past_key_values=None):

        if past_key_values is None:
            self.config.fwpkm_update_chunk_size = input_len
        else:
            fwpkm_cache_len = _get_fwpkm_cache_length(past_key_values)
            if fwpkm_cache_len is not None:
                self.config.fwpkm_update_chunk_size = input_len + fwpkm_cache_len
            else:
                self.config.fwpkm_update_chunk_size = input_len

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Qwen3PKMCausalLMOutput:
        outputs: Qwen3PKMModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])


        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return Qwen3PKMCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            all_pkm_idcs=outputs.all_pkm_idcs,
            all_fwpkm_losses=outputs.all_fwpkm_losses,
            all_fwpkm_idcs=outputs.all_fwpkm_idcs,
            all_fwpkm_scores=outputs.all_fwpkm_scores,
            all_fwpkm_addr_stats=outputs.all_fwpkm_addr_stats,
            all_fwpkm_grad_norms=outputs.all_fwpkm_grad_norms,
            all_fwpkm_gates=outputs.all_fwpkm_gates,
        )


__all__ = [
    "Qwen3PKMForCausalLM",
    "Qwen3PKMModel",
    "Qwen3PKMPreTrainedModel",
    "Qwen3PKMDynamicCache",
    "apply_fwpkm_identity_init",
]

