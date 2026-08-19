from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Qwen3PKMConfig(Qwen3Config):
    model_type = "qwen3_pkm"

    def __init__(
        self,

        # PKM (slow weights; replaces the MLP on the given layers)
        pkm_layers=None,
        pkm_k_dim=512,
        pkm_v_dim=512,
        pkm_heads=4,
        pkm_topk=32,
        pkm_n_subkeys=512,
        pkm_query_rmsnorm=True,
        # FwPKM (fast weights; extra block before/after attention)
        # general
        fwpkm_layers=None,
        fwpkm_k_dim=512,
        fwpkm_v_dim=512,
        fwpkm_heads=1,
        fwpkm_topk=8,
        fwpkm_n_subkeys=512,
        fwpkm_variant="pkm",  # "pkm" | "mlp"
        fwpkm_fp32_fw=True,
        fwpkm_before_attn=True,
        # optimization
        fwpkm_update_chunk_size=512,
        fwpkm_loss_type="mse",  # "mse" only atm
        fwpkm_optimizer_type="sgd",  # "sgd" only atm
        fwpkm_optimizer_lr=1.0,
        fwpkm_optimizer_weight_decay=0.0,
        fwpkm_grad_clip=False,
        fwpkm_addr_loss="me",  # None | "me"
        fwpkm_addr_loss_weight=1.0,
        fwpkm_weight_loss_with_gates=True,
        fwpkm_mem_grad_to_values_only=True,
        # input
        fwpkm_query_src="hidden",  # only "hidden" is supported in this Qwen3 port
        fwpkm_value_src="hidden",  # only "hidden" is supported in this Qwen3 port
        fwpkm_compress_query=None,  # None | "l2norm" | "zero_mean"
        fwpkm_compress_value="zero_mean",  # None | "l2norm" | "zero_mean"
        fwpkm_target_value_lookahead=1,
        # score
        fwpkm_score_nonlinear="softmax",  # "softmax" | "silu" | "relu"
        fwpkm_qk_score_type="idw",  # "dot_product" | "idw"
        fwpkm_score_temperature=1.0,
        # output
        fwpkm_out_fuse_gate=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pkm_layers = pkm_layers if pkm_layers is not None else []
        self.pkm_k_dim = pkm_k_dim
        self.pkm_v_dim = pkm_v_dim
        self.pkm_heads = pkm_heads
        self.pkm_topk = pkm_topk
        self.pkm_n_subkeys = pkm_n_subkeys
        self.pkm_query_rmsnorm = pkm_query_rmsnorm

        self.fwpkm_layers = fwpkm_layers if fwpkm_layers is not None else []
        self.fwpkm_k_dim = fwpkm_k_dim
        self.fwpkm_v_dim = fwpkm_v_dim
        self.fwpkm_heads = fwpkm_heads
        self.fwpkm_topk = fwpkm_topk
        self.fwpkm_n_subkeys = fwpkm_n_subkeys
        self.fwpkm_variant = fwpkm_variant
        self.fwpkm_fp32_fw = fwpkm_fp32_fw
        self.fwpkm_before_attn = fwpkm_before_attn
        self.fwpkm_update_chunk_size = fwpkm_update_chunk_size
        self.fwpkm_loss_type = fwpkm_loss_type
        self.fwpkm_optimizer_type = fwpkm_optimizer_type
        self.fwpkm_optimizer_lr = fwpkm_optimizer_lr
        self.fwpkm_optimizer_weight_decay = fwpkm_optimizer_weight_decay
        self.fwpkm_grad_clip = fwpkm_grad_clip
        self.fwpkm_addr_loss = fwpkm_addr_loss
        self.fwpkm_addr_loss_weight = fwpkm_addr_loss_weight
        self.fwpkm_weight_loss_with_gates = fwpkm_weight_loss_with_gates
        self.fwpkm_mem_grad_to_values_only = fwpkm_mem_grad_to_values_only
        self.fwpkm_out_fuse_gate = fwpkm_out_fuse_gate
        self.fwpkm_query_src = fwpkm_query_src
        self.fwpkm_value_src = fwpkm_value_src
        self.fwpkm_target_value_lookahead = fwpkm_target_value_lookahead
        self.fwpkm_compress_value = fwpkm_compress_value
        self.fwpkm_compress_query = fwpkm_compress_query
        self.fwpkm_score_nonlinear = fwpkm_score_nonlinear
        self.fwpkm_qk_score_type = fwpkm_qk_score_type
        self.fwpkm_score_temperature = fwpkm_score_temperature

        self.num_fwpkm_layers = len(self.fwpkm_layers)


        if self.fwpkm_query_src != "hidden" or self.fwpkm_value_src != "hidden":
            raise NotImplementedError(
                "This only supports fwpkm_query_src='hidden' and "
                "fwpkm_value_src='hidden' 
            )
        for l in self.pkm_layers:
            if not (0 <= l < self.num_hidden_layers):
                raise ValueError(f"pkm_layers contains invalid layer index {l}")
        for l in self.fwpkm_layers:
            if not (0 <= l < self.num_hidden_layers):
                raise ValueError(f"fwpkm_layers contains invalid layer index {l}")


__all__ = ["Qwen3PKMConfig"]
