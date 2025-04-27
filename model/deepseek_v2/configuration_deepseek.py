from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
import torch
import io
import sys
import json
import pandas as pd
from datetime import timedelta
import os
import threading
import time
from dataclasses import dataclass
import torch.distributed as dist
from typing import Dict, List, Optional, Union, Callable

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DeepseekV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV2Model`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-V2.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DeepseekV2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts, None means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor or routed experts.
        topk_method (`str`, *optional*, defaults to `gready`):
            Topk method used in routed gate.
        n_group (`int`, *optional*, defaults to None):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to None):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of selected experts, None means dense model.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Method of computing expert weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss weight coefficient.
        seq_aux = (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import DeepseekV2Model, DeepseekV2Config

    >>> # Initializing a Deepseek-V2 style configuration
    >>> configuration = DeepseekV2Config()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size=1407,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=None,
        n_routed_experts=None,
        ep_size=1,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        topk_method="gready",
        n_group=None,
        n_local_group=8,
        topk_group=None,
        num_experts_per_tok=None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        expert_slice_degree=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        expert_window_size=0,
        balance=False,
        attention_type="MHA",
        module_type="LLAMA",
        token_dist=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.expert_slice_degree = expert_slice_degree
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.n_local_group = n_local_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.expert_window_size = expert_window_size
        self.balance = balance
        self.attention_type = attention_type
        self.module_type = module_type
        self.token_dist = token_dist
        self.set_comm_group()
        assert (module_type == "GPT" and hidden_act == "gelu") or (
            module_type == "LLAMA" and hidden_act == "silu"
        ), "If module_type is GPT, hidden_act must use gelu, if module_type is LLAMA hidden_act must use silu"
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def set_comm_group(self):
        assert self.ep_size * self.expert_slice_degree == self.n_group, (
            "world_size should be equal ep_size * expert_slice_degree"
            + f"but got world_size={self.n_group} and ep_size={self.ep_size} * expert_slice_degree={self.expert_slice_degree}"
        )
        assert mpu.world_size == self.n_group, f"Inconsistent parallel configuration {mpu.world_size} {self.n_group}"
        assert (
            mpu.local_world_size == self.n_local_group
        ), "Inconsistent parallel configuration"
        initialize_groups(
            mpu, self.ep_size, self.pretraining_tp, self.expert_slice_degree
        )

    @property
    def mpu(self):
        return mpu


@dataclass
class ParallelState:
    world_size: int = None
    global_rank: int = None
    local_rank: int = None
    local_world_size: int = None
    ep_first_rank: int = None
    ep_group: dist.ProcessGroup = None
    ep_group_world_size: int = 0
    tp_group: dist.ProcessGroup = None
    tp_group_world_size: int = 0
    expert_slice_group: dist.ProcessGroup = None
    expert_slice_group_world_size: int = 0
    # intra-node groups for every node rank
    inter_ep_groups: dist.ProcessGroup = None
    inter_ep_groups_world_size: int = 0
    global_ep_group: dist.ProcessGroup = None


def initialize_groups(
    mpu: ParallelState, ep_size: int = 1, tp_size: int = 1, expert_slice: int = 1
):
    mpu.ep_first_rank = 0
    mpu.global_ep_group = torch.distributed.new_group(
            [i for i in range(mpu.world_size)], pg_options=get_nccl_options("ep", {})
        )
    # 专家并行通信组
    for i in range(mpu.world_size // ep_size):
        ranks = list(range(i * ep_size, (i + 1) * ep_size))
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options("ep", {})
        )
        if mpu.global_rank in ranks:
            mpu.ep_first_rank = ranks[0]
            mpu.ep_group = group
            mpu.ep_group_world_size = len(ranks)

    # inter ep通信组
    for i in range(mpu.world_size // mpu.local_world_size):
        ranks = list(range(i * mpu.local_world_size, (i + 1) * mpu.local_world_size))
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options("inter_group", {})
        )
        if mpu.global_rank in ranks:
            mpu.inter_ep_groups = group
            mpu.inter_ep_groups_world_size = len(ranks)

    # 张量并行通信组
    for i in range(mpu.world_size // tp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options("tp", {})
        )
        if mpu.global_rank in ranks:
            mpu.tp_group = group
            mpu.tp_group_world_size = len(ranks)

    # 张量并行通信组
    for i in range(mpu.world_size // expert_slice):
        ranks = list(range(i, mpu.world_size, mpu.world_size // expert_slice))
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options("expert_slice", {})
        )
        if mpu.global_rank in ranks:
            mpu.expert_slice_group = group
            mpu.expert_slice_group_world_size = len(ranks)


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Arguments:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get(
            "cga_cluster_size", 4
        )
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get("max_ctas", 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get("min_ctas", 1)
        return nccl_options
    else:
        return None


mpu = ParallelState()
