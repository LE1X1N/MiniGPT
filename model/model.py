from transformers import PretrainedConfig

class MiniGPTConfig(PretrainedConfig):
    model_type = "minigpt"
    
    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        intermediate_size: int = None,
        max_positional_embeddings: int = 32768,
        num_attention_heads: int = 8,   
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1e6,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############# MoE ################
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_positional_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention

        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func
        
        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

# RMS norm
import torch
import torch.nn as nn
from typing import Optional, Union
import math
import torch.nn.functional as F
from transformers.activations import ACT2FN

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

# RoPE & YaRN
def precompute_freqs_cis(dim: int, end: int=32*1024, rope_base: float=1e6, rope_scaling: Optional[dict] = None):
    """
    compute cosθ + i·sinθ （cis）for RoPE
    
    :param dim (int): dim of channels / features
    :param end (int): max sequence length
    :param rope_base: base number 
    :param rope_scaling: frequency scaling configuration for inference
    """
    freqs = 1 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    
    if rope_scaling is not None:
        # YaRN, RoPE scaling freqs
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_positional_embeddings", 2048),
            rope_scaling.get("factor", 4),  # max support 2048*4 sequence length
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )
        
        if end > orig_max:
            corr_dim = next((i for i in range(dim // 2) if 2*math.pi / freqs[i] > orig_max), dim//2)    # first rotation period that large than pre-trained orig_max  
            power = torch.arange(0, dim//2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            scale = torch.where(
                torch.arange(dim//2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor), 
                1.0 / factor
            )
            freqs = freqs * scale
    
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()   # [end, dim/2] 
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [end, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # [end, dim]
    return freqs_cos, freqs_sin
    

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, unsqueeze_dim : int= 1):
    """
    apply RoPE on Q and K
    
    for a feature tuple (x0, x1), when apply RoPE
    
    [x0', =  [[cosθ, -sinθ],   * [x0
     x1']     [sinθ, cosθ]]       x1]
     
     => x0' = x0 * cosθ - x1 * sinθ
        x1' = x0 * cosθ + x1 * sinθ
                     
    :param q (torch.Tensor): Query matrix  (B, H, L, D)
    :param k (torch.Tensor): Key matrix    (B, H, L, D)
    :param sin (torch.Tensor): sinθ, (L, D)
    :param cos (torch.Tensor): cosθ  (L, D)
    :param unsqueeze_dim: target squeeze dimension for matching
    """
    
    def rotate_half(x):
        # [1, 2, 3, 4] -> [-3, -4, 1, 2]
        return torch.cat([-x[..., x.shape[-1] // 2: ],  # left part
                         x[..., : x.shape[-1] // 2]],   # right part
                         dim= -1)
    
    q_embed = (q*cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_embed = (k*cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k)*sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
     
    B, L, H, D = x.shape    # batch_size, sequence_len, num_head, hidden_dim
    return x[:, :, :, None, :].expand(B, L, H, n_rep, D).reshape(B, L, H*n_rep, D)


class Attention(nn.Module):
    def __init__(self, args: MiniGPTConfig):
        super().__init__()
        
        # it num_key_value_heads is None -> MHA, else GQA
        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        
        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads*self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads*self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention
    
    def forward(self, x:torch.Tensor, 
                positional_embeddings: tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool=False,
                attention_mask: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # split to heads and transpose (B, L, H, d) 
        xq = xq.view(B, L, self.n_local_heads, self.head_dim)         # (B, L, H_q, d)
        xk = xk.view(B, L, self.num_key_value_heads, self.head_dim)   # (B, L, H_kv, d)
        xv = xv.view(B, L, self.num_key_value_heads, self.head_dim)   # (B, L, H_kv, d)
        
        # apply RoPE on Q and K    
        cos, sin = positional_embeddings  
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:L], sin[:L], unsqueeze_dim=1)
        
        # apply repeat on K and V, and also KV cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)    # (B, L, H_kv, d) -> (B, L_new, H_kv, d)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None   # KV cache
        
        xq = xq.transpose(1, 2) # (B, H, L, d)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        
        # attention calculation
        if self.flash and L>1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (None if attention_mask is None else attention_mask.view(B, 1, 1, -1).expand(B, self.n_local_heads, L, -1).bool())
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) 
            causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=scores.device), 
                                         diagonal=1)
            
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        
        output = output.transpose(1, 2).reshape(B, L, -1)
        output = self.resid_dropout(self.o_proj(output))
        
        return output, past_kv
        
        
class FeedForward(nn.Module):
    def __init__(self, args: MiniGPTConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]
        
    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class MiniGPTBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniGPTConfig):
        super().__init__()
        self.layer_id = layer_id
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.self_attn = Attention(config)   # GQA layer
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = FeedForward(config)
    
    def forward(self, hidden_states: torch.Tensor,
                positional_embeddings: torch.Tensor,
                past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache : bool=False,
                attention_mask: torch.Tensor = None):
        residual = hidden_states
        
        hidden_states, present_key_value = self.self_attn(self.input_layernorm(hidden_states), 
                                                          positional_embeddings,
                                                          past_key_value,
                                                          use_cache,
                                                          attention_mask)
        hidden_states = residual + hidden_states    # residual
        
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states)) + hidden_states
        
        return hidden_states, present_key_value

class MiniGPTModel(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)   # input embedding
        
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList([
            MiniGPTBlock(i, config) for i in range(self.num_hidden_layers)
        ])
        
        # RoPE precompute
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs
                ):
        B, L = input_ids.shape
        
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
            
        past_key_values = past_key_values or [None]*len(self.layers)  # storing each layer's KV cache
        
        # past_key_values[0][0]: 0 layer, K cache of shape (B, L, H_kv, d)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos+L],
            self.freqs_sin[start_pos : start_pos+L]
        )
        
        presents = []   # present KV cache per layer
        
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(hidden_states,
                                            position_embeddings,
                                            past_key_value,
                                            use_cache,
                                            attention_mask)
            presents.append(present)
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, presents


from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class MiniGPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniGPTConfig
    
    def __init__(self, config: MiniGPTConfig):
        self.config = config
        super().__init__(config)
        
        self.model = MiniGPTModel(config)
        
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # weight sharing
        # the weight of embedding layer is the same with the projection head
        self.model.embed_tokens.weight = self.lm_head.weight
        
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_states, past_key_values = self.model(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
            **args)    
        
        # if logits_to_keep is an interger, thus we only retains the last target positions.
        slice_indices = (slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep)
           
        logits = self.lm_head(hidden_states[:, slice_indices, :])   # (B, L, V), V for vocab size, before softmax
        
        return CausalLMOutputWithPast(
            logits=logits, 
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
    