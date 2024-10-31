import torch, math
from torch import nn
from torch.nn import functional as func_nn
import numpy as np
from transformers import AutoTokenizer, T5EncoderModel
from timm.models.vision_transformer import Mlp, DropPath
from einops import rearrange
from typing import TypeAlias
from minified.utils import config
from collections.abc import Iterable
from itertools import repeat


tensor: TypeAlias = torch.Tensor # alias typing for torch.Tensor

# T5 text encoder
t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
t5_model = T5EncoderModel.from_pretrained("google-t5/t5-small")

def text_t5_encode(text_input: str, tokenizer=t5_tokenizer, model=t5_model):
    input_ids = tokenizer(text_input, return_tensors="pt").input_ids  # Batch size 1
    outputs = model(input_ids=input_ids)
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states

def modulate_t2i(x, scale, shift):
    return x * (1 + scale) + shift


# DiT block module
class DitBlock(nn.Module):
    """
    Pixart DiT block [
        embed/layernorm - scale/shift - Multi self-attention - scale ->
        concat w/input - multi cross-attention -> concat w/attention tokens ->
        scale/shift - pointwise feedforward -> scale -> concat w/ff_tokens
    
        MLP for time embedding.
    ]
    """
    
    def __init__(self, hidden_size=config.hidden_size, mlp_ratio=4.0, drop_rate=0.0):
        super().__init__()
        
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = SelfAttention()
        self.cross_attention = CrossAttention()
        
        self.pointwise_feedforward = None

        self.mlp_layer = Mlp(
            in_features=hidden_size, hidden_features=int(mlp_ratio*hidden_size), 
            act_layer=nn.GELU(approximate='tanh'), drop=drop_rate
        )
        # lookup table 
        self.shift_scale_params = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        self.drop_path = DropPath(drop_rate)
        
    def _modulate(self, scale, shift):
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x
    

    def forward(self, x_input: tensor, y_cond: tensor, timestep):
        b, l, c = x_input.shape # get noised input shape
        
        # get scale/shift values from table, reshape
        scale_shift_params = (self.shift_scale_params[None] + timestep.reshape(b, 6, -1)) 
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = scale_shift_params.chunk(6, dim=1) 
        
        x_mod = modulate_t2i((self.layernorm(x_input)), shift_msa, scale_msa)
        x_attn = x_input + self.drop_path(gate_msa * self.self_attention(x_mod))
        
        x_crossattn = x_attn + self.cross_attention(x_attn, y_cond).reshape(b, l, c)

        x2 = modulate_t2i(self.layernorm(x_crossattn), shift_mlp, scale_mlp)

        x_mlp = x_crossattn + self.drop_path(gate_mlp * self.mlp_layer(x2))
                                             
        return x_mlp


# self attention blocks for DiT block
class SelfAttention(nn.Module):
    def __init__(self, n_heads=config.attn_heads, embed_dim=config.embed_dim, drop_rate=0.1):
        super().__init__()
        
        self.attn_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        
        self.input_project = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_project = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(drop_rate)
        
        
    def forward(self, x: torch.Tensor):
        
        x = self.input_project(x) # input linear projection
        
        # chunk into query, key, value vectors
        q, k, v = x.chunk(3, dim=1)
        
        # unfold tensors for attention computation
        # b - batch, n - sequence length, h - attn heads, e - embed_dim
        q = rearrange(q, 'b n (h e) -> b h n e', h=self.attn_heads) 
        k = rearrange(k, 'b n (h e) -> b h n e', h=self.attn_heads) 
        v = rearrange(v, 'b n (h e) -> b h n e', h=self.attn_heads) 
        
        # attention computatation
        attn_weight = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # dot product of query/keys, divide by root of head di
        attn_score = func_nn.softmax(attn_weight, dim=1) # calculate softmax (squeeze into range(0, 1))
        attn_output = attn_score @ v # multiply with value vectors
        
        output = rearrange(attn_output, 'b h n e -> b n (h e)') # fold back to input shape
        output = self.dropout(self.output_project(output)) # output projection
        
        return output


# self attention blocks for DiT block
class CrossAttention(nn.Module):
    def __init__(self, n_heads=config.attn_heads, embed_dim=config.embed_dim, cross_dim=120, drop_rate=0.1):
        super().__init__()
        
        self.attn_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_linear = nn.Linear(cross_dim, embed_dim, bias=True)
        self.v_linear = nn.Linear(cross_dim, embed_dim, bias=True)
        
        self.output_project = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x_input: torch.Tensor, y_cond: torch.Tensor):
                
        # get query for iput, key/value vectors fror conditions
        q = self.q_linear(x_input)
        k = self.k_linear(y_cond)
        v = self.v_linear(y_cond)
        
        # unfold tensors for attention computation
        # b - batch, n - sequence length, h - attn heads, e - embed_dim
        q = rearrange(q, 'b n (h e) -> b h n e', h=self.attn_heads) 
        k = rearrange(k, 'b n (h e) -> b h n e', h=self.attn_heads) 
        v = rearrange(v, 'b n (h e) -> b h n e', h=self.attn_heads) 
        
        # attention computatation
        attn_weight = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # dot product of query/keys, divide by root of head di
        attn_score = func_nn.softmax(attn_weight, dim=1) # calculate softmax (squeeze into range(0, 1))
        output = attn_score @ v # multiply with value vectors
        
        output = rearrange(output, 'b h n e -> b n (h e)') # fold back to input shape
        output = self.output_project(output) # output projection
        
        return output


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.shift_scale_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, timestep):

        shift_scale_params = [self.shift_scale_table[None] + timestep[:, None]]
        shift, scale = shift_scale_params.chunk(2, dim=1)

        x_modulate = modulate_t2i(self.layernorm(x), shift, scale)

        x = self.linear(x_modulate)

        return x


# embeds tet conditioning vectors, token dropout
class CaptionEmbedder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        self.project = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) // in_channels**0.5),
        )
        self.uncond_prob = uncond_prob

    # drop labels to enable classifier-free guidance
    def token_drop(self, caption, force_drop_ids=None):
        drop_ids = None

        if force_drop_ids is None:
            drop_ids = torch.randn(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1

        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(
        self, caption: torch.Tensor, training: bool = True, force_drop_ids=None
    ):
        if training:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0

        if (training and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)

        caption = self.project(caption)

        return caption


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_embed_size=config.freq_embed):
        super().__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_embed_size = freq_embed_size
        self.dtype = next(self.parameters()).dtype

    # sinusoidal timestep embeddings
    @staticmethod
    def timestep_embed(timestep: tensor, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timestep.device
            )
            / half
        )
        args = timestep[:, None].float() * freqs[None]

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding

    def forward(self, timestep):
        time_freq = self.timestep_embed(timestep, self.freq_embed_size).to(self.dtype)

        return self.mlp_layer(time_freq)


def get_1d_sincos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2
    omega = 1. / 10000 ** omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d -> md', pos, omega)
    
    sin_embed = np.sin(out)
    cos_embed = np.cos(out)
    sincos_embed = np.concatenate([sin_embed, cos_embed], axis=1)
    
    return sincos_embed

def get_2d_sincos_embed_from_grid(embed_dim, grid):

    h_embed = get_1d_sincos_embed_from_grid(embed_dim // 2, grid[0])
    w_embed = get_1d_sincos_embed_from_grid(embed_dim // 2, grid[1])

    sincos_2d = np.concatenate([h_embed, w_embed], axis=1)
    
    return sincos_2d

# copied directly from official code
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)

    grid_h = (
        np.arange(grid_size[0], dtype=np.float32)
        / (grid_size[0] / base_size)
        / lewei_scale
    )
    grid_w = (
        np.arange(grid_size[1], dtype=np.float32)
        / (grid_size[1] / base_size)
        / lewei_scale
    )
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed
