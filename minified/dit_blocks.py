import torch, math
from torch import nn
from torch.nn import functional as func_nn
from transformers import AutoTokenizer, T5EncoderModel
from einops import rearrange
from minified.utils import config

# T5 text encoder
t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
t5_model = T5EncoderModel.from_pretrained("google-t5/t5-small")

def text_t5_encode(text_input: str, tokenizer=t5_tokenizer, model=t5_model):
    input_ids = tokenizer(text_input, return_tensors="pt").input_ids  # Batch size 1
    outputs = model(input_ids=input_ids)
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states


# DiT block module
class DitBlock(nn.Module):
    """
    Pixart DiT block [
        embed - scale/shift - Multi self-attention - scale ->
        concat w/input - multi cross-attention -> concat w/attention tokens ->
        scale/shift - pointwise feedforward -> scale
    
        MLP for time embedding
    ]
    """
    
    def __init__(self):
        super().__init__()
        self.layernorm = nn.LayerNorm(, eps=1e-6)
        self.self_attention = None
        self.cross_attention = None
        self.pointwise_feedforward = None
        
    def _modulate(self, scale, shift):
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x

    def forward(self, x: torch.Tensor):
        
        return x
    
    
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
        output = attn_score @ v # multiply with value vectors
        
        output = rearrange(output, 'b h n e -> b n (h e)') # fold back to input shape
        output = self.output_project(output) # output projection
        
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