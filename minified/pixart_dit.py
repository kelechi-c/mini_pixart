import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed, Mlp
from .utils import config
from .dit_blocks import DitBlock, FinalLayer


## full/central Pixart model
class PixartDit(nn.Module):
    def __init__(
        self,
        n_layers: int = config.tr_blocks,
        input_size=32,
        in_channels=4,
        out_channels=3,
        config=config, 
        drop_path=0.0,
        return_sigma=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if return_sigma else in_channels
        self.patch_size = config.patch_size
        self.num_heads = config.attn_heads
        self.hidden_size = config.hidden_size

        self.patch_embed = PatchEmbed(input_size, patch_size=config.patch_size, in_chans=in_channels, embed_dim=self.hidden_size, bias=True)
        self.num_patches = self.patch_embed.num_patches
        self.base_size = input_size // self.patch_size
        self.time_embedder = None
        self.caption_embedder = None
        
        self.register_buffer('pos_embed', torch.zeros(1, self.num_patches, self.hidden_size))
        
        dit_blocks = [DitBlock for _ in range(n_layers)] # list of DiT blocks
        self.dit_layers = nn.Sequential(**dit_blocks)
        self.adaln_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )
        drop_path = [x.item() for x in torch.linspace(0, drop_path, n_layers)] # depth decay rule
        
        self.mlp_layer = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=out_channels
        )

    def forward(self, image: torch.Tensor, encoded_text: torch.Tensor, timestep: torch.Tensor, config=config) -> torch.Tensor:
        x_img, y_text = image.to(config.dtype), encoded_text.to(config.dtype)
        timestep = timestep.to(config.dtype)
        pos_embed = self.pos_embed.to(config.dtype)
        self.height = x_img.shape[-2] // self.patch_size
        self.width = x_img.shape[-1] // self.patch_size
        

# embeds tet conditioning vectors, token dropout
class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        self.project = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.register_buffer('y_embedding', nn.Parameter(torch.randn(token_num, in_channels) // in_channels**0.5))
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
    
    
    def forward(self, caption: torch.Tensor, training: bool, force_drop_ids=None):
        if training:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        
        if (training and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        
        caption = self.project(caption)
        
        return caption