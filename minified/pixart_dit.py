import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed
from .utils import config
from .dit_blocks import DitBlock, FinalLayer, TimestepEmbedder, CaptionEmbedder, get_2d_sincos_pos_embed
from typing import TypeAlias

tensor: TypeAlias = torch.Tensor


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
        cond_drop_prob=0.1,
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
        self.time_embedder = TimestepEmbedder(self.hidden_size)
        self.caption_embedder = CaptionEmbedder(
            in_channels, self.hidden_size, 
            uncond_prob=cond_drop_prob
        )

        self.register_buffer('pos_embed', torch.zeros(1, self.num_patches, self.hidden_size))

        # diffusion transformer blocks
        self.dit_blocks = [DitBlock for _ in range(n_layers)] # list of DiT blocks
        self.dit_layers = nn.Sequential(**self.dit_blocks)

        self.time_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )
        drop_path = [x.item() for x in torch.linspace(0, drop_path, n_layers)] # depth decay rule

        self.mlp_layer = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=out_channels
        )

        self.init_weights()

    def forward(self, latent_image: torch.Tensor, encoded_text: torch.Tensor, timestep: torch.Tensor, config=config) -> torch.Tensor:
        x_img, y_text = latent_image.to(config.dtype), encoded_text.to(config.dtype)
        timestep = timestep.to(config.dtype)
        pos_embed = self.pos_embed.to(config.dtype)
        self.height = x_img.shape[-2] // self.patch_size
        self.width = x_img.shape[-1] // self.patch_size

        # patchify and add pos embed for images
        x = self.patch_embed(x_img) + pos_embed
        print(f"patched shape {x.shape}")

        # prepart timestep
        time_embed = self.time_embedder(timestep.to(x.dtype))
        print(f"time_embed shape {x.shape}")

        time_cond = self.time_block(time_embed)
        print(f"time_cond shape {time_cond.shape}")

        text_cond = self.caption_embedder(y_text) # embed text caption
        print(f"text_cond shape {text_cond.shape}")

        # run through all the dit blocks
        x = self.dit_layers(x, text_cond, time_cond)
        print(f"dit output shape {x.shape}")

        out_image = self.unpatchify(x) # return as
        print(f"out_image shape {out_image.shape}")

        return out_image

    def unpatchify(self, x: tensor):
        out_ch = self.out_channels
        p_size = self.patch_embed.patch_size[0]
        height = width = int(x.shape[1] ** 0.5)

        assert height * width == x.shape[1]

        x = x.reshape(shape=(x.shape[0], height, width, p_size, p_size, out_ch))
        print(f"pre-einsum shape {x.shape}")

        x = torch.einsum('nhwpqc -> nchpwq', x)
        print(f"after einsum shape {x.shape}")

        out_shape = (x.shape[0], out_ch, height * p_size, height * p_size)
        x = x.reshape(out_shape)

        return x 

    def init_weights(self):
        # for transformer blocks
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # init and freeze pos_embed by sincos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches ** 0.5), 
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # init patch embed
        patch_weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(patch_weight.view([patch_weight.shape[0], -1]))

        # inti timestep layers
        nn.init.normal_(self.timestep_embedder.mlp_layer[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp_layer[2].weight, std=0.02)
        nn.init.normal_(self.time_block[1].weight, std=0.02)

        # init caption embeds
        nn.init.normal_(self.caption_embedder.project.fc1.weight, std=0.02)
        nn.init.normal_(self.caption_embedder.project.fc2.weight, std=0.02)

        # apply adaLN zero init for DiT blocks
        for layer in self.dit_layers:
            nn.init.constant_(layer.cross_attention.output_project.weight, std=0.02)
            nn.init.constant_(layer.cross_attention.output_project.bias, std=0.02)

        # zero init output layers
        nn.init.constant_(self.mlp_layer.linear.weight, 0)
        nn.init.constant_(self.mlp_layer.linear.bias, 0)
