import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from .layers import (
    TimestepEmbedder, LabelEmbedder, SiTBlock, FinalLayer,
    get_2d_sincos_pos_embed
)

class EqM(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        uncond=True,
        energy_head='implicit',
        kv_heads=None,
        use_qk_norm=True,
        head_drop=0.0,
        window_size=None,
        use_abs_pos_emb=False,
        use_rope=True,
        rope_base=10000,
        use_liere=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_abs_pos_emb = use_abs_pos_emb
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.sigreg_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        block_kwargs = dict(
            kv_heads=kv_heads,
            use_qk_norm=use_qk_norm,
            head_drop=head_drop,
            window_size=window_size,
            use_rope=use_rope,
            rope_base=rope_base,
            use_liere=use_liere,
        )
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.uncond = uncond
        self.energy_head = energy_head

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # Backward-compatible alias for legacy configs/tests
    @property
    def ebm(self):
        return self.energy_head

    @ebm.setter
    def ebm(self, value):
        self.energy_head = value

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x0, t, y, return_act=False, get_energy=False, train=False):
        use_energy = self.energy_head in ['dot', 'l2']
        if use_energy:
            x0.requires_grad_(True)
        if self.uncond:
            t = torch.zeros_like(t)
        act = []
        x = self.x_embedder(x0)
        if self.use_abs_pos_emb:
            x = x + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
            act.append(x)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)

        energy = None
        if self.energy_head == 'l2':
            energy = -0.5 * torch.sum(x ** 2, dim=(1, 2, 3))
        elif self.energy_head in ['dot', 'mean']:
            energy = torch.sum(x * x0, dim=(1, 2, 3))

        if use_energy and energy is not None and energy.requires_grad:
            x = torch.autograd.grad([energy.sum()], [x0], create_graph=train)[0]

        if get_energy:
            return x, None if energy is None else -energy
        if return_act: 
            return x, act
        return x

    def sigreg_embedding(self, penultimate_tokens: torch.Tensor):
        """Pool penultimate tokens and pass through a lightweight head for SIGReg."""
        pooled = penultimate_tokens.mean(dim=1)
        return self.sigreg_head(pooled)

    def forward_with_cfg(self, x, t, y, cfg_scale, return_act=False, get_energy=False, train=False):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, return_act=return_act, get_energy=get_energy, train=train)
        if get_energy:
            x, E = model_out
            model_out=x
        if return_act:
            act = model_out[1]
            model_out = model_out[0]
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1), act
        
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        if get_energy:
            return torch.cat([eps, rest], dim=1), E
        return torch.cat([eps, rest], dim=1)
