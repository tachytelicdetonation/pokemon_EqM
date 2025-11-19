# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings



#################################################################################
#                                 Core EqM Model                                #
#################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1_g = nn.Linear(in_features, hidden_features)
        self.fc1_x = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_g = self.fc1_g(x)
        x_x = self.fc1_x(x)
        x = self.act(x_g) * x_x
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DifferentialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Differential Attention requires two sets of Q and K to compute the difference
        # We project to 2 * dim for Q and K
        self.q_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Learnable lambda parameter for the difference
        # Initialized to a value that makes sense (e.g. corresponding to lambda ~ 0.8)
        self.lambda_p = nn.Parameter(torch.zeros(num_heads, 1, 1) + 0.0) # exp(0) = 1.0, or we can tune init
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # Project Q, K, V
        # q: (B, N, 2*C) -> (B, N, 2, H, D) -> (2, B, H, N, D)
        q = self.q_proj(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k_proj(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (B, H, N, D)

        q1, q2 = q[0], q[1] # (B, H, N, D)
        k1, k2 = k[0], k[1] # (B, H, N, D)
        
        # Flash Attention for both maps
        # Differential Attention: Attn = Softmax(Q1K1) - lambda * Softmax(Q2K2)
        # Output = Attn @ V = Softmax(Q1K1)@V - lambda * Softmax(Q2K2)@V
        # This allows us to use Flash Attention implementation for both terms!
        
        out1 = torch.nn.functional.scaled_dot_product_attention(
            q1, k1, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out2 = torch.nn.functional.scaled_dot_product_attention(
            q2, k2, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        
        # Calculate lambda = exp(lambda_p) - 1 + 1? Or just exp.
        # Paper usually uses exp to ensure positivity.
        # Let's use a simple learnable scalar.
        lam = torch.exp(self.lambda_p) # (H, 1, 1)
        
        # Combine
        out = out1 - lam * out2
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = DifferentialAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # Approx GELU is not needed for SwiGLU, but keeping structure similar
        self.mlp = SwiGLU(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


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
        ebm='none'
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.uncond = uncond
        self.ebm = ebm

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x0, t, y, return_act=False, get_energy=False, train=False):
        """
        Forward pass of EqM.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if self.ebm != 'none':
            x0.requires_grad_(True)
        if self.uncond: # removes noise/time conditioning by setting to 0
            t = torch.zeros_like(t)
        act = []
        x = self.x_embedder(x0) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
            act.append(x)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)

        # explicit energy
        E=0
        if self.ebm == 'l2':
            E = -torch.sum(x**2, dim=(1,2,3))/2
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()],[x0],create_graph=train)[0] 
        if self.ebm == 'dot':
            E = torch.sum(x*x0, dim=(1,2,3))
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()],[x0],create_graph=train)[0]
        if self.ebm == 'mean':
            E = torch.sum(x*x0, dim=(1,2,3))
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()],[x0],create_graph=train)[0]           
        if get_energy:
            return x, -E
        if return_act: 
            return x, act
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, return_act=False, get_energy=False, train=False):
        """
        Forward pass of EqM, but also batches the uncondional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
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
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        if get_energy:
            return torch.cat([eps, rest], dim=1), E
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   EqM Configs                                  #
#################################################################################

def EqM_XL_2(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def EqM_XL_4(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def EqM_XL_8(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def EqM_L_2(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def EqM_L_4(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def EqM_L_8(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def EqM_B_2(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def EqM_B_4(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def EqM_B_8(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def EqM_S_2(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def EqM_S_4(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def EqM_S_8(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


EqM_models = {
    'EqM-XL/2': EqM_XL_2,  'EqM-XL/4': EqM_XL_4,  'EqM-XL/8': EqM_XL_8,
    'EqM-L/2':  EqM_L_2,   'EqM-L/4':  EqM_L_4,   'EqM-L/8':  EqM_L_8,
    'EqM-B/2':  EqM_B_2,   'EqM-B/4':  EqM_B_4,   'EqM-B/8':  EqM_B_8,
    'EqM-S/2':  EqM_S_2,   'EqM-S/4':  EqM_S_4,   'EqM-S/8':  EqM_S_8,
}
