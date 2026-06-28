"""国标麻将 Dual Path 模型 — 160通道"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def _safe_linear_forward(x, layer):
    # Work around MKLDNN primitive descriptor errors on aarch64.
    try:
        return layer(x)
    except RuntimeError as e:
        if x.device.type == 'cpu' and 'primitive descriptor' in str(e):
            return F.linear(x, layer.weight, layer.bias)
        raise e

class SafeSelfAttention(nn.Module):
    """Self-attention with safe linear forward (avoids aarch64 MKLDNN crash)."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.size()
        qkv = _safe_linear_forward(x, self.qkv_proj)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_out = (F.softmax(attn_scores, dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return _safe_linear_forward(attn_out, self.out_proj)


class FFN(nn.Module):
    """Transformer FFN: Linear → Mish → Linear"""
    def __init__(self, embed_dim, expansion=4):
        super().__init__()
        hidden = embed_dim * expansion
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.Mish(inplace=True)
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        return _safe_linear_forward(self.act(_safe_linear_forward(x, self.fc1)), self.fc2)


class TransformerBlock(nn.Module):
    """SelfAttention + FFN, each with residual + LayerNorm + Mish"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = SafeSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, expansion=4)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.act = nn.Mish(inplace=True)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.act(x)
        x = self.norm2(x + self.ffn(x))
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = _safe_linear_forward(y, self.fc[0])
        y = self.fc[1](y)
        y = _safe_linear_forward(y, self.fc[2])
        y = self.fc[3](y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Conv → GN → Mish → Conv → GN → SE → +x → Mish"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(16, channels)
        self.act = nn.Mish(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(16, channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.act(out)


class CNNModel(nn.Module):
    def __init__(self, in_channels=160, num_transformer_blocks=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_transformer_blocks = num_transformer_blocks
        fc_dim = 1024

        grid_y, grid_x = torch.meshgrid(torch.arange(4), torch.arange(9), indexing='ij')
        grid_y = (grid_y.float() / 3.0) * 2.0 - 1.0
        grid_x = (grid_x.float() / 8.0) * 2.0 - 1.0
        self.register_buffer('coord_grid', torch.stack([grid_y, grid_x], dim=0).unsqueeze(0))

        self._conv_init = nn.Conv2d(in_channels + 2, 256, kernel_size=3, padding=1, bias=False)
        self._gn_init = nn.GroupNorm(16, 256)
        self._act = nn.Mish(inplace=True)

        self._stage1 = nn.ModuleList([ResidualBlock(256) for _ in range(3)])

        self._trans1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 128), nn.Mish(inplace=True))

        self._stage2 = nn.ModuleList([ResidualBlock(128) for _ in range(3)])

        self._trans2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64), nn.Mish(inplace=True))

        self._stage3 = nn.ModuleList([ResidualBlock(64) for _ in range(3)])

        self._lstm_proj_in = nn.Linear(64, 128)
        self._lstm = nn.LSTM(
            input_size=128, hidden_size=128,
            num_layers=1, batch_first=True, bidirectional=True)
        # output: (B, 36, 256)  — 128×2 bidir
        self._lstm_proj_out = nn.Linear(256, 64)

        self._branch_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(9)])

        self._pos_embed = nn.Parameter(torch.randn(1, 36, 128) * 0.02)
        self._transformers = nn.ModuleList([
            TransformerBlock(embed_dim=128, num_heads=4)
            for _ in range(num_transformer_blocks)
        ])

        self._flatten = nn.Flatten()

        flat_dim = 36 * 128  # 4608
        self._logits = nn.Sequential(
            nn.Linear(flat_dim, fc_dim),
            nn.Mish(inplace=True),
            nn.Linear(fc_dim, 235))

        self._value_branch = nn.ModuleList([
            nn.Linear(flat_dim, fc_dim),
            nn.Mish(inplace=True),
            nn.Linear(fc_dim, 1)])

        self._apply_orthogonal_init()
        self._zero_branch_weights()

    def _zero_branch_weights(self):
        """将下路 CNN (_branch_blocks) 所有 Conv/Linear 权重置零。

        GroupNorm 保持默认 (weight=1, bias=0)，保证前向传播数值稳定。
        效果: 下路每个 ResidualBlock 退化为 Mish(residual)，卷积部分不贡献。
        """
        for module in self._branch_blocks.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_orthogonal_init(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

        if isinstance(self._logits[-1], nn.Linear):
            nn.init.orthogonal_(self._logits[-1].weight, gain=0.01)
            nn.init.constant_(self._logits[-1].bias, 0)
        if isinstance(self._value_branch[-1], nn.Linear):
            nn.init.orthogonal_(self._value_branch[-1].weight, gain=1.0)
            nn.init.constant_(self._value_branch[-1].bias, 0)

    def forward(self, input_dict: dict):
        obs = input_dict["observation"].float()
        B = obs.size(0)

        # --- Shared CNN ---
        coords = self.coord_grid.expand(B, -1, -1, -1)
        x = torch.cat([obs, coords], dim=1)
        x = self._act(self._gn_init(self._conv_init(x)))

        for blk in self._stage1: x = blk(x)          # (B,256,4,9)
        x = self._trans1(x)                           # (B,128,4,9)
        for blk in self._stage2: x = blk(x)           # (B,128,4,9)
        x = self._trans2(x)                           # (B,64,4,9)
        for blk in self._stage3: x = blk(x)           # (B,64,4,9)

        # --- 上路: BiLSTM ---
        seq_a = x.view(B, 64, 36).permute(0, 2, 1)   # (B,36,64)
        seq_a = _safe_linear_forward(seq_a, self._lstm_proj_in)  # (B,36,128)
        lstm_out, _ = self._lstm(seq_a)               # (B,36,256)
        lstm_out = _safe_linear_forward(lstm_out, self._lstm_proj_out)  # (B,36,64)

        # --- 下路: CNN 恒宽 64ch ---
        x_b = x
        for blk in self._branch_blocks: x_b = blk(x_b)  # (B,64,4,9)
        seq_b = x_b.view(B, 64, 36).permute(0, 2, 1)   # (B,36,64)

        # --- Fuse ---
        fused = torch.cat([lstm_out, seq_b], dim=-1)  # (B,36,128)
        fused = fused + self._pos_embed
        for blk in self._transformers:
            fused = blk(fused)                         # (B,36,128)

        flat_out = self._flatten(fused)               # (B,4608)

        # --- Heads ---
        x_logits = flat_out
        for layer in self._logits:
            if isinstance(layer, nn.Linear):
                x_logits = _safe_linear_forward(x_logits, layer)
            else:
                x_logits = layer(x_logits)

        mask = input_dict["action_mask"].float()
        masked_logits = torch.where(
            mask > 0.5, x_logits,
            torch.tensor(-1e8, device=x_logits.device, dtype=x_logits.dtype))

        vh = _safe_linear_forward(flat_out, self._value_branch[0])
        vh = self._value_branch[1](vh)
        value = _safe_linear_forward(vh, self._value_branch[2])

        return masked_logits, value
