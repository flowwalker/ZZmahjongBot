"""国标麻将 CNN + BiLSTM + SelfAttention 模型 — 224通道"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def _safe_linear(x, layer):
    try:
        return layer(x)
    except RuntimeError as e:
        if x.device.type == 'cpu' and 'primitive descriptor' in str(e):
            return F.linear(x, layer.weight, layer.bias)
        raise e


class SafeSelfAttention(nn.Module):
    """自注意力 — 原版设计 + dropout"""
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, L, D = x.size()
        qkv = _safe_linear(x, self.qkv_proj).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return _safe_linear(out, self.out_proj)


class SEBlock(nn.Module):
    """SE block with reduction=8"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.Mish(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """标准残差块 — 原版设计"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(16, channels)
        self.se = SEBlock(channels)
        self.act = nn.Mish(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        out = out + residual
        return self.act(out)


class CNNModel(nn.Module):
    """vfinal 原始架构 — 适配 224 通道"""
    def __init__(self, in_channels=224):
        super().__init__()
        self.in_channels = in_channels

        grid_y, grid_x = torch.meshgrid(torch.arange(4), torch.arange(9), indexing='ij')
        grid_y = (grid_y.float() / 3.0) * 2.0 - 1.0
        grid_x = (grid_x.float() / 8.0) * 2.0 - 1.0
        self.register_buffer('coord_grid', torch.stack([grid_y, grid_x], dim=0).unsqueeze(0))

        self._conv_init = nn.Conv2d(in_channels + 2, 256, kernel_size=3, padding=1, bias=False)
        self._gn_init = nn.GroupNorm(16, 256)
        self._res_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(16)])
        self._conv_reduce = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self._gn_reduce = nn.GroupNorm(16, 128)
        self._bilstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        # FIX: forget gate bias = 1.0 缓解梯度消失
        for name, param in self._bilstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        self._attention = SafeSelfAttention(256, num_heads=8, dropout=0.1)
        self._act = nn.Mish(inplace=True)

        flat_dim = 36 * 256
        self._policy_head = nn.ModuleList([
            nn.Dropout(0.1),
            nn.Linear(flat_dim, 1024),
            nn.LayerNorm(1024),
            nn.Mish(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(1024, 235)
        ])
        self._value_head = nn.ModuleList([
            nn.Dropout(0.1),
            nn.Linear(flat_dim, 1024),
            nn.LayerNorm(1024),
            nn.Mish(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(1024, 1)
        ])
        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.LSTM)):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
                    elif 'bias' in name and 'lstm' not in name.lower():
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self._policy_head[-1].weight, gain=0.01)
        nn.init.constant_(self._policy_head[-1].bias, 0)
        nn.init.orthogonal_(self._value_head[-1].weight, gain=1.0)
        nn.init.constant_(self._value_head[-1].bias, 0)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        batch_size = obs.size(0)
        coords = self.coord_grid.expand(batch_size, -1, -1, -1)
        x = torch.cat([obs, coords], dim=1)
        x = self._act(self._gn_init(self._conv_init(x)))
        for block in self._res_blocks:
            x = block(x)
        x = self._act(self._gn_reduce(self._conv_reduce(x)))
        x = x.view(batch_size, 36, 128)
        x, _ = self._bilstm(x)
        x = self._attention(x)
        x = x.view(batch_size, -1)

        # Policy
        px = x
        for layer in self._policy_head:
            if isinstance(layer, nn.Linear):
                px = _safe_linear(px, layer)
            else:
                px = layer(px)

        mask = input_dict["action_mask"].float()
        px = torch.where(mask > 0.5, px,
                         torch.tensor(-1e8, device=px.device, dtype=px.dtype))

        # Value
        vx = x
        for layer in self._value_head:
            if isinstance(layer, nn.Linear):
                vx = _safe_linear(vx, layer)
            else:
                vx = layer(vx)

        return px, vx
