"""国标麻将 Cross-Attention 融合模型."""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def _safe_linear(x, layer):
    try: return layer(x)
    except RuntimeError as e:
        if x.device.type == 'cpu' and 'primitive descriptor' in str(e):
            return F.linear(x, layer.weight, layer.bias)
        raise e


class ResBlock(nn.Module):
    """BN残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.Mish(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class SafeSelfAttention(nn.Module):
    """安全自注意力 — 回避 Botzone aarch64 MKLDNN 崩溃"""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.size()
        qkv = _safe_linear(x, self.qkv_proj).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = _safe_linear(out, self.out_proj)
        return self.norm(x + out)


class CrossAttentionFusion(nn.Module):
    """3分支Cross-Attention融合 (精简版)"""
    def __init__(self, d_model=192, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_global, key_s, val_s, key_t, val_t, key_g, val_g):
        B = query_global.size(0)
        q = _safe_linear(query_global, self.q_proj).unsqueeze(1)
        keys = torch.cat([key_s, key_t, key_g], dim=1)
        values = torch.cat([val_s, val_t, val_g], dim=1)
        k = _safe_linear(keys, self.k_proj)
        v = _safe_linear(values, self.v_proj)
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, 1, self.d_model)
        out = _safe_linear(out.squeeze(1), self.out_proj[0])
        out = self.out_proj[1](out)
        attn_weights = attn.mean(dim=1).squeeze(1)
        return out, attn_weights


class MonsterModel(nn.Module):
    def __init__(self, in_channels=224):
        super().__init__()

        # CoordConv
        grid_y, grid_x = torch.meshgrid(torch.arange(4), torch.arange(9), indexing='ij')
        grid_y = (grid_y.float() / 3.0) * 2.0 - 1.0
        grid_x = (grid_x.float() / 8.0) * 2.0 - 1.0
        self.register_buffer('coord_grid', torch.stack([grid_y, grid_x], dim=0).unsqueeze(0))

        # DeepStem: 2层 128→192
        self._stem = nn.Sequential(
            nn.Conv2d(in_channels + 2, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.Mish(inplace=True),
            nn.Conv2d(128, 192, 3, padding=1, bias=False), nn.BatchNorm2d(192), nn.Mish(inplace=True),
        )

        # Spatial Tower: 6×ResBlock(192ch)
        self._spatial_blocks = nn.ModuleList([ResBlock(192) for _ in range(6)])

        # Temporal Tower: BiLSTM(128→256) + SelfAttn(256,4h) + Proj
        self._temporal_bilstm = nn.LSTM(192, 128, batch_first=True, bidirectional=True)
        for name, param in self._temporal_bilstm.named_parameters():
            if 'bias' in name: param.data[param.size(0)//4:param.size(0)//2].fill_(1.0)
        self._temporal_attn = SafeSelfAttention(256, num_heads=4)
        self._temporal_proj = nn.Sequential(nn.Linear(256, 192), nn.Mish(inplace=True))

        # Cross-Attention Fusion: 192ch, 4heads
        self._fusion = CrossAttentionFusion(d_model=192, num_heads=4)

        # 输出头 (缩减FC维度)
        self._policy = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(192, 256), nn.Mish(inplace=True),
            nn.Linear(256, 235)
        )
        self._value = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(192, 128), nn.Mish(inplace=True),
            nn.Linear(128, 1)
        )

        # 辅助头 (挂在 Stem-GAP 上)
        self._stem_gap = nn.AdaptiveAvgPool2d(1)
        self._aux_win_prob = nn.Sequential(nn.Linear(192, 64), nn.Mish(inplace=True), nn.Linear(64, 1), nn.Sigmoid())
        self._aux_opp_action = nn.Sequential(nn.Linear(192, 64), nn.Mish(inplace=True), nn.Linear(64, 5))
        self._aux_phase = nn.Sequential(nn.Linear(192, 64), nn.Mish(inplace=True), nn.Linear(64, 4))
        self._aux_shanten = nn.Sequential(nn.Linear(192, 64), nn.Mish(inplace=True), nn.Linear(64, 1))

        self._apply_init()

    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self._policy[-1].weight, gain=0.01); nn.init.constant_(self._policy[-1].bias, 0)
        nn.init.orthogonal_(self._value[-1].weight, gain=1.0); nn.init.constant_(self._value[-1].bias, 0)

    def forward(self, input_dict, return_aux=False):
        obs = input_dict["observation"].float()
        B = obs.size(0)

        # CoordConv + DeepStem
        coords = self.coord_grid.expand(B, -1, -1, -1)
        x = torch.cat([obs, coords], dim=1)
        stem_out = self._stem(x)  # (B, 192, 4, 9)
        stem_gap = self._stem_gap(stem_out).view(B, -1)  # (B, 192)

        # Spatial Path
        sp = stem_out
        for block in self._spatial_blocks: sp = block(sp)
        sp_feat = sp.view(B, 192, 36).permute(0, 2, 1)  # (B, 36, 192)
        key_s = sp_feat; val_s = sp_feat

        # Temporal Path
        tp = stem_out.view(B, 192, 36).permute(0, 2, 1)  # (B, 36, 192)
        tp, _ = self._temporal_bilstm(tp)  # (B, 36, 256)
        tp = self._temporal_attn(tp)  # (B, 36, 256)
        tp_proj = self._temporal_proj(tp)  # (B, 36, 192)
        key_t = tp_proj; val_t = tp_proj

        # Global Path
        global_feat = stem_gap.unsqueeze(1)  # (B, 1, 192) 单token
        key_g = global_feat; val_g = global_feat

        # Cross-Attention Fusion
        sp_gap = sp_feat.mean(dim=1)  # (B, 192)
        fused, attn_weights = self._fusion(sp_gap, key_s, val_s, key_t, val_t, key_g, val_g)

        # 主输出
        logits = self._policy(fused)
        mask = input_dict["action_mask"].float()
        logits.masked_fill_(mask <= 0.5, -1e8)
        value = self._value(fused)

        if not return_aux or not self.training:
            return logits, value

        aux = {
            'win_prob': self._aux_win_prob(stem_gap),
            'opp_action': self._aux_opp_action(stem_gap),
            'phase': self._aux_phase(stem_gap),
            'shanten': self._aux_shanten(stem_gap),
            'attn_weights': attn_weights,
        }
        return logits, value, aux
