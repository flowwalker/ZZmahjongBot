"""国标麻将 Gated Fusion 模型."""

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
    """BN残差块 — GN→BN（batch_size=4096统计量充足）"""
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
    def __init__(self, d_model, num_heads=8, dropout=0.1):
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


class VultraModel(nn.Module):
    def __init__(self, in_channels=224):
        super().__init__()

        # CoordConv
        grid_y, grid_x = torch.meshgrid(torch.arange(4), torch.arange(9), indexing='ij')
        grid_y = (grid_y.float() / 3.0) * 2.0 - 1.0
        grid_x = (grid_x.float() / 8.0) * 2.0 - 1.0
        self.register_buffer('coord_grid', torch.stack([grid_y, grid_x], dim=0).unsqueeze(0))

        self._stem = nn.Sequential(
            nn.Conv2d(in_channels + 2, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.Mish(inplace=True),
            nn.Conv2d(128, 192, 3, padding=1, bias=False), nn.BatchNorm2d(192), nn.Mish(inplace=True),
            nn.Conv2d(192, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.Mish(inplace=True),
            nn.Conv2d(256, 320, 3, padding=1, bias=False), nn.BatchNorm2d(320), nn.Mish(inplace=True),
        )

        self._spatial_blocks = nn.ModuleList([ResBlock(320) for _ in range(12)])
        self._spatial_gap = nn.AdaptiveAvgPool2d(1)

        self._temporal_bilstm = nn.LSTM(input_size=320, hidden_size=320, batch_first=True, bidirectional=True)
        # forget gate bias = 1.0
        for name, param in self._temporal_bilstm.named_parameters():
            if 'bias' in name: param.data[param.size(0)//4:param.size(0)//2].fill_(1.0)
        self._temporal_attn = SafeSelfAttention(640, num_heads=8, dropout=0.1)  # 320*2=640
        self._temporal_gap = nn.AdaptiveAvgPool1d(1)
        self._temporal_proj = nn.Sequential(
            nn.Linear(640, 320), nn.Mish(inplace=True)
        )

        self._gate_gap = nn.AdaptiveAvgPool2d(1)
        self._gate_fc = nn.Sequential(
            nn.Linear(320, 64), nn.Mish(inplace=True),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        flat_dim = 320
        self._policy = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(flat_dim, 512), nn.Mish(inplace=True),
            nn.Linear(512, 235)
        )
        self._value = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(flat_dim, 512), nn.Mish(inplace=True),
            nn.Linear(512, 1)
        )

        stem_gap_dim = 320
        self._aux_win_prob = nn.Sequential(nn.Linear(stem_gap_dim, 64), nn.Mish(inplace=True), nn.Linear(64, 1), nn.Sigmoid())
        self._aux_opp_action = nn.Sequential(nn.Linear(stem_gap_dim, 64), nn.Mish(inplace=True), nn.Linear(64, 5))
        self._aux_phase = nn.Sequential(nn.Linear(stem_gap_dim, 64), nn.Mish(inplace=True), nn.Linear(64, 4))
        self._aux_shanten = nn.Sequential(nn.Linear(stem_gap_dim, 64), nn.Mish(inplace=True), nn.Linear(64, 1))

        self._apply_init()

    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self._policy[-1].weight, gain=0.01); nn.init.constant_(self._policy[-1].bias, 0)
        nn.init.orthogonal_(self._value[-1].weight, gain=1.0); nn.init.constant_(self._value[-1].bias, 0)

    def forward(self, input_dict, return_aux=False):
        obs = input_dict["observation"].float()
        B = obs.size(0)

        # CoordConv + DeepStem
        coords = self.coord_grid.expand(B, -1, -1, -1)
        x = torch.cat([obs, coords], dim=1)
        stem_out = self._stem(x)  # (B, 320, 4, 9)

        sp = stem_out
        for block in self._spatial_blocks: sp = block(sp)
        sp_global = self._spatial_gap(sp).view(B, -1)  # (B, 320)

        tp = stem_out.view(B, 320, 36).permute(0, 2, 1)  # (B, 36, 320)
        tp, _ = self._temporal_bilstm(tp)  # (B, 36, 640)
        tp = self._temporal_attn(tp)  # (B, 36, 640)
        tp = self._temporal_gap(tp.permute(0, 2, 1)).view(B, -1)  # (B, 640)
        tp_global = self._temporal_proj(tp)  # (B, 320)

        gate_feat = self._gate_gap(stem_out).view(B, -1)  # (B, 320)
        alpha = self._gate_fc(gate_feat)  # (B, 1)

        fused = alpha * sp_global + (1 - alpha) * tp_global  # (B, 320)

        logits = self._policy(fused)
        mask = input_dict["action_mask"].float()
        logits = torch.where(mask > 0.5, logits,
                             torch.tensor(-1e8, device=logits.device, dtype=logits.dtype))
        value = self._value(fused)

        if not return_aux or not self.training:
            return logits, value

        aux = {
            'win_prob': self._aux_win_prob(gate_feat),
            'opp_action': self._aux_opp_action(gate_feat),
            'phase': self._aux_phase(gate_feat),
            'shanten': self._aux_shanten(gate_feat),
            'gate_alpha': alpha,  # 可观察门控值
        }
        return logits, value, aux
