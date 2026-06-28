"""国标麻将 CNN + SE 模型 — 160通道"""

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

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
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
    def __init__(self, in_channels=160):
        super().__init__()
        self.in_channels = in_channels
        fc_dim = 1024

        # CoordConv
        grid_y, grid_x = torch.meshgrid(torch.arange(4), torch.arange(9), indexing='ij')
        grid_y = (grid_y.float() / 3.0) * 2.0 - 1.0
        grid_x = (grid_x.float() / 8.0) * 2.0 - 1.0
        self.register_buffer('coord_grid', torch.stack([grid_y, grid_x], dim=0).unsqueeze(0))

        # 2. Stem: 160+2 → 256
        self._conv_init = nn.Conv2d(in_channels + 2, 256, kernel_size=3, padding=1, bias=False)
        self._gn_init = nn.GroupNorm(16, 256)
        self._act = nn.Mish(inplace=True)

        # 3. Stage1: 3× ResBlock-SE @256ch
        self._stage1 = nn.ModuleList([ResidualBlock(256) for _ in range(3)])

        # 4. Transition1: 256 → 128
        self._trans1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.Mish(inplace=True),
        )

        # 5. Stage2: 3× ResBlock-SE @128ch
        self._stage2 = nn.ModuleList([ResidualBlock(128) for _ in range(3)])

        # 6. Transition2: 128 → 64
        self._trans2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.Mish(inplace=True),
        )

        # 7. Stage3: 3× ResBlock-SE @64ch
        self._stage3 = nn.ModuleList([ResidualBlock(64) for _ in range(3)])

        self._flatten = nn.Flatten()

        # 8. Policy Head: 64×4×9=2304 → 1024 → 235
        flat_dim = 64 * 4 * 9
        self._logits = nn.Sequential(
            nn.Linear(flat_dim, fc_dim),
            nn.Mish(inplace=True),
            nn.Linear(fc_dim, 235)
        )

        # 9. Value Head
        self._value_branch = nn.ModuleList([
            nn.Linear(flat_dim, fc_dim),
            nn.Mish(inplace=True),
            nn.Linear(fc_dim, 1)
        ])

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if isinstance(self._logits[-1], nn.Linear):
            nn.init.orthogonal_(self._logits[-1].weight, gain=0.01)
            nn.init.constant_(self._logits[-1].bias, 0)

        if isinstance(self._value_branch[-1], nn.Linear):
            nn.init.orthogonal_(self._value_branch[-1].weight, gain=1.0)
            nn.init.constant_(self._value_branch[-1].bias, 0)

    def forward(self, input_dict: dict):
        obs = input_dict["observation"].float()
        batch_size = obs.size(0)

        coords = self.coord_grid.expand(batch_size, -1, -1, -1)
        x = torch.cat([obs, coords], dim=1)

        x = self._act(self._gn_init(self._conv_init(x)))

        for block in self._stage1:
            x = block(x)                     # (B, 256, 4, 9)
        x = self._trans1(x)                  # (B, 128, 4, 9)
        for block in self._stage2:
            x = block(x)                     # (B, 128, 4, 9)
        x = self._trans2(x)                  # (B, 64, 4, 9)
        for block in self._stage3:
            x = block(x)                     # (B, 64, 4, 9)

        flat_out = self._flatten(x)          # (B, 2304)

        # Policy
        x_logits = flat_out
        for layer in self._logits:
            if isinstance(layer, nn.Linear):
                x_logits = _safe_linear_forward(x_logits, layer)
            else:
                x_logits = layer(x_logits)

        mask = input_dict["action_mask"].float()
        masked_logits = torch.where(
            mask > 0.5,
            x_logits,
            torch.tensor(-1e8, device=x_logits.device, dtype=x_logits.dtype)
        )

        # Value
        v_hidden = _safe_linear_forward(flat_out, self._value_branch[0])
        v_hidden = self._value_branch[1](v_hidden)
        value = _safe_linear_forward(v_hidden, self._value_branch[2])

        return masked_logits, value
