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
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.size()
        
        # Safe Linear QKV
        qkv = _safe_linear_forward(x, self.qkv_proj)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled Dot-Product Attention
        attn_scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        
        # Safe Linear Output
        output = _safe_linear_forward(attn_out, self.out_proj)
        return output

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
    """Residual block: Conv-GN-Mish-Conv-GN-SE-Add-Mish."""
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
    def __init__(self, in_channels=148):
        super().__init__()
        # Dimensions
        self.in_channels = in_channels
        hidden_dim = 256
        res_blocks = 16
        reduce_dim = 128
        fc_dim = 1024
        
        # CoordConv
        grid_y, grid_x = torch.meshgrid(torch.arange(4), torch.arange(9), indexing='ij')
        grid_y = (grid_y.float() / 3.0) * 2.0 - 1.0
        grid_x = (grid_x.float() / 8.0) * 2.0 - 1.0
        self.register_buffer('coord_grid', torch.stack([grid_y, grid_x], dim=0).unsqueeze(0))

        # Stem
        self._conv_init = nn.Conv2d(in_channels + 2, hidden_dim, kernel_size=3, padding=1, bias=False)
        self._gn_init = nn.GroupNorm(16, hidden_dim)
        self._act = nn.Mish(inplace=True)

        # Residual tower
        self._res_blocks = self._make_layer(ResidualBlock, hidden_dim, res_blocks)
        
        # Dimension reduction
        self._conv_reduce = nn.Conv2d(hidden_dim, reduce_dim, kernel_size=1, bias=False)
        self._gn_reduce = nn.GroupNorm(8, reduce_dim)

        # BiLSTM
        # 将降维后的空间特征展平为 36 的序列，提取顺子/刻子级别的连贯信息
        self._lstm = nn.LSTM(
            input_size=reduce_dim, 
            hidden_size=reduce_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Self-attention
        lstm_out_dim = reduce_dim * 2  # Bidirectional doubles the dimension
        self._attention = SafeSelfAttention(embed_dim=lstm_out_dim, num_heads=8)
        self._attn_norm = nn.LayerNorm(lstm_out_dim)

        self._flatten = nn.Flatten()
        
        # flat_dim = 36 * lstm_out_dim
        flat_dim = 36 * lstm_out_dim

        # Policy and value heads
        self._logits = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(flat_dim, fc_dim),
            nn.Mish(inplace=True),
            nn.Linear(fc_dim, 235)
        )

        self._value_branch = nn.ModuleList([
            nn.Dropout(0.1),
            nn.Linear(flat_dim, fc_dim),
            nn.Mish(inplace=True),
            nn.Linear(fc_dim, 1)
        ])

        # Initialize
        self._apply_orthogonal_init()

    def _make_layer(self, block, channels, num_blocks):
        """Build a sequence of residual blocks."""
        layers = []
        for _ in range(num_blocks):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def _apply_orthogonal_init(self):
        """Apply orthogonal initialization."""
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

        # Near-uniform initial policy
        if isinstance(self._logits[-1], nn.Linear):
            nn.init.orthogonal_(self._logits[-1].weight, gain=0.01)
            nn.init.constant_(self._logits[-1].bias, 0)

        if isinstance(self._value_branch[-1], nn.Linear):
            nn.init.orthogonal_(self._value_branch[-1].weight, gain=1.0)
            nn.init.constant_(self._value_branch[-1].bias, 0)

    def forward(self, input_dict: dict):
        obs = input_dict["observation"].float()
        batch_size = obs.size(0)
        
        # --- Spatial features ---
        coords = self.coord_grid.expand(batch_size, -1, -1, -1)
        x = torch.cat([obs, coords], dim=1)

        x = self._act(self._gn_init(self._conv_init(x)))
        x = self._res_blocks(x)  # (B, 256, 4, 9)
        
        x = self._conv_reduce(x)
        x = self._gn_reduce(x)
        x = self._act(x)         # (B, 128, 4, 9)

        # --- Sequence and attention ---
        # Reshape to sequence
        b, c, h, w = x.size()
        seq_in = x.view(b, c, h * w).permute(0, 2, 1)
        
        # BiLSTM
        lstm_out, _ = self._lstm(seq_in) # (B, 36, 256)
        
        # Self-attention
        attn_out = self._attention(lstm_out)
        
        # Residual + LayerNorm
        seq_out = self._attn_norm(lstm_out + attn_out)
        seq_out = self._act(seq_out)
        
        # --- Policy/value heads ---
        flat_out = self._flatten(seq_out) # (B, 9216)

        # Policy
        x_logits = flat_out
        for layer in self._logits:
            if isinstance(layer, nn.Linear):
                x_logits = _safe_linear_forward(x_logits, layer)
            else:
                x_logits = layer(x_logits)
        logits = x_logits

        # Mask invalid actions
        mask = input_dict["action_mask"].float()
        masked_logits = torch.where(
            mask > 0.5, 
            logits, 
            torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
        )

        # Value
        v_hidden = flat_out
        for layer in self._value_branch:
            if isinstance(layer, nn.Linear):
                v_hidden = _safe_linear_forward(v_hidden, layer)
            else:
                v_hidden = layer(v_hidden)
        value = v_hidden

        return masked_logits, value

