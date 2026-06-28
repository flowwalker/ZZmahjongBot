# Record

这是第三个失败的版本。

也是真正训练过几个epoch发现loss爆炸的版本。

---

猜测是融合机制过于复杂，以及跨口注意力的非稳定性造成了如何训都acc只有百分之三十几的惨状。

---

分支 Cross-Attention 融合 + 激进裁减至 ~5.4M。

---

## model_monster.py：Cross-Attention 3分支融合

在 v9 基础上，用 multi-source cross-attention 替代标量门控：

```
CoordConv → DeepStem (2层, 128→192ch)   [原版 4层 320ch 砍半]
    │
    ├─ Spatial Tower:  6×ResBlock(192) → Flatten → (B,36,192)  [K_s, V_s]
    ├─ Temporal Tower:  Flatten → BiLSTM(128→256) → SelfAttn(4h) → (B,36,192)  [K_t, V_t]
    └─ Global:          GAP(Stem) → (B,192) → expand   [K_g, V_g]
        │
    Query = GAP(Spatial) → FC → (B, 1, 192)
    Keys/Values = Concat[K_s, K_t, K_g] → (B, 108, 192)
    Attn = softmax(Query @ Keys^T / √d) → (B, 1, 108)
    Fused = Attn @ Values → (B, 192)

    ├── Policy Head: (B,192) → FC(256) → 235
    └── Value Head:  (B,192) → FC(128) → 1
```
