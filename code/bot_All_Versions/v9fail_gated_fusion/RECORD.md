# Record

**这是第二个失败的版本。**

但是可以认为是双网络融合尝试的开始。

刚开始自认为很有用的辅助头也因为实现麻烦、细想不能深究原因而省略

---

可学习门控融合 BiLSTM+Attn+ CNN双路径。

---

## model_vultra.py：标量门控双路径

```
CoordConv → DeepStem (4层, BN, 128→192→256→320ch)
    │
    ├── Spatial Path (v5v6 纯卷积): 12×ResBlock(BN) → GAP → (B,320)
    │
    ├── Temporal Path (vfinal BiLSTM+Attn): Flatten → BiLSTM(320→640) → SelfAttn(8h) → GAP → Proj→320
    │
    └── Gate Network: GAP(DeepStem) → FC→64→1→sigmoid → α (B,1)

Fused = α ⊗ spatial + (1-α) ⊗ temporal   (逐通道加权)

    ├── Policy Head: Fused → Dropout(0.1) → FC(512) → 235
    ├── Value Head:  Fused → Dropout(0.1) → FC(512) → 1
    └── Aux Heads (挂在 DeepStem-GAP):
        ├── win_prob
        ├── opp_action (5分类)
        ├── phase (4分类)
        └── shanten
```
