# Record

这是第一个失败的版本。训练一会就loss爆炸。

双塔解耦 + 多任务辅助。基于 224ch V3 特征，尝试将策略和价值网络完全分离，同时引入4个辅助任务。

但事实上设计美好，真正开始训练loss直接开始爆炸，估计原因是

- 双塔独立优化导致 Policy/Value 不同步
- 辅助任务标签设计事实上非常随意
- 多损失权重**根本没时间**调优，辅助损失震荡严重

---

## model.py：DualTower + MultiTask

```
Shared Stem (2层, GN, 128→256ch)
  ├── Policy Tower (12×SEResBlock, GN, SE reduction=8)
  │     └── Policy Head: 1×1Conv→128→64 → Flatten(2304) → FC(512) → 235
  │
  └── Value Tower (8×SEResBlock, GN, SE reduction=8)
        ├── Value Head: GAP → FC(512) → 1
        ├── Aux1: win_prob (Sigmoid)
        ├── Aux2: opp_action (5分类)
        ├── Aux3: phase (4分类)
        └── Aux4: shanten (回归 0-8)
```

本来的思考：

- Policy Tower 更深(12层)：手牌模式复杂需要深层特征
- Value Tower 更浅(8层)：全局评估不需要太深
- 辅助任务挂在 Value Tower 后（局面判断类任务共享价值表示）

---

## sl_pretrain.py

- 双塔独立学习率：Policy 1.5e-3, Value 5e-4
- Multi-task aux losses (win_prob/phase/shanten)
- Value target = -log_prob of ground truth action
- Label Smoothing 0.1 + OneCycleLR + AdamW

