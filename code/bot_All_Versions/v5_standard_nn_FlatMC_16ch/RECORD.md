# Record

本质并无改变，保持cnn_Res_Mish_SE训练架构，仅仅只是为了充分运用botzone上的时间采用了小mc搜索

- **决策**: 纯NN argmax → **NN + FlatMC混合搜索**
  - NN候选排序 → Progressive Pruning → FlatMC(VHR) → Discard-Twice
  - 模拟预算 3000次/步, NN高置信度时减半
  - VHR: 短rollout 3步 + value head批量评估叶节点
  - 时间限制 4.5s (anytime)
- **编码层**: FeatureAgent + agent.py → **StateEncoder + GameState**
  - 直接从 GameState 编码 (16,4,9)
- **鸣牌**: 纯shanten比较 → **NN value-guided** 
- **__main__.py**: 为了适配mc使用GameState直接追踪牌局

