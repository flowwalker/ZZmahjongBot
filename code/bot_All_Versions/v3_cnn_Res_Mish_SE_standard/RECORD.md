# Record

### 模型 — 完整重写
- **CoordConv**: 输入拼接 2 通道坐标网格 (y, x 归一化到 [-1, 1])
- **ResNet-SE**: 4 个残差块，每个含 GroupNorm(4) + Mish + SE 通道注意力
- **SEBlock**: Squeeze-and-Excitation (全局均值池化 → Linear → Mish → Linear → Sigmoid)
- ReLU → **Mish**
- **归一化**: 无 → **GroupNorm(4)**
- **初始化**: Kaiming → **Orthogonal** (gain=√2, 策略头最终层 gain=0.01)
- **Mask**: `log(mask)` inf → `torch.where(mask>0.5, logits, -1e8)`
-  `_safe_linear_forward()` 

### 预处理
- 修复 Ignore 子句处理 (Peng/Gang/Hu 后的其他玩家重新动作)

### 数据集
- 按需加载 (bisect_right) → **全量内存预加载** (连续np.concatenate)

### SL 
- 新增 SIGINT/SIGTERM 信号处理，Ctrl+C 安全中断
- MPS 优先于 CUDA（本地Mac）
- 保存最佳模型同时导出纯权重 (model_weights.pt)

### __main__.py
- 环境变量限制多线程 (OMP/MKL/OPENBLAS = 1)
- 碰/吃后出牌 → 二次决策 (Peng/Chi 后的 Play 需要模型推理)
- 杠牌状态跟踪 (angang/zimo 变量)

