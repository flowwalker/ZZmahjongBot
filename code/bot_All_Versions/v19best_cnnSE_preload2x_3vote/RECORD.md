# Record

 Botzone 部署提升方案：3 模型投票（虽然里面写了TTA，但实际因为bug，当时传上去无用，当然现在是对的）

## 策略

- 3 个同架构 CNNModel（SE Pyramid，160ch，v14 产出），分别加载 `model_vote1/2/3.pt`
- 全局平均 → argmax
- 变换间独立逆映射回原始动作空间

## 极大创新点：此处仅改变main接口的变化

多模型投票更加稳定。
