# Record

重新开始，纯 CNN 版本

目的是restart，重新先利用速度最快的pure cnn来尝试做一些关于feature和数据增强的实验，因为目前已经知道SE和lstm都实质有效。

## 定位

在 v6 (148ch + BiLSTM + SelfAttention + SE) 的基础上：
- **特征扩展**：148ch → 155ch，增加 7 个标量通道（当前牌 one-hot、牌源方、牌墙最后一张标志、刚杠标志、对手手牌数×3）

- 回退到纯 ResBlock CNN

  相当于**减法实验**试试，也主要是想走出连续三次大换血失败——v6/v7 加了 BiLSTM+Attention 真的有用吗？还是纯 CNN + 更丰富特征就够了？

## 架构

```
Stem:  155ch → 256 → 256 (Conv-BN-ReLU)
Stage1: 8× ResBlock(256)     (Conv-BN-ReLU-Conv-BN + skip → ReLU)
Trans:  256 → 128 (Conv-BN-ReLU)
Stage2: 8× ResBlock(128)
Head:   Flatten(128×36=4608) → Linear(4608, 235)
```

## 特征 (155ch)

v6 的 148ch + 7 个标量通道：
- Ch 0-147: 同 v6（手牌4ch + 可见牌4ch + 鸣牌16ch + 风位8ch + 牌墙1ch + 弃牌112ch）
- Ch 148: 当前待决策牌 one-hot
- Ch 149: 牌源方归一化 (tileFrom/3)
- Ch 150: 牌墙最后一张标志
- Ch 151: 刚杠标志
- Ch 152-154: 对手手牌数归一化 (handSize/14)

## 数据管线

- int8 编码存储：缩放通道 ×因子后存 int8（WALL×21, TILE_FROM×3, HAND_SIZE×14）
- 解码时恢复 ÷因子
