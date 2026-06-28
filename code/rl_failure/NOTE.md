# RL失败note

主要是对已经训的不错v6_to_148ch_add_lstm_attn尝试进行RL训练场是

## 现象

对CNNModel（BiLSTM + SelfAttention 双头结构）做 PPO/RL 微调时，无论怎么调学习率、clip、entropy_coeff、reward_scale、max_grad_norm，loss 都会迅速爆炸，训练无法稳定。

耗费了大量时间！

毫无所用。

猜测原因如下：

- sl时候要么就是v头没有进行有效训练

  （因为只有尾步才可以得到分数，之前毫无所知，且你不一定坏可能是别人牌运气好）

- 要么就是设计的v头和rl时数量级之类的不够匹配

- 总之尝试多次后放弃
