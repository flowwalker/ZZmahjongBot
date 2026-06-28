# Record

大幅修改了feature.py以及model.py

- 输入 16ch → **148ch**

  - 手牌4ch + 可见牌4ch + 鸣牌16ch(4家×4类) + 圈风4ch + 门风4ch + 牌墙4ch + 弃牌112ch(4家×28步)

    

- SE reduction 4→16, GroupNorm 4→**16**, reduce层 GN→**8**

- ResNet 后插入 **BiLSTM**(128→256双向) → **SafeSelfAttention**(8头) → LayerNorm 残差

- 双头加 **Dropout 0.1** (这个后来认为是失败的操作)

- ~29M → **~39M**

### SL
- sl_pretrained_cpu.pt: epoch=6, acc=**0.8894** (v2.1: epoch=4/0.8829)
- 开始使用NPU，因此做了sl_pretrain.py适配: NPU 支持 + batch 4096 + DataLoader workers



> 后期，由于此版的性能超出预料，后期反复尝试RL训练，皆因loss爆炸失败
>
> 也是自此以后，认清了RL对于自己这种所谓“平凡人”和时间紧迫的人之困难，
>
> 因此聚焦如何把sl数据压缩到极致来训练bot
