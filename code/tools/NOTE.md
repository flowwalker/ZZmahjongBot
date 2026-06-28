# Note

- data_test.txt是小测试数据集

- validate_version.py用于检测流程是否跑通

- export_weights.py用于导出纯CPU权重的文件

- tournament.py用于本地测试四者对战水平如何，无权重文件则随机初始化

  > 但是有个非常重要的问题是！他似乎洗牌随机性很低！（洗牌很浅）所以有一种现象是开局好，后面局面也好，比较难以体现bot真正水平，仅供参考
  >
  > 一般是开多终端进行综合比对
