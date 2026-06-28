CNN+Transformer（SE Pyramid CNN骨架 → Reshape序列 → PosEmbed → SelfAttn+FFN → Head）。重用了v13旧管线（preprocess_mp.py/dataset.py），IO288x问题未解决。无权重。
