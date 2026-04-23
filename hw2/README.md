
## 环境配置
```bash
pip install -r requirements.txt 
```

## 运行脚本
```bash
python CRF.py   # 直接调包版本
python CRF_1.py # 手动训练版本
```

## 重现实验结果
- 我在checkpoints文件夹中保存了训练好的三个模型权重，在main.py函数中可以设置force_retrain=True来重新训练模型，默认是直接加载已经训练好的模型，全局的随机种子设置为42.