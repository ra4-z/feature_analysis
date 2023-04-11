This repo is built for my feature analysis on vehicle re-identification.

# Function
分析提取的特征的距离。
先分为训练集和测试集两部分。每部分都做这样的分析。
1. 相同相机同ID的距离分布、平均值、方差，作为参照
2. 不同相机同ID的距离分布、平均值、方差
3. 不同相机不同ID的距离分布、平均值、方差


# Input/Output
## Data structure
数据准备好之后，按照如下的目录结构存放数据。
```python
├── data
│   ├── feature
│   │   ├── feature.pkl         # 保存的特征
│   │   ├── pic_name.txt        # 图片名字，与特征一一对应
│   │   └── dataset_split.json  # 数据集划分
│   ├── picture                 # 图片
│   │   ├── 0001_c1_00000001.jpg
│   │   ├── 0001_c1_00000002.jpg
```

# Quick start


# TODO
