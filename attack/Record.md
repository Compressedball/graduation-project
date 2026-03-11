# 1.初始处理dataset
adj 实现norm归一化
feature 行归一化
label 数据集的划分方式
    1.固定划分：train:20/类 val:500 test:1000
    2.随机划分：train:10% val:10% test:80%

# 2.模型的使用
1.GCN
    单纯的adj @ x作为输入计算，加权所有的邻居信息
2.GAT
    注意力机制，自动学习邻居对节点的影响力
3.GraphSAGE
    可以学习聚合函数，比如最大化，均值邻居等等

dropout = 0.5
对于train集小的模型来说，dropout大了可以更好的让模型不过拟合