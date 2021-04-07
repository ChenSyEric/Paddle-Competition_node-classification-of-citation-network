#!/usr/bin/env python
# coding: utf-8

# # 论文引用网络节点分类比赛 Baseline
# 
# ## 赛题介绍
# 
# 
# 图神经网络（Graph Neural Network）是一种专门处理图结构数据的神经网络，目前被广泛应用于推荐系统、金融风控、生物计算中。图神经网络的经典问题主要有三种，包括节点分类、连接预测和图分类三种。本次比赛是图神经网络7日打卡课程的大作业，主要让同学们熟悉如何图神经网络处理节点分类问题。
# 
# 数据集为论文引用网络，图由大量的学术论文组成，节点之间的边是论文的引用关系，每一个节点提供简单的词向量组合的节点特征。我们的目的是给每篇论文推断出它的论文类别。
# 
# 
# 
# 

# ## 运行方式
# 本次基线基于飞桨PaddlePaddle 1.8.4版本，若本地运行则可能需要额外安装pgl、easydict、pandas等模块。
# 
# ## 本地运行
# 下载左侧文件夹中的所有py文件（包括build_model.py, model.py）,以及work目录，然后在右上角“文件”->“导出Notebook到py”，这样可以保证代码是最新版本），执行导出的py文件即可。完成后下载submission.csv提交结果即可。
# 
# ## AI Studio (Notebook)运行
# 依次运行下方的cell，完成后下载submission.csv提交结果即可。若运行时修改了cell，推荐在右上角重启执行器后再以此运行，避免因内存未清空而产生报错。 Tips：若修改了左侧文件夹中数据，也需要重启执行器后才会加载新文件。

# ## 代码整体逻辑
# 
# 1. 读取提供的数据集，包含构图以及读取节点特征（用户可自己改动边的构造方式）
# 
# 2. 配置化生成模型，用户也可以根据教程进行图神经网络的实现。
# 
# 3. 开始训练
# 
# 4. 执行预测并产生结果文件
# 

# ## 环境配置
# 
# 该项目依赖飞桨paddlepaddle==1.8.4, 以及pgl==1.2.0。请按照版本号下载对应版本就可运行。

# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
#!mkdir /home/aistudio/external-libraries
#!pip install pgl easydict -q -t /home/aistudio/external-libraries


# In[ ]:





# In[1]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# 

# In[2]:


import pgl
import paddle.fluid as fluid
import numpy as np
import time
import pandas as pd


# ## 图网络配置
# 
# 这里已经有很多强大的模型配置了，你可以尝试简单的改一下config的字段。
# 例如，换成GAT的配置
# ```
# config = {
#     "model_name": "GAT",
#     "num_layers":  1,
#     "dropout": 0.5,
#     "learning_rate": 0.01,
#     "weight_decay": 0.0005,
#     "edge_dropout": 0.00,
# }
# ```

# In[3]:


from easydict import EasyDict as edict

config = {
    "model_name": "DeeperGCN",
    "num_layers": 8,
    "dropout": 0.5,
    "learning_rate": 0.005,
    "weight_decay": 0.0005,
    "edge_dropout": 0.00,
}

config = edict(config)


# ## 数据加载模块
# 
# 这里主要是用于读取数据集，包括读取图数据构图，以及训练集的划分。

# In[4]:


from collections import namedtuple

Dataset = namedtuple("Dataset", 
               ["graph", "num_classes", "train_index",
                "train_label", "valid_index", "valid_label", "test_index"])

def load_edges(num_nodes, self_loop=True, add_inverse_edge=True):
    # 从数据中读取边
    edges = pd.read_csv("work/edges.csv", header=None, names=["src", "dst"]).values

    if add_inverse_edge:
        edges = np.vstack([edges, edges[:, ::-1]])

    if self_loop:
        src = np.arange(0, num_nodes)
        dst = np.arange(0, num_nodes)
        self_loop = np.vstack([src, dst]).T
        edges = np.vstack([edges, self_loop])
    
    return edges

def load():
    # 从数据中读取点特征和边，以及数据划分
    node_feat = np.load("work/feat.npy")
    num_nodes = node_feat.shape[0]
    edges = load_edges(num_nodes=num_nodes, self_loop=True, add_inverse_edge=True)
    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges, node_feat={"feat": node_feat})
    
    indegree = graph.indegree()
    norm = np.maximum(indegree.astype("float32"), 1)
    norm = np.power(norm, -0.5)
    graph.node_feat["norm"] = np.expand_dims(norm, -1)
    
    df = pd.read_csv("work/train.csv")
    node_index = df["nid"].values
    node_label = df["label"].values
    train_part = int(len(node_index) * 0.8)
    train_index = node_index[:train_part]
    train_label = node_label[:train_part]
    valid_index = node_index[train_part:]
    valid_label = node_label[train_part:]
    test_index = pd.read_csv("work/test.csv")["nid"].values
    dataset = Dataset(graph=graph, 
                    train_label=train_label,
                    train_index=train_index,
                    valid_index=valid_index,
                    valid_label=valid_label,
                    test_index=test_index, num_classes=35)
    return dataset


# In[5]:


dataset = load()

train_index = dataset.train_index
train_label = np.reshape(dataset.train_label, [-1 , 1])
train_index = np.expand_dims(train_index, -1)

val_index = dataset.valid_index
val_label = np.reshape(dataset.valid_label, [-1, 1])
val_index = np.expand_dims(val_index, -1)

test_index = dataset.test_index
test_index = np.expand_dims(test_index, -1)
test_label = np.zeros((len(test_index), 1), dtype="int64")


# ## 组网模块
# 
# 这里是组网模块，目前已经提供了一些预定义的模型，包括**GCN**, **GAT**, **APPNP**等。可以通过简单的配置，设定模型的层数，hidden_size等。你也可以深入到model.py里面，去奇思妙想，写自己的图神经网络。

# In[6]:


import pgl
import model
import paddle.fluid as fluid
import numpy as np
import time
from build_model import build_model

#place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
train_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()
with fluid.program_guard(train_program, startup_program):
    with fluid.unique_name.guard():
        gw, loss, acc, pred = build_model(dataset,
                            config=config,
                            phase="train",
                            main_prog=train_program)

test_program = fluid.Program()
with fluid.program_guard(test_program, startup_program):
    with fluid.unique_name.guard():
        _gw, v_loss, v_acc, v_pred = build_model(dataset,
            config=config,
            phase="test",
            main_prog=test_program)


test_program = test_program.clone(for_test=True)

exe = fluid.Executor(place)


# ## 开始训练过程
# 
# 图神经网络采用FullBatch的训练方式，每一步训练就会把所有整张图训练样本全部训练一遍。
# 
# 

# In[ ]:


import os
epoch = 600


exe.run(startup_program)
#if os.path.isdir('./best_model/'):
#    fluid.load(train_program, os.path.join('./best_model/', 'model'), exe)

# 将图数据变成 feed_dict 用于传入Paddle Excecutor
best_acc = 0.0
feed_dict = gw.to_feed(dataset.graph)
'''
for epoch in range(epoch):
    # Full Batch 训练
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    feed_dict["node_index"] = np.array(train_index, dtype="int64")
    feed_dict["node_label"] = np.array(train_label, dtype="int64")
    
    train_loss, train_acc = exe.run(train_program,
                                feed=feed_dict,
                                fetch_list=[loss, acc],
                                return_numpy=True)

    # Full Batch 验证
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    feed_dict["node_index"] = np.array(val_index, dtype="int64")
    feed_dict["node_label"] = np.array(val_label, dtype="int64")
    val_loss, val_acc = exe.run(test_program,
                            feed=feed_dict,
                            fetch_list=[v_loss, v_acc],
                            return_numpy=True)
    print("Epoch", epoch, "Train Acc", train_acc[0], "Train Loss", train_loss[0],"Valid Acc", val_acc[0],"Valid Loss", val_loss[0])
    if  val_acc[0]> best_acc:
        best_acc = val_acc[0]
        ckpt_dir =  os.path.join('./', 'best_model')
        print("Save model checkpoint to {}".format(ckpt_dir))
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        fluid.save(test_program, os.path.join(ckpt_dir, 'model'))
'''    


# ## 对测试集进行预测
# 
# 训练完成后，我们对测试集进行预测。预测的时候，由于不知道测试集合的标签，我们随意给一些测试label。最终我们获得测试数据的预测结果。
# 

# In[ ]:


feed_dict["node_index"] = np.array(test_index, dtype="int64")
feed_dict["node_label"] = np.array(test_label, dtype="int64") #假标签
if os.path.isdir('./best_model/'):
    fluid.load(test_program, os.path.join('./best_model/', 'model'), exe)
test_prediction = exe.run(test_program,
                            feed=feed_dict,
                            fetch_list=[v_pred],
                            return_numpy=True)[0]


# ## 生成提交文件
# 
# 最后一步，我们可以使用pandas轻松生成提交文件，最后下载 submission.csv 提交就好了。

# In[ ]:


submission = pd.DataFrame(data={
                            "nid": test_index.reshape(-1),
                            "label": test_prediction.reshape(-1)
                        })
submission.to_csv("submission.csv", index=False)


# ## 进阶资料
# 
# 
# ### 1. 自己动手实现图神经网络
# 
# 这里以想自己实现一个CustomGCN为例子
# 
# 首先，我们在model.py 创建一个类CustomGCN
# 
# ```
# import paddle.fluid.layers as L
# 
# class CustomGCN(object):
#     """实现自己的CustomGCN"""
#     def __init__(self, config, num_class):
#         # 分类的数目
#         self.num_class = num_class
#         # 分类的层数，默认为1
#         self.num_layers = config.get("num_layers", 1)
#         # 中间层 hidden size 默认64
#         self.hidden_size = config.get("hidden_size", 64)
#         # 默认dropout率0.5
#         self.dropout = config.get("dropout", 0.5)
#         
#     def forward(self, graph_wrapper, feature, phase):
#         """定义前向图神经网络
#         
#         graph_wrapper: 图的数据的容器
#         feature: 节点特征
#         phase: 标记训练或者测试阶段
#         """
#         # 通过Send Recv来定义GCN
#         def send_func(src_feat, dst_feat, edge_feat):
#             # 发送函数简单将输入src发送到输出节点dst。
#             return { "output": src_feat["h"] }
#             
#         def recv_func(msg):
#             # 对消息函数进行简单求均值
#             return L.sequence_pool(msg["output"], "average")
# 		
#         # 调用发送
#         message = graph_wrapper.send(send_func,
#                                nfeat_list=[ ("h", feature)])
#                           
#         # 调用接收
#         output = graph_wrapper.recv(message, recv_func)
#         
#         # 对输出过一层MLP
#         output = L.fc(output, size=self.num_class, name="final_output")
# ```
# 
# 
# 然后，我们只要改一下本notebook脚本的config，把model_type改成CustomGCN就能跑起来了。
# 
# ```
# config = {
#     "model_name": "CustomGCN",
# }
# ```

# 

# ### 2. One More Thing
# 
# 如果大家还想要别的奇思妙想，可以参考以下论文，他们都在节点分类上有很大提升。
# 
# * Predict then Propagate: Graph Neural Networks meet Personalized PageRank (https://arxiv.org/abs/1810.05997)
# 
# * Simple and Deep Graph Convolutional Networks (https://arxiv.org/abs/2007.02133)
# 
# * Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification (https://arxiv.org/abs/2009.03509)
# 
# * Combining Label Propagation and Simple Models Out-performs Graph Neural Networks (https://arxiv.org/abs/2010.13993)

# In[ ]:




