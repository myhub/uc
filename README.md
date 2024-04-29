# uc 一款专注于数据分析的神经网络
***Windows*** ***Linux***

## 对比测试
+ [uc.mlp.MLP VS sklearn.neural_network.MLPRegressor
](examples/demo.ipynb)

## 安装
pip install uc

## 功能特性
+ 支持特征重要性
+ 支持缺失值
+ 支持am2/am2l/a2m2/a2m2l激活函数
+ 支持softmax/hardmax/mse/hardmse损失函数
+ 支持fc/add/conv运算

## 第一个例子
<pre>
# let's use a simple example to learn how to use
from uc.mlp import MLP
import numpy as np

# generate sample
X = np.linspace(-np.pi, np.pi, num=5000).reshape(-1, 1)
Y = np.sin(X)
print(X.shape, Y.shape)

# fit and predict
mlp = MLP(layer_size=[X.shape[1], 8, 8, 8, 1], rate_init=0.02, loss_type="mse", epoch_train=100, epoch_decay=10, verbose=1)

mlp.fit(X, Y)
pred = mlp.predict(X)

# show the result
import matplotlib.pyplot as plt  
plt.plot(X, pred)
plt.show()
</pre>

## 更多示例
+ [分类: iris](examples/iris/)
+ [回归: image painting](examples/image-painting/)
+ [分类: mnist](examples/mnist/)
