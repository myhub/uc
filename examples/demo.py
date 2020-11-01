# coding=utf-8

# let's use a simple example to learn how to use
from uc.mlp import MLP
import numpy as np

# generate sample
X = np.linspace(-np.pi, np.pi, num=5000).reshape(-1, 1)
Y = 2 * np.sin(5 * X) + 5 * np.cos(2 * X)
print(X.shape, Y.shape)

# fit and predict
mlp = MLP(layer_size=[X.shape[1], 8, 8, 8, 1], rate_init=0.02, loss_type="mse", epoch_train=100, epoch_decay=10, verbose=1)

mlp.fit(X, Y)
pred = mlp.predict(X)

# show the result
import matplotlib.pyplot as plt
plt.subplot(211)
plt.plot(X, Y)
plt.subplot(212)
plt.plot(X, pred)
plt.show()
