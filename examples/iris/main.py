# coding: utf-8
from uc.mlp import MLP
import numpy as np
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()

train_in = np.array(iris['data'], dtype=np.float32)
train_out = np.array(iris['target'], dtype=np.float32)

random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    train_in, train_out, test_size=0.2, random_state=random_state)

random.seed(random_state)

param = {
    'loss_type': 'softmax',
    'layer_size': [4, 16, 16, 16, 3],
    'activation': 'a2m2l',
    'output_range': [0, 1],
    'output_shrink': 0.001,
    'importance_out': True,
    'rate_init': 0.02,
    'rate_decay': 0.9,
    
    'epoch_log': 200,
    'epoch_decay': 40,
    'epoch_train': 2000,
    'verbose': 1,
}

mlp = MLP(param)

if 1:
    mlp.fit(X_train, y_train)
    model = mlp.save_model()
else:
    mlp.load_model()

predict_out = mlp.predict(X_test)
score = accuracy_score(y_test, predict_out)
print("test accuracy: %.2f%%" % (score*100))

from uc.plotting import plot_importance
plot_importance(mlp.feature_importances_)
