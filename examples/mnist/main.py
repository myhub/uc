from uc.mlp import MLP
from get_sample import train_in, train_out, test_in, test_out

param = {
    'layer_size': [
        (28, 28, 1),
        (14, 14, 4),
        (7, 7, 8),
        (1, 1, 10),
    ],
    'exf': [
        # 'Turbulence': [2, 2]
        {'Kernel': [6, 6], 'Stride': [2, 2], 'Pad': [2, 2]},
        {'Kernel': [6, 6], 'Stride': [2, 2], 'Pad': [2, 2]},
        {'Kernel': [7, 7], 'Stride': [1, 1], 'Pad': [0, 0]},
    ],

    'loss_type': 'hardmax',
    'output_range': [-1, 1],
    'output_shrink': 0.001,
    'regularization': 1,
    'op': 'conv',

    'verbose': 1,
    'rate_init': 0.01,
    'rate_decay': 0.8,
    'epoch_train': 2,
    'epoch_decay': 0.2,
    'epoch_log': 0.2,
}

mlp = MLP(param)
mlp.show_filter()
if 1:
    mlp.fit(train_in, train_out)
    mlp.save_model()
else:
    mlp.load_model()

predict_out = mlp.predict(test_in)
from sklearn.metrics import accuracy_score
score = accuracy_score(test_out, predict_out)
print("test accuracy: %.2f%%" % (score*100))