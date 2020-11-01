# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

import os
import sys
import math
import numpy
import time
import copy
import ctypes

try:
    from .mesh import Mesh
except ImportError:
    from mesh import Mesh


class MLP(Mesh):
    def __init__(self, params={}, **kwargs):
        self.rate_ = None
        self.loss_ = None

        self.feature_importances_ = None

        self.params_ = {}
        self.params_.update(params)
        self.params_.update(kwargs)

        self.count_ = None
        self.total_ = None

        self.iteration_log_ = None
        self.iteration_decay_ = None

        self.sample_size = None

        self.mesh_init(dtype=self.params_.get('dtype', 'float64'))

        self.mlp_init(**self.params_)

    def get_params(self, deep=False):
        return self.params_

    def set_params(self, **kwargs):
        self.params_.update(kwargs)
        self.mlp_init(**self.params_)

        return self

    def mlp_init(self,
                 op='fc',
                 exf=[],
                 activation=[],
                 layer_size=[],
                 input_type='pointwise',
                 loss_type='mse',
                 output_range=(-1, 1),
                 output_shrink=0.001,               # 0.1
                 importance_mul=0.001,
                 leaky=-0.001,
                 dropout=0,
                 bias_rate=[0.005],
                 weight_rate=[],
                 regularization=1,

                 importance_out=False,
                 loss_mul=0.001,

                 epoch_log=1,
                 epoch_train=1,
                 epoch_decay=1,

                 sample_size=None,

                 rate_init=0.06,
                 rate_decay=0.9,

                 shuffle=True,

                 verbose=1,
                 ):

        self.epoch_log = epoch_log
        self.epoch_decay = epoch_decay
        self.epoch_train = epoch_train
        
        self.sample_size = sample_size

        self.rate_init = rate_init
        self.rate_decay = rate_decay
        self.importance_out = importance_out
        self.loss_mul = loss_mul
        self.shuffle = shuffle
        self.output_range = output_range

        self.verbose = verbose

        self.set_conf({})

        for idx, shape in enumerate(layer_size):
            if isinstance(shape, int):
                shape = (1, 1, shape)
            assert isinstance(shape, tuple)

            arg = {'Shape': shape}
            if idx == 0:
                arg['InputType'] = input_type
                arg['ImportanceMul'] = importance_mul
            if idx == len(layer_size) - 1:
                arg['Regularization'] = regularization
                arg['LossType'] = loss_type
                arg['OutputRange'] = output_range
                arg['OutputShrink'] = output_shrink

            self.set_tensor(idx, arg)

        for idx in range(max(len(layer_size) - 1, len(exf))):
            arg = {'Tins': [idx], 'Touts': [idx+1]}

            kv = {
                'BiasRate': bias_rate,
                'WeightRate': weight_rate,
                'Dropout': dropout,
                'Leaky': leaky,
                'Activation': activation,
                'Op': op,
            }

            for k in kv:
                if not isinstance(kv[k], list):
                    if kv[k] is not None:
                        arg[k] = kv[k]
                elif idx < len(kv[k]) and kv[k][idx] is not None:
                    arg[k] = kv[k][idx]

            if isinstance(exf, dict):
                arg.update(exf)
            elif idx < len(exf) and isinstance(exf[idx], dict):
                arg.update(exf[idx])

            self.set_filter(idx, arg)

        self.run_filler()

    def check_arr(self, arr):
        if not isinstance(arr, numpy.ndarray):
            if hasattr(arr, 'toarray'):
                arr = arr.toarray()
            arr = numpy.array(arr, self.DTYPE)

        if arr.dtype != self.DTYPE:
            arr = arr.astype(self.DTYPE)
        if arr.ndim == 1:
            arr = arr.reshape((-1, 1))
        if not arr.flags['C_CONTIGUOUS']:
            arr = numpy.ascontiguousarray(arr)
        return arr

    def fit(self, in_arr, target_arr):
        self.set_conf({'IsTrain': True})

        in_arr = self.check_arr(in_arr)
        target_arr = self.check_arr(target_arr)

        arr_size = len(in_arr)
        assert arr_size == len(target_arr)

        if self.sample_size is None:
            self.sample_size = arr_size

        if self.total_ is None: self.total_ = 0
        if self.count_ is None: self.count_ = 0

        self.total_ += int(self.sample_size * self.epoch_train)

        if self.iteration_log_ is None:
            self.iteration_log_ = int(self.epoch_log * self.sample_size)
        if self.iteration_decay_ is None and self.epoch_decay is not None:
            self.iteration_decay_ = int(self.epoch_decay * self.sample_size)

        if self.rate_ is None:
            self.rate_ = self.rate_init

        if self.shuffle:
            idx_range = numpy.arange(arr_size, dtype=numpy.int32)
        else:
            idx_range = range(arr_size)

        while self.count_ < self.total_:
            if self.shuffle:
                self.shuffle_array(idx_range)

            for idx in idx_range:
                loss = self.train_one(
                    in_arr[idx], target_arr[idx], self.rate_, self.importance_out)

                if self.loss_ is None:
                    self.loss_ = loss
                else:
                    self.loss_ += (loss - self.loss_) * self.loss_mul

                if math.isnan(self.loss_):
                    print("loss is nan", file=sys.stderr)
                    sys.exit()

                self.count_ += 1

                if self.iteration_decay_ is not None and self.count_ % self.iteration_decay_ == 0:
                    self.rate_ *= self.rate_decay

                if self.verbose > 0:
                    if (self.count_ % self.iteration_log_ == 0 or
                            self.count_ >= self.total_):

                        if self.verbose > 0:
                            print("Iteration %g/%g\tEpoch %g/%g" % (
                                self.count_ % self.sample_size, self.sample_size,
                                self.count_ // self.sample_size, self.epoch_train))

                            print("    rate: %g loss: %g" % (self.rate_, self.loss_))
                        sys.stdout.flush()

                if self.count_ >= self.total_:
                    break

        if self.importance_out and self.finished():
            self.feature_importances_ = numpy.zeros(
                (numpy.prod(self.tensors[0]['Shape']),), dtype=self.DTYPE)
            self.read_tensor(0, self.feature_importances_, self.FLAG_EX)

        return self

    def feed(self, in_arr, out_arr=None):
        self.set_conf({'IsTrain': False})

        in_arr = self.check_arr(in_arr)

        if out_arr is None:
            out_arr = numpy.zeros(
                (len(in_arr), numpy.prod(self.tensors[-1]['Shape'])), dtype=self.DTYPE)
        else:
            out_arr = self.check_arr(out_arr)

        arr_size = len(in_arr)
        assert arr_size == len(out_arr)

        for idx in range(arr_size):
            self.predict_one(in_arr[idx], out_arr[idx])

        return out_arr

    def predict_proba(self, in_arr, out_arr=None):
        out_arr = self.feed(in_arr, out_arr)

        return out_arr

    def predict(self, in_arr, out_arr=None):
        out_arr = self.feed(in_arr, out_arr)

        loss_type = self.tensors[-1]['LossType']
        if loss_type in ["softmax", "hardmax"]:
            return out_arr.argmax(axis=1)

        if out_arr.ndim == 2 and out_arr.shape[1] == 1:
            out_arr = out_arr.reshape(-1)

        return out_arr

    def train_one(self, in_buf, target_buf, rate, importance_out):
        self.clear_tensor(-1, self.FLAG_V | self.FLAG_DV)
        self.clear_filter(-1, self.FLAG_DV)
        self.input(0, in_buf)

        self.forward()

        loss = self.cal_loss(len(self.tensors) - 1, target_buf)

        self.backward()
        self.renew(rate)

        if importance_out:
            self.importance(0)

        return loss

    def predict_one(self, in_buf, out_buf):
        self.clear_tensor(-1, self.FLAG_V)
        self.input(0, in_buf)
        self.forward()

        flag = self.FLAG_V
        loss_type = self.tensors[-1]['LossType']
        if loss_type in ["softmax", "hardmax"]:
            flag = self.FLAG_EX

        self.read_tensor(len(self.tensors) - 1, out_buf, flag)


if __name__ == '__main__':
    mlp = MLP(layer_size=[2, 1],  op="fc")
    print(mlp.mi_)

    mem = mlp.save_model()
    print(len(mem), mem[-5:])

    # mlp_copy = mlp_clone(mlp)
    # print(mlp_copy.mi_)
    # mem1 = mlp_copy.save_model()
    # print(len(mem1), mem1[-5:])

    # mlp.run_filler()

    # mem = mlp.save_model()
    # print(len(mem), mem[-5:])

    # # mlp_copy = clone(mlp)
    # print(mlp_copy.mi_)
    # mem1 = mlp_copy.save_model()
    # print(len(mem1), mem1[-5:])

    # print("appname:", m.info("appname"))
    # sys.stdout.flush()

    # m.fit()
    # m.set_tensor(0,
    #              {'Shape': [1, 2], 'Regularization': 0.1})

    # m.show_conf()
    # m.show_tensor()
    # m.show_filter()
    # m.destroy()
