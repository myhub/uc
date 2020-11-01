# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

import os
import sys
import ctypes
import platform
import numpy

try:
    from . import conlib
except ImportError:
    import conlib


class Mesh(object):
    FLAG_V = 1
    FLAG_DV = 2
    FLAG_EX = 4

    MESH_ID = 0

    def __init__(self, dtype="float64"):
        self.mesh_init(dtype=dtype)

    def __del__(self):
        # print('__del__', self.mi_)
        self.destroy()

    def mesh_init(self, dtype):
        self.load_library(dtype)

        self.mi_ = Mesh.MESH_ID
        Mesh.MESH_ID += 1

        self.conf_ = {}
        self.tensors = []
        self.filters = []

        sys.stdout.flush()

    def load_library(self, dtype):
        self._libc = None
        if dtype in [numpy.float64, "float64", "double"]:
            self.DTYPE = numpy.float64
            self.C_FLOAT = ctypes.c_double

            _basedir = os.path.dirname(os.path.abspath(__file__))
            if platform.system() == "Windows":
                self._libc = ctypes.cdll.LoadLibrary(
                    os.path.join(_basedir, 'mesh.dll.2'))
            elif platform.system() == "Linux":
                self._libc = ctypes.cdll.LoadLibrary(
                    os.path.join(_basedir, 'mesh.so.2'))

        if self._libc is None:
            raise NotImplementedError()

        self._libc.mesh_cal_loss.restype = self.C_FLOAT
        self._libc.mesh_random.restype = self.C_FLOAT
        self._libc.mesh_model_size.restype = ctypes.c_uint64

        self._libc.mesh_info()

    def set_conf(self, con={}):
        if not isinstance(con, dict):
            raise NotImplementedError()

        self.conf_.update(con)

        con = conlib.dumps(con).encode()
        self._libc.mesh_set_conf(self.mi_, con)

    def set_tensor(self, ti, con={}):
        if not isinstance(con, dict):
            raise NotImplementedError()

        while ti >= len(self.tensors):
            self.tensors.append({})

        self.tensors[ti].update(con)

        con = conlib.dumps(con).encode()
        self._libc.mesh_set_tensor(self.mi_, ti, con)

    def set_filter(self, fi, con={}):
        if isinstance(fi, dict):
            con = fi
            fi = len(self.filters)

        if not isinstance(con, dict):
            raise NotImplementedError()

        while fi >= len(self.filters):
            self.filters.append({})

        self.filters[fi].update(con)

        con = conlib.dumps(con).encode()
        self._libc.mesh_set_filter(self.mi_, fi, con)

    def clear_tensor(self, ti=-1, flag=FLAG_V | FLAG_DV):
        self._libc.mesh_clear_tensor(self.mi_, ti, flag)

    def clear_filter(self, fi=-1, flag=FLAG_DV):
        self._libc.mesh_clear_filter(self.mi_, fi, flag)

    def importance(self, ti=-1):
        self._libc.mesh_importance(self.mi_, ti)

    def input(self, ti, buf):
        self._libc.mesh_input(self.mi_, ti, numpy.ctypeslib.as_ctypes(buf))

    def cal_loss(self, ti, buf):
        return self._libc.mesh_cal_loss(self.mi_, ti, numpy.ctypeslib.as_ctypes(buf))

    def read_tensor(self, ti, buf, flag):
        self._libc.mesh_read_tensor(
            self.mi_, ti, numpy.ctypeslib.as_ctypes(buf), flag)

    def run_filler(self, fi=-1):
        self._libc.mesh_run_filler(self.mi_, fi)

    def forward(self):
        self._libc.mesh_forward(self.mi_)

    def backward(self):
        self._libc.mesh_backward(self.mi_)

    def renew(self, rate):
        self._libc.mesh_renew(self.mi_, self.C_FLOAT(rate))

    def destroy(self):
        self._libc.mesh_destroy(self.mi_)

    def random(self):
        return self._libc.mesh_random()

    def randrange(self, n):
        return self._libc.mesh_randrange(ctypes.c_uint32(n))

    def shuffle_array(self, arr):
        assert arr.flags['C_CONTIGUOUS']

        self._libc.mesh_shuffle(
            numpy.ctypeslib.as_ctypes(arr), arr.size, arr.itemsize)

    def shuffle_multi_array(self, *bufs):
        pos = numpy.arange(bufs[0].size, dtype=numpy.int32)
        self._libc.mesh_shuffle(
            numpy.ctypeslib.as_ctypes(pos), pos.size, pos.itemsize)

        for buf in bufs:
            buf[:] = buf[pos]

    def show_conf(self):
        self._libc.mesh_show_conf(self.mi_)

    def show_tensor(self, ti=-1):
        self._libc.mesh_show_tensor(self.mi_, ti)

    def show_filter(self, fi=-1):
        self._libc.mesh_show_filter(self.mi_, fi)

    def model_size(self):
        return self._libc.mesh_model_size(self.mi_)

    def save_model(self, filepath="model.bin"):
        mem = numpy.zeros(self.model_size(), dtype=numpy.uint8)

        assert mem.flags['C_CONTIGUOUS']
        ret = self._libc.mesh_save_model(
            self.mi_, numpy.ctypeslib.as_ctypes(mem))
        assert ret > 0

        if filepath is not None:
            with open(filepath, 'wb') as f:
                f.write(mem.tobytes())

        return mem

    def load_model(self, filepath="model.bin"):
        if conlib.is_str(filepath):
            mem = numpy.fromfile(filepath, dtype=numpy.uint8)
        else:
            mem = filepath

        assert mem.flags['C_CONTIGUOUS']
        ret = self._libc.mesh_load_model(
            self.mi_, numpy.ctypeslib.as_ctypes(mem))
        assert ret > 0


def test():
    m = Mesh()

    x1 = numpy.array([1, 2, 3, 4, 5])
    x2 = numpy.array([1, 2, 3, 4, 5])
    m.shuffle_multi_array(x1, x2)
    print('shuffle_multi_array', x1)
    sys.stdout.flush()
    assert all(x1 == x2)

    m.set_conf()
    m.set_tensor(0, {'Shape': [1, 1, 2]})
    m.set_tensor(1, {'Shape': [1, 1, 3]})
    m.set_filter(0, {
        'Tins': [0], 'Touts': [1],
        'Op': 'fc',
        'WeightFiller.Adj': [80, 81]})
    m.set_filter(1, {'Tins': [0], 'Touts': [1]})
    m.show_conf()
    m.show_tensor()
    m.show_filter()

    print('model_size', m.model_size())

    mem = numpy.arange(192, dtype=numpy.uint8)

    # print('load_model', len(mem), mem[:5])

    m.load_model(mem)

    m.save_model('test1.bin')

    m.load_model('test1.bin')

    mem1 = m.save_model()

    print('save_model', len(mem1), mem1[:5])


if __name__ == '__main__':
    test()

    print('quit')
