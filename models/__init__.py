import json
import os
from collections import UserDict, UserString
from functools import partial
from itertools import chain
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from models.colorRF import ColorVMSplit, PoissonMLPRender
from models.loss import PLTLoss
from models.renderBase import PLTRender, MultiplePLTRender, SHRender, RGBRender, MLPRender_Fea, MLPRender_PE, MLPRender
from models.tensoRF import TensorVM, TensorCP, TensorVMSplit


class ClassCollection(UserDict):
    class CCMeta(partial):
        def __str__(self):
            return self.func.__name__.__str__()

        def __repr__(self):
            return self.func.__qualname__.__repr__()

        def __getattr__(self, item):
            return getattr(self.func, item)

    def __init__(self, *classes):
        super().__init__(
            (alias, self.CCMeta(cls)) for cls in classes for alias in chain(
                (cls.__name__,), getattr(cls, '_aliases', ())
            ))

    def __contains__(self, item):
        return str(item) in self.data.keys() or item in self.data.values()

    def load(self, name):
        return self.get(name, name)


class TransformFile(UserString):
    class SimpleTransform(SimpleNamespace):
        def __init__(self, d):
            super().__init__(**d)

        def matrix(self):
            scale = np.diag((*scale, 1.)) if (scale := getattr(self, 'scale', None)) else np.diag((1., 1., 1., 1.))
            if r := getattr(self, 'rot', None):
                r = Rotation.from_quat(np.roll(r, -1)).as_matrix()
                r = np.hstack((r, np.expand_dims(np.asarray(self.trans), -1)))
                scale = np.vstack((r, np.array((0, 0, 0, 1)))) @ scale
            return scale

    def __init__(self, filename):
        super().__init__(filename)
        with open(filename, mode='r') as f:
            self.ns = json.load(f, object_hook=self.SimpleTransform)

    def get(self, *args):
        prefix = os.path.commonprefix(args)
        suffix = '_VM'
        strip = '_'
        return [getattr(
            self.ns, item.removeprefix(prefix).removesuffix(suffix).strip(strip).replace('scene', 'scan'), None
        ) for item in args]


MODEL_ZOO = ClassCollection(TensorVM, TensorCP, TensorVMSplit, ColorVMSplit)
LOSS_ZOO = ClassCollection(PLTLoss)
RENDER_ZOO = ClassCollection(MLPRender_PE, MLPRender_Fea, MLPRender, SHRender, RGBRender, PLTRender, MultiplePLTRender,
                             PoissonMLPRender)
