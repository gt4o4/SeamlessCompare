import json
from collections import UserDict
from functools import partial
from itertools import chain
from types import SimpleNamespace

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


class TransformFile(SimpleNamespace):
    @staticmethod
    def load(filename):
        with open(filename) as f:
            return json.load(f, object_hook=TransformFile)

    def __init__(self, d):
        super().__init__(**d)

    def get(self, *args):
        return [getattr(self, item.removesuffix('_VM').replace('scene', 'scan'), None) for item in args]


MODEL_ZOO = ClassCollection(TensorVM, TensorCP, TensorVMSplit, ColorVMSplit)
LOSS_ZOO = ClassCollection(PLTLoss)
RENDER_ZOO = ClassCollection(MLPRender_PE, MLPRender_Fea, MLPRender, SHRender, RGBRender, PLTRender, MultiplePLTRender,
                             PoissonMLPRender)
