import json
from types import SimpleNamespace

from models.colorRF import ColorVMSplit, PoissonMLPRender
from models.loss import PLTLoss
from models.renderBase import PLTRender, MultiplePLTRender, SHRender, RGBRender, MLPRender_Fea, MLPRender_PE, MLPRender
from models.tensoRF import TensorVM, TensorCP, TensorVMSplit


class ClassCollection(dict):
    class CCMeta(type):
        def __str__(self):
            return self.__name__

        def __repr__(self):
            return self.__qualname__

    def __init__(self, *classes):
        super().__init__((cls.__name__, cls) for cls in classes)
        self.aliases = {alias: cls for cls in classes for alias in getattr(cls, '_aliases', ())}

    def __contains__(self, item):
        return str(item) in self.keys() or item in self.values()

    def get(self, name):
        return self.CCMeta(name, (super().get(name, self.aliases.get(name, type(name, (), {}))),), {})


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
