import torch
import torch.nn.functional as F

from .tensoRF import TensorVMSplit
from .tensorBase import positional_encoding


class AlphaBlendingPLTRender(torch.nn.Module):
    """
    Color decomposition scheme: alpha blending
    """

    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, palette=None):
        super().__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        self.n_dim = 3 + (len_palette := len(palette)) - 1
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, len_palette - 1)
        torch.nn.init.constant_(layer3.bias, 0)
        palette = torch.as_tensor(palette, dtype=torch.float32)
        # torch.nn.init.uniform_(palette)
        self.register_parameter('palette', torch.nn.Parameter(palette.T, requires_grad=True))

        self.mlp = torch.nn.Sequential(layer1, torch.nn.LeakyReLU(inplace=True),
                                       layer2, torch.nn.LeakyReLU(inplace=True),
                                       layer3)

    def rgb_from_palette_rev(self, logits):
        opaque = torch.sigmoid(logits)
        log_opq = F.logsigmoid(logits)
        log_wa = torch.cumsum(F.logsigmoid(torch.neg(logits)), dim=-1)
        w_0 = opaque[..., :1]
        w_a = torch.exp(log_wa[..., :-1] + log_opq[..., 1:])
        w_last = torch.exp(log_wa[..., -1:])
        bary_coord = torch.cat((w_0, w_a, w_last), dim=-1)
        # bary_coord guarantee sum .eq. 1
        # assert torch.allclose(bary_coord.sum(dim=-1), torch.ones(()), atol=1e-3)
        return F.linear(bary_coord, self.palette), opaque

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata.append(positional_encoding(features, self.feape))
        if self.viewpe > 0:
            indata.append(positional_encoding(viewdirs, self.viewpe))
        logits = self.mlp(torch.cat(indata, dim=-1))
        rgb, opaque = self.rgb_from_palette_rev(logits)
        return torch.cat((rgb, opaque), dim=-1)


class PaletteTensorVM(TensorVMSplit):
    def init_render_func(self, shadingMode='PLT_Direct', pos_pe=6, view_pe=6, fea_pe=6, featureC=128, palette=None,
                         **kwargs):
        print('[init_render_func]', "shadingMode", shadingMode, "pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        if shadingMode == 'PLT_AlphaBlend':
            return AlphaBlendingPLTRender(self.app_dim, view_pe, fea_pe, featureC, palette).to(self.device)
        else:
            raise NotImplementedError
