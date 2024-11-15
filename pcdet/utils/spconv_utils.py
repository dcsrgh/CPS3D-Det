from typing import Set

import spconv
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False
    
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

import torch.nn as nn


def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class Conv2dBase(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True, isNorm=True,
                 isRelu=True):
        super(Conv2dBase, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.isNorm = isNorm
        self.isRelu = isRelu
        self.conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        if self.isNorm:
            self.bn = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        x = self.conv(x)
        if self.isNorm:
            x = self.bn(x)
        if self.isRelu:
            x = F.relu(x)
        return x
