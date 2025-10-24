from tadaconv import TAdaConv2d, RouteFuncMLP
import torch
from enum import Enum, unique
import torch.nn as nn

@unique
class View(Enum):
    ATTACK = torch.tensor(0)
    DEFEND = torch.tensor(1)
    PASS   = torch.tensor(2)
    FOUL   = torch.tensor(3)


class FoulModel(nn.Module):
    def __init__(self):
        self.emb = nn.Embedding(4, 64)  # Exadmple embedding layer
        self.conv_rf = RouteFuncMLP(
                c_in=64,            # number of input filters
                ratio=4,            # reduction ratio for MLP
                kernels=[3,3],      # list of temporal kernel sizes
        )
        self.conv = TAdaConv2d(
                    in_channels     = 64,
                    out_channels    = 64,
                    kernel_size     = [1, 3, 3], # usually the temporal kernel size is fixed to be 1
                    stride          = [1, 1, 1], # usually the temporal stride is fixed to be 1
                    padding         = [0, 1, 1], # usually the temporal padding is fixed to be 0
                    bias            = False,
                    cal_dim         = "cin"
                )
        self.classifier = nn.Linear(64, 10) 
         # Example classifier layer


    def forward(self, x, view: View):        
        x = x + self.emb(nn.functional.one_hot(view.value, 4))  # Example embedding lookup
        x = self.conv(x, self.conv_rf(x))
        x = self.classifier(x.max(dim=[2,3,4]))  # Global max pooling and classification
        return x