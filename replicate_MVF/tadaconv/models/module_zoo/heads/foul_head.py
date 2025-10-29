

from multiprocessing import pool
import torch
from tadaconv.models.base.base_blocks import HEAD_REGISTRY, BaseHead
from tadaconv.models.utils.init_helper import _init_transformer_weights
from tadaconv.utils import logging

logger = logging.get_logger(__name__)


@HEAD_REGISTRY.register()
class FOULHead(BaseHead):
    def __init__(self, cfg):
        super(FOULHead, self).__init__(cfg)
        self.apply(_init_transformer_weights)

    def _construct_head(
        self,
        dim,
        num_classes,
        dropout_rate,
        activation_func
    ):
        self.emb = torch.nn.Embedding(2, dim)

        self.global_avg_pool = torch.nn.AdaptiveAvgPool3d(1)

        if dropout_rate > 0.0: 
            self.dropout = torch.nn.Dropout(dropout_rate)

        self.out = torch.nn.Linear(dim, num_classes, bias=True)

        if activation_func == "softmax":
            self.activation = torch.nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError(
            "{} is not supported as an activation"
            "function.".format(activation_func)
            )

    def forward(self, x, mask):
        """
        Returns:
            x (Tensor): classification predictions.
            logits (Tensor): global average pooled features.
        """
        if len(x.shape) == 6:
            B, V, C, T, H, W = x.shape
            x_reshaped = x.view(B * V, C, T, H, W)
            x_pooled = self.global_avg_pool(x_reshaped)         # (B*V, C, 1, 1, 1)
            x = x_pooled.view(B, V, C)


        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x

        emb = self.emb(mask-1)

        assert emb.shape == out.shape, f"emb shape {emb.shape} and out shape {out.shape} do not match"

        out = out + emb

        out, _ = pooled, idx = torch.max(out, dim=1)

        logger.info(f"FOULHead pooled output shape: {out.shape}")
        
        out = self.out(out)

        if not self.training:
            out = self.activation(out)
        
        out = out.view(out.shape[0], -1)
        return out, x.view(x.shape[0], -1)