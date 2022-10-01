from typing import Optional
import logging
logger = logging.getLogger(__name__)

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from bycha.modules.encoders.layers import AbstractEncoderLayer
from bycha.modules.layers.autopruning_ffn import AutoPruningFFN


class AutoPruningEncoderLayer(AbstractEncoderLayer):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0,
                 activation="relu",
                 normalize_before=False,):
        super(AutoPruningEncoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout)
        # Implementation of Feedforward model
        self.ffn = AutoPruningFFN(d_model, dim_feedforward=dim_feedforward, activation=activation)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_mask = None
        self.fc_alpha = nn.Parameter(1e-3*torch.randn(8))
        self._sorted_fc_indeces = None
        self._tau = 0.

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                :math:`(S, B, D)`, where S is sequence length, B is batch size and D is feature dimension
            src_mask: the attention mask for the src sequence (optional).
                :math:`(S, S)`, where S is sequence length.
            src_key_padding_mask: the mask for the src keys per batch (optional).
                :math: `(B, S)`, where B is batch size and S is sequence length
        """
        residual = src
        if self.normalize_before:
            src = self.self_attn_norm(src)
        src = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = self.dropout1(src)
        src = residual + src
        if not self.normalize_before:
            src = self.self_attn_norm(src)

        residual = src
        if self.normalize_before:
            src = self.ffn_norm(src)
        src = self.ffn(src, self.fc_alpha, self._sorted_fc_indeces, self._tau)
        src = self.dropout2(src)
        src = residual + src
        if not self.normalize_before:
            src = self.ffn_norm(src)
        return src

    def reset_fc_weights(self):
        if self.fc_mask is None:
            weights_final = F.softmax(self.fc_alpha.data, dim=-1)
            max_index = weights_final.max(-1, keepdim=True)[1].item()
            prune_ratio = 0.1 * max_index
            logger.info("encoder layer, weights: %r; prune ratio: %f" % (weights_final, prune_ratio))
            prune_num = int(prune_ratio * self._sorted_fc_indeces.size(0))
            self.fc_mask = self._sorted_fc_indeces[:prune_num]
            self._sorted_fc_indeces = None
        self.ffn._fc1.weight.data[self.fc_mask] = self.ffn._fc1.weight.data[self.fc_mask].fill_(0)
        if self.ffn._fc1.bias is not None:
            self.ffn._fc1.bias.data[self.fc_mask] = self.ffn._fc1.bias.data[self.fc_mask].fill_(0)
        self.ffn._fc2.weight.data[:, self.fc_mask] = self.ffn._fc2.weight.data[:, self.fc_mask].fill_(0)
    
    def set_tau(self, tau):
        self._tau = tau
