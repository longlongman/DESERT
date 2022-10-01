from typing import Optional
import logging
logger = logging.getLogger(__name__)

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from bycha.modules.decoders.layers import AbstractDecoderLayer
from bycha.modules.layers.autopruning_ffn import AutoPruningFFN

class AutoPruningDecoderLayer(AbstractDecoderLayer):
    """
    TransformerDecoderLayer performs one layer of time-masked transformer operation,
    namely self-attention and feed-forward network.

    Args:
        d_model: feature dimension
        nhead: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.,
                 activation="relu",
                 normalize_before=False):
        super(AutoPruningDecoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        # Implementation of Feedforward model
        self.ffn = AutoPruningFFN(d_model, dim_feedforward=dim_feedforward, activation=activation)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.fc_mask = None
        self.fc_alpha = nn.Parameter(1e-3*torch.randn(8))
        self._sorted_fc_indeces = None
        self._tau = 0.

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Pass the inputs (and mask) through the decoder layer in training mode.

        Args:
            tgt: the sequence to the decoder layer (required).
                :math:`(T, B, D)`, where T is sequence length, B is batch size and D is feature dimension
            memory: the sequence from the last layer of the encoder (required).
                :math:`(M, B, D)`, where M is memory size, B is batch size and D is feature dimension
            tgt_mask: the mask for the tgt sequence (optional).
                :math:`(T, T)`, where T is sequence length.
            memory_mask: the mask for the memory sequence (optional).
                :math:`(M, M)`, where M is memory size.
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                :math: `(B, T)`, where B is batch size and T is sequence length.
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
                :math: `(B, M)`, where B is batch size and M is memory size.
        """
        if self._mode == 'infer':
            tgt = tgt[-1:]
            tgt_mask, tgt_key_padding_mask = None, None
        residual = tgt
        if self.normalize_before:
            tgt = self.self_attn_norm(tgt)
        prevs = self._update_cache(tgt) if self._mode == 'infer' else tgt
        tgt = self.self_attn(tgt, prevs, prevs, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.dropout1(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.self_attn_norm(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.multihead_attn_norm(tgt)
        tgt = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.dropout2(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.multihead_attn_norm(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.ffn_norm(tgt)
        tgt = self.ffn(tgt, self.fc_alpha, self._sorted_fc_indeces, self._tau)
        tgt = self.dropout3(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.ffn_norm(tgt)
        return tgt

    def _update_cache(self, cur):
        """
        Update cache with current states

        Args:
            cur: current state
        """
        prev = torch.cat([self._cache['prev'], cur], dim=0) if 'prev' in self._cache else cur
        self._cache['prev'] = prev
        return prev
    
    def reset_fc_weights(self):
        if self.fc_mask is None:
            weights_final = F.softmax(self.fc_alpha.data, dim=-1)
            max_index = weights_final.max(-1, keepdim=True)[1].item()
            prune_ratio = 0.1 * max_index
            logger.info("decoder layer, weights: %r; prune ratio: %f" % (weights_final, prune_ratio))
            prune_num = int(prune_ratio * self._sorted_fc_indeces.size(0))
            self.fc_mask = self._sorted_fc_indeces[:prune_num]
            self._sorted_fc_indeces = None
        self.ffn._fc1.weight.data[self.fc_mask] = self.ffn._fc1.weight.data[self.fc_mask].fill_(0)
        if self.ffn._fc1.bias is not None:
            self.ffn._fc1.bias.data[self.fc_mask] = self.ffn._fc1.bias.data[self.fc_mask].fill_(0)
        self.ffn._fc2.weight.data[:, self.fc_mask] = self.ffn._fc2.weight.data[:, self.fc_mask].fill_(0)
    
    def set_tau(self, tau):
        self._tau = tau
