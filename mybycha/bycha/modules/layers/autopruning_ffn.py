import torch.nn as nn
import torch
import torch.nn.functional as F

from bycha.modules.utils import get_activation_fn
from bycha.modules.layers.gumbel import gumbel_softmax_topk

class AutoPruningFFN(nn.Module):
    """
    Feed-forward neural network

    Args:
        d_model: input feature dimension
        dim_feedforward: dimensionality of inner vector space
        dim_out: output feature dimensionality
        activation: activation function
        bias: requires bias in output linear function
    """

    def __init__(self,
                 d_model,
                 dim_feedforward=None,
                 dim_out=None,
                 activation="relu",
                 bias=True):
        super().__init__()
        self._dim_feedforward = dim_feedforward or d_model
        self._dim_out = dim_out or d_model
        self._bias = bias

        self._fc1 = nn.Linear(d_model, self._dim_feedforward)
        self._fc2 = nn.Linear(self._dim_feedforward, self._dim_out, bias=self._bias)
        self._activation = get_activation_fn(activation)

    def forward(self, x, weights=None, sorted_indeces=None, tau=0.):
        """
        Args:
            x: feature to perform feed-forward net
                :math:`(*, D)`, where D is feature dimension

        Returns:
            - feed forward output
                :math:`(*, D)`, where D is feature dimension
        """
        if weights is None or sorted_indeces is None or not self.training:
            x = self._fc1(x)
            x = self._activation(x)
            x = self._fc2(x)
            return x
        
        gumbel_onehot, _ = gumbel_softmax_topk(weights, tau=tau)  
        one_index = gumbel_onehot.max(-1, keepdim=True)[1].item()
        prune_ratio = gumbel_onehot[one_index] * one_index * 0.1
        prune_num = torch.floor(prune_ratio * self._dim_feedforward).long()

        x = F.linear(x, self._fc1.weight[sorted_indeces[prune_num:]], 
                        self._fc1.bias[sorted_indeces[prune_num:]]) if self._bias else \
            F.linear(x, self._fc1.weight[sorted_indeces[prune_num:]])
        x *= gumbel_onehot[one_index]       
        x = self._activation(x)
        x = F.linear(x, self._fc2.weight[:, sorted_indeces[prune_num:]], self._fc2.bias)
        return x

