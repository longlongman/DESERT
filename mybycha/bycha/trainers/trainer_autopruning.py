from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)

import torch

from bycha.trainers import register_trainer
from bycha.trainers.trainer import Trainer
from bycha.utils.ops import merge_states
from bycha.utils.tensor import possible_autocast, to_device
from bycha.utils.profiling import profiler


@register_trainer
class AutoPruningTrainer(Trainer):

    def __init__(self,
                 search_fc_start=0,
                 search_fc_end=-1,
                 tau_max = 10.,
                 tau_min = 0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._search_fc_start = search_fc_start
        self._search_fc_end = search_fc_end
        self._tau_max = tau_max
        self._tau_min = tau_min

    def _step(self, samples, is_dummy=False):
        """
        Train a set of batches with only one gradient update.

        Args:
            samples: a set of batches

        Returns:
            logging_states: states to display in progress bar
        """
        
        if self._tot_step_cnt<=self._search_fc_end:
            new_tau = self._tau_max-(self._tau_max-self._tau_min)*self._tot_step_cnt/self._search_fc_end
            self._model._encoder.set_tau(new_tau)
            self._model._decoder.set_tau(new_tau)

        self._optimizer.zero_grad()
        samples = to_device(samples, device=self._env.device)
        logging_states = OrderedDict()
        for i, batch in enumerate(samples):
            self._model.reset(mode='train')
            with profiler.timeit("forward"):
                if self._enable_apex:
                    loss, logging_state = self._forward_loss(batch)
                else:
                    with possible_autocast():
                        loss, logging_state = self._forward_loss(batch)
            with profiler.timeit("backward"):
                self._backward_loss(loss)
            logging_states = merge_states(logging_states,
                                          logging_state,
                                          weight=1./(i+1.))
        
        if self._tot_step_cnt >= self._search_fc_start and \
            not 0<=self._search_fc_end<self._tot_step_cnt:
            self._sort_fc()

        if is_dummy:
            logger.info('dummy batch detected! set gradients to zero!')
            self._optimizer.multiply_grads(0.)

        with profiler.timeit("optimizer"):
            self._optimizer.step()
        
        if self._tot_step_cnt>self._search_fc_end>0:
            self._reset_fc_weights()

        return logging_states

    def _sort_fc(self):
        for coder in [self._model._encoder, self._model._decoder]:
            for layer in coder._layers:
                fc2_grad = layer.ffn._fc2.weight.grad
                fc2_weight = layer.ffn._fc2.weight.data
                importance = torch.einsum("dh,dh->h",[fc2_grad, fc2_weight])
                _, indeces = torch.sort(importance)
                layer._sorted_fc_indeces = indeces
    
    def _reset_fc_weights(self):
        for coder in [self._model._encoder, self._model._decoder]:
            for layer in coder._layers:
                layer.reset_fc_weights()       
