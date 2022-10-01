from bycha.trainers.trainer import Trainer
from bycha.trainers import register_trainer

@register_trainer
class MoETrainer(Trainer):

    def __init__(self,
                 load_balance_alpha=0.,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_balance_alpha = load_balance_alpha

    def _forward_loss(self, samples):
        """
        Forward neural model and compute the loss of given samples

        Args:
            samples: a batch of samples

        Returns:
            - derived loss as torch.Tensor
            - states for updating log
        """
        loss, logging_states = self._criterion(**samples)
        loss, logging_states = self._load_balance_loss(loss, logging_states)
        return loss, logging_states

    def _load_balance_loss(self, loss, logging_states):
        moe_loss = self._model._encoder.moe_loss + self._model._decoder.moe_loss
        moe_loss /= 2
        loss += moe_loss*self._load_balance_alpha
        logging_states['moe_loss'] = moe_loss.data.item()
        return loss, logging_states

