import logging
logger = logging.getLogger(__name__)

from torch.nn import Module

from bycha.utils.runtime import Environment


class AbstractCriterion(Module):
    """
    Criterion is the base class for all the criterion within ByCha.
    """

    def __init__(self):
        super().__init__()
        self._model = None

    def build(self, *args, **kwargs):
        """
        Construct a criterion for model training.
        Typically, `model` should be provided.
        """
        self._build(*args, **kwargs)

        e = Environment()
        if e.device.startswith('cuda'):
            logger.info('move criterion to {}'.format(e.device))
            self.cuda(e.device)

    def _build(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """
        Compute the loss from neural model input, and produce a loss.
        """
        raise NotImplementedError

    def step_update(self, *args, **kwargs):
        """
        Perform step-level update
        """
        pass
