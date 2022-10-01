from typing import List
import logging
logger = logging.getLogger(__name__)

from bycha.dataloaders import AbstractDataLoader, register_dataloader
from bycha.datasets.abstract_dataset import AbstractDataset
from bycha.samplers.abstract_sampler import AbstractSampler
from bycha.samplers.distributed_sampler import DistributedSampler


@register_dataloader
class InMemoryDataLoader(AbstractDataLoader):
    """
    InMemoryDataLoader targets to sample and process data from InMemoryDataset

    Args:
        dataset (bycha.datasets.AbstractDataset): source dataset to load.
        sampler (bycha.samplers.AbstractSampler): sampler to retrieve data from the dataset with customized strategy.
        collate_fn: data process pipeline embedded in torch.utils.data.DataLoader
        post_collate_fn: data process pipeline after torch.utils.data.DataLoader,
            which can be adjusted withing a training epoch.
    """

    def __init__(self,
                 dataset: AbstractDataset,
                 sampler: AbstractSampler,
                 collate_fn=None,
                 post_collate_fn=None,
                 **kwargs):
        super().__init__(dataset,
                         sampler=None,
                         batch_sampler=sampler.batch_sampler,
                         collate_fn=collate_fn,
                         post_collate_fn=post_collate_fn,
                         **kwargs)
        self._sampler = sampler
        self._kwargs = kwargs

    def reset(self, epoch=0, *args, **kwargs):
        """
        Reset dataloader
        In torch, parameters of dataloader cannot be modified. Here we reset by re-build a new DataLoader with the same
        parameters.

        Args:
            epoch: training epoch
            step: training step

        Returns:
            dataloader (bycha.dataloaders.AbstractDataLoader): re-build a new DataLoader with possibly new collate_fn
        """
        self.dataset.reset()
        self._sampler.reset(epoch)
        return InMemoryDataLoader(self.dataset,
                                  sampler=self._sampler,
                                  collate_fn=self.collate_fn,
                                  post_collate_fn=self._post_collate_fn,
                                  **self._kwargs)

    def step_update(self, step, states=None):
        """
        Step-level updating on training states

        Args:
            step: learning steps
            states: states recorded in training process, and is used to update `sampler'
        """
        self._sampler.step_update(step, states)

    def epoch_update(self, epoch, states=None):
        """
        Epoch-level updating on training states

        Args:
            epoch: learning epoch
            states: states recorded in training process, and is used to update `sampler'
        """
        self._sampler.epoch_update(epoch, states)

    def __iter__(self):
        """
        Wrap the original data loading process with `post_collate`

        Returns:
            samples: a list of sample with `post_collate` process
        """
        for samples in super().__iter__():
            yield self._callback(samples)

    def finalize(self):
        """
        finalization
        """
        self._dataset.finalize()
        self._sampler.finalize()
