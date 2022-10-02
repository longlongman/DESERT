import pickle
from bycha.datasets import register_dataset
from bycha.datasets.in_memory_dataset import InMemoryDataset
from bycha.utils.runtime import logger, progress_bar
from .utils import get_mol_centroid, centralize

@register_dataset
class ShapePretrainingDatasetPocket(InMemoryDataset):
    def __init__(self,
                 path):
        super().__init__(path)

        vocab_path = self._path['vocab']
        with open(vocab_path, 'rb') as fr:
            self._vocab = pickle.load(fr)
    
    def _load(self):
        self._data = []
        
        samples_path = self._path['samples']
        
        with open(samples_path, 'rb') as fr:
            samples = pickle.load(fr)
        
        accecpted, discarded = 0, 0
        for i, sample in enumerate(progress_bar(samples, desc='Loading Samples...')):
            try:
                self._data.append(self._full_callback(sample))
                accecpted += 1
            except Exception:
                logger.warning('sample {} is discarded'.format(i))
                discarded += 1
        
        self._length = len(self._data)
        logger.info(f'Totally accept {accecpted} samples, discard {discarded} samples')
    
    def _callback(self, sample):
        return {
            'shape': sample
        }
