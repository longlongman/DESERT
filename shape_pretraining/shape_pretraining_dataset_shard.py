from bycha.datasets import register_dataset
from bycha.datasets.streaming_dataset import StreamingDataset
from .io import MyUniIO
from bycha.utils.runtime import logger
import pickle
from .utils import get_mol_centroid, centralize, set_atom_prop

@register_dataset
class ShapePretrainingDatasetShard(StreamingDataset):
    def __init__(self,
                 path,
                 vocab_path,
                 sample_each_shard,
                 shuffle=False):
        super().__init__(path)
        self._sample_each_shard = sample_each_shard
        self._shuffle = shuffle
        self._fake_epoch = 0

        with open(vocab_path, 'rb') as fr:
            self._vocab = pickle.load(fr)
    
    def build(self, collate_fn=None, preprocessed=False):
        self._collate_fn = collate_fn
        if self._path:
            self._fin = MyUniIO(self._path, self._fake_epoch, mode='rb', shuffle=self._shuffle)
    
    def __iter__(self):
        for sample in self._fin:
            try:
                sample = self._full_callback(sample)
                yield sample
            except StopIteration:
                raise StopIteration
            except Exception as e:
                logger.warning(e)
    
    def reset(self):
        self._pos = 0
        self._fin = MyUniIO(self._path, self._fake_epoch, mode='rb', shuffle=self._shuffle)
        self._fake_epoch += 1
    
    def _callback(self, sample):
        # centralize a molecule and translate its fragments
        mol = sample[0]
        centroid = get_mol_centroid(mol)
        mol = centralize(mol)

        for atom in mol.GetAtoms():
            set_atom_prop(atom, 'origin_atom_idx', str(atom.GetIdx()))
        
        fragment_list = []
        for fragment in sample[1]:
            if not fragment[3] is None:
                trans_vec = fragment[3] - centroid
            else:
                trans_vec = fragment[3]
            fragment_list.append({
                'vocab_id': fragment[0],
                'vocab_key': fragment[1],
                'frag_smi': fragment[2],
                'trans_vec': trans_vec,
                'rotate_mat': fragment[4]
            })
        
        tree_list = sample[2]

        return {
            'mol': mol,
            'frag_list': fragment_list,
            'tree_list': tree_list
        }

    def finalize(self):
        self._fin.close()
