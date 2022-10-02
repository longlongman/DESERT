from bycha.utils.io import _InputStream, _OutputStream, _InputBytes, _OutputBytes
import pickle
from bycha.utils.runtime import logger
import random
from bycha.utils.ops import local_seed

class _MyInputBytes(_InputBytes):
    def __init__(self, fake_epoch, *args, shuffle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_iter = None
        self._shuffle = shuffle
        self._fake_epoch = fake_epoch
    
    def __next__(self):
        try:
            if self._idx >= len(self._fins):
                raise IndexError
            if not self._data_iter:
                data = pickle.load(self._fins[self._idx])
                if self._shuffle:
                    with local_seed((self._fake_epoch * len(self._fins)) + self._idx):
                        old_state = random.getstate()
                        random.seed((self._fake_epoch * len(self._fins)) + self._idx)
                        random.shuffle(data)
                        random.setstate(old_state)
                self._data_iter = iter(data)
            sample = next(self._data_iter)
            return sample
        except StopIteration:
            self._idx += 1
            self._data_iter = None
            sample = self.__next__()
            return sample
        except IndexError:
            raise StopIteration
    
    def reset(self):
        self._idx = 0
        for fin in self._fins:
            fin.seek(0)
        self._data_iter = None

class MyUniIO(_InputStream, _OutputStream, _MyInputBytes, _OutputBytes):
    def __init__(self, path, fake_epoch, mode='r', encoding='utf8', shuffle=False):
        pass

    def __new__(cls, path, fake_epoch, mode='r', encoding='utf8', shuffle=False):
        if 'r' in mode.lower():
            if 'b' in mode.lower():
                return _MyInputBytes(fake_epoch, path, mode=mode, shuffle=shuffle)
            return _InputStream(path, encoding=encoding)
        elif 'w' in mode.lower():
            if 'b' in mode.lower():
                return _OutputBytes(path, mode=mode)
            return _OutputStream(path, encoding=encoding)
        logger.warning(f'Not support file mode: {mode}')
        raise ValueError
