import os
from rdkit import Chem
import gzip
import pickle
from datetime import datetime
from fragmenizer import BRICS_RING_R_Fragmenizer
from utils import centralize, canonical_frag_smi

data_path = 'ZINC SDF PATH'
save_pkl_path = 'Your PATH'
save_pkl_pattern = 'BRICS_RING_R.{}.pkl'
save_pkl_file = os.path.join(save_pkl_path, save_pkl_pattern)
file_list = os.listdir(data_path)[:]

vocab = dict()

save_interval = 1000 * 10000
print_interval = 1000
mo_cnt = 0

fragmenizer = BRICS_RING_R_Fragmenizer()

start = datetime.now()
for f_idx, file_name in enumerate(file_list):
    file_path = os.path.join(data_path, file_name)
    gzip_data = gzip.open(file_path)
    with Chem.ForwardSDMolSupplier(gzip_data) as mos:
        for mo in mos:
            if mo is None:
                continue
            mo_cnt += 1

            frags, _ = fragmenizer.fragmenize(mo)
            frags = Chem.GetMolFrags(frags, asMols=True)
            for frag in frags:
                frag = centralize(frag)
                frag_smi = canonical_frag_smi(Chem.MolToSmiles(frag))
                
                if frag_smi not in vocab:
                    vocab[frag_smi] = frag
            
            if mo_cnt % save_interval == 0:
                with open(save_pkl_file.format(mo_cnt), 'wb') as fw:
                    pickle.dump(vocab, fw, protocol=pickle.HIGHEST_PROTOCOL)
            
            if mo_cnt % print_interval == 0:
                now = datetime.now()
                time_interval = (now - start).total_seconds()
                print('current {} file {} molecule {:.3f} ms/mol'.format(f_idx, mo_cnt, time_interval * 1000 / print_interval))
                start = datetime.now()

with open(save_pkl_file.format(mo_cnt), 'wb') as fw:
    pickle.dump(vocab, fw, protocol=pickle.HIGHEST_PROTOCOL)

def mapping_star(mol):
    star_mapping = dict()
    star_cnt = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            star_cnt += 1
            star_mapping[star_cnt] = atom.GetSmarts()
            star_mapping[atom.GetSmarts()] = star_cnt
    return star_mapping

vocab_w_mapping = {'PAD': [None, None, 0],'UNK': [None, None, 1], 'BOS': [None, None, 2], 'EOS': [None, None, 3], 'BOB': [None, None, 4], 'EOB': [None, None, 5]}
for key in vocab.keys():
    star_mapping = mapping_star(vocab[key])
    vocab_w_mapping[key] = [vocab[key], star_mapping, len(vocab_w_mapping)]

with open(save_pkl_file.format('vocab'), 'wb') as fw:
    pickle.dump(vocab, fw, protocol=pickle.HIGHEST_PROTOCOL)
