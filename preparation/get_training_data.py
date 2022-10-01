import os
from utils.utils import get_align_points
from rdkit import Chem
import gzip
import pickle
from datetime import datetime
from fragmenizer import BRICS_RING_R_Fragmenizer
from utils import centralize, canonical_frag_smi, get_center, get_align_points, \
    get_tree, tree_linearize, get_surrogate_frag, get_atom_mapping_between_frag_and_surrogate
import pickle as pkl
import rmsd
import numpy as np
from collections import defaultdict

data_path = 'ZINC SDF PATH'
save_pkl_path = 'YOUR PATH'
save_pkl_pattern = 'BRICS_RING_R.{}.pkl'
save_pkl_file = os.path.join(save_pkl_path, save_pkl_pattern)
file_list = os.listdir(data_path)[:]

vocab_path = 'VOCAB PATH'

data = []

save_interval = 50 * 10000
print_interval = 1000
mo_cnt = 0

fragmenizer = BRICS_RING_R_Fragmenizer()

with open(vocab_path, 'rb') as fr:
    vocab = pkl.load(fr)

start = datetime.now()
for f_idx, file_name in enumerate(file_list):
    file_path = os.path.join(data_path, file_name)
    gzip_data = gzip.open(file_path)
    with Chem.ForwardSDMolSupplier(gzip_data) as mos:
        for mo in mos:
            if mo is None:
                continue
            mo_cnt += 1

            # molecule itself
            curr_data = [mo]
            
            frags, _ = fragmenizer.fragmenize(mo)
            frags = Chem.GetMolFrags(frags, asMols=True)

            # fragment list of current molecule
            frags_list = []
            start_frag_idx = []
            cluster_dict = defaultdict(list)
            for idx, frag in enumerate(frags):
                frag_smi = canonical_frag_smi(Chem.MolToSmiles(frag))
                if frag_smi not in vocab:
                    # (frag_idx, frag_key, frag_smi, translate_vec, rotation_mat, rmsd_diff, attach_mapping)
                    frags_list.append((vocab['UNK'][2], 'UNK', frag_smi, None, None, None, None))

                    for atom in frag.GetAtoms():
                        if atom.GetSymbol() == '*':
                            cluster_dict[atom.GetSmarts()].append(idx)

                    if frag_smi.count('*') == 1:
                        start_frag_idx.append(idx)
                    continue
                
                frag_idx = vocab[frag_smi][2]
                frag_center = get_center(frag)
                
                v_frag = vocab[frag_smi][0]

                # get translation vector
                trans_vec = frag_center
                
                # get rotation matrix
                c_frag = centralize(frag)
                c_frag_sur = get_surrogate_frag(c_frag)
                v_frag_sur = get_surrogate_frag(v_frag)
                c_points, v_points, c2v_atom_mapping, v2c_atom_mapping = get_align_points(c_frag_sur, v_frag_sur)
                
                r_matrix = rmsd.kabsch(v_points, c_points)
                r_v_points = np.dot(v_points, r_matrix)
                diff = rmsd.rmsd(r_v_points, c_points)
                r_matrix = r_matrix.T

                # get attach_mapping
                v_frag_attach_mapping = vocab[frag_smi][1]
                c_frag2surro_atom_mapping, c_surro2frag_atom_mapping = \
                    get_atom_mapping_between_frag_and_surrogate(c_frag, c_frag_sur)
                v_frag2surro_atom_mapping, v_surro2frag_atom_mapping = \
                    get_atom_mapping_between_frag_and_surrogate(v_frag, v_frag_sur)
                
                c_frag_attach_mapping = dict()
                for c_atom in c_frag.GetAtoms():
                    if c_atom.GetSymbol() == '*':
                        c_smarts = c_atom.GetSmarts()
                        c_surro_idx = c_frag2surro_atom_mapping[c_atom.GetIdx()]
                        v_surro_idx = c2v_atom_mapping[c_surro_idx]
                        v_frag_idx = v_surro2frag_atom_mapping[v_surro_idx]
                        v_smarts = v_frag.GetAtomWithIdx(v_frag_idx).GetSmarts()
                        attach_idx = v_frag_attach_mapping[v_smarts]
                        c_frag_attach_mapping[c_smarts] = attach_idx
                        c_frag_attach_mapping[attach_idx] = c_smarts
                
                frags_list.append((frag_idx, frag_smi, frag_smi, trans_vec, r_matrix, diff, c_frag_attach_mapping))

                for atom in frag.GetAtoms():
                    if atom.GetSymbol() == '*':
                        cluster_dict[atom.GetSmarts()].append(idx)
                
                if frag_smi.count('*') == 1:
                    start_frag_idx.append(idx)

            curr_data.append(frags_list)
            
            # linear tree of molecule
            adj_dict = defaultdict(str)
            for key in cluster_dict.keys():
                l = cluster_dict[key]
                for i in range(len(l)):
                    for j in range(i + 1, len(l)):
                        adj_dict[(l[i], l[j])] = key
                        adj_dict[(l[j], l[i])] = key
            
            linear_trees = []
            for start_idx in start_frag_idx:
                tree = get_tree(adj_dict, start_idx, [], len(frags))
                linear_tree = []
                tree_linearize(tree, linear_tree)
                father_stack = []
                b_stack = []
                new_linear_tree = []
                new_linear_tree.append(('BOS', (None, None)))
                for node in linear_tree:
                    if node not in ['b', 'e']:
                        if len(father_stack) == 0:
                            new_linear_tree.append((node, (None, None)))
                        else:
                            father = father_stack[-1]
                            attach = adj_dict[(father, node)]
                            if frags_list[father][-1]:
                                father_attach = frags_list[father][-1][attach]
                            else:
                                father_attach = None
                            if frags_list[node][-1]:
                                son_attach = frags_list[node][-1][attach]
                            else:
                                son_attach = None
                            new_linear_tree.append((node, (father_attach, son_attach)))
                        father_stack.append(node)
                    elif node == 'b':
                        b_stack.append(len(father_stack))
                        new_linear_tree.append(('BOB', (None, None)))
                    elif node == 'e':
                        father_stack = father_stack[:b_stack.pop()]
                        new_linear_tree.append(('EOB', (None, None)))
                new_linear_tree.append(('EOS', (None, None)))
                linear_trees.append(new_linear_tree)

            curr_data.append(linear_trees)
            data.append(curr_data)

            if mo_cnt % save_interval == 0:
                with open(save_pkl_file.format(mo_cnt), 'wb') as fw:
                    pickle.dump(data, fw, protocol=pickle.HIGHEST_PROTOCOL)
            
            if mo_cnt % print_interval == 0:
                now = datetime.now()
                time_interval = (now - start).total_seconds()
                print('current {} file {} molecule {:.3f} ms/mol'.format(f_idx, mo_cnt, time_interval * 1000 / print_interval))
                start = datetime.now()

with open(save_pkl_file.format(mo_cnt), 'wb') as fw:
    pickle.dump(data, fw, protocol=pickle.HIGHEST_PROTOCOL)
