import pickle as pkl
from shape_utils import get_atom_stamp
from shape_utils import get_shape
from rdkit import Chem
from shape_utils import ROTATIONS
from random import sample
from shape_utils import centralize
from shape_utils import get_mol_centroid
from shape_utils import trans
from rdkit.Chem import rdMolTransforms
import copy
from shape_utils import get_binary_features
from tfbio_data import make_grid
import numpy as np

data_path = '--TRAINING DATA PATH--'
with open(data_path, 'rb') as fr:
    data = pkl.load(fr)

atom_stamp = get_atom_stamp(grid_resolution=0.5, max_dist=4.0)

cavity_file_path = '--CAVITY PDB PATH--'
cavity = Chem.MolFromPDBFile(cavity_file_path, proximityBonding=False)

protein_file_path = '--PROTEIN PDB PATH--'
protein = Chem.MolFromPDBFile(protein_file_path, proximityBonding=False)

cavity_centroid = get_mol_centroid(cavity)
cavity = centralize(cavity)
translation = trans(-cavity_centroid[0], -cavity_centroid[1], -cavity_centroid[2]) 
protein_conformer = protein.GetConformer()
rdMolTransforms.TransformConformer(protein_conformer, translation) # move protein according to cavity centroid

sample_shapes = []
sample_n_o_f = []
for i in range(200):
    print(i)
    i = i % 24

    copied_cavity = copy.deepcopy(cavity)
    copied_protein = copy.deepcopy(protein)

    cavity_conformer = copied_cavity.GetConformer()
    protein_conformer = copied_protein.GetConformer()

    rotation_mat = ROTATIONS[i]
    rotation = np.zeros((4, 4))
    rotation[:3, :3] = rotation_mat
    rdMolTransforms.TransformConformer(cavity_conformer, rotation)
    rdMolTransforms.TransformConformer(protein_conformer, rotation)

    curr_cavity_shape = get_shape(copied_cavity, atom_stamp, 0.5, 15)
    large_cavity_shape = np.zeros((61*3, 61*3, 61*3))
    large_cavity_shape[61*1:61*2,61*1:61*2,61*1:61*2] = curr_cavity_shape

    protein_coords, protein_features = get_binary_features(copied_protein, -1, False)
    protein_grid, feature_dict = make_grid(protein_coords, protein_features, 0.5, 15)
    protein_grid = protein_grid.squeeze()
    
    n_o_f_grid = np.zeros(protein_grid.shape)
    for xyz in feature_dict[(7.0,)] + feature_dict[(8.0,)] + feature_dict[(9.0,)]:
        x, y, z = xyz[0], xyz[1], xyz[2]
        
        x_left = x - 4 if x - 4 >=0 else 0
        x_right = x + 4 if x + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1
        
        y_left = y - 4 if y - 4 >=0 else 0
        y_right = y + 4 if y + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

        z_left = z - 4 if z - 4 >=0 else 0
        z_right = z + 4 if z + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

        tmp = n_o_f_grid[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
        tmp += 1
    large_n_o_f_grid = np.zeros((61*3, 61*3, 61*3))
    large_n_o_f_grid[61*1:61*2,61*1:61*2,61*1:61*2] = n_o_f_grid
    
    seed_data = sample(data, 10)
    union_shape = np.zeros((61, 61, 61))
    for seed in seed_data:
        mol_shape = get_shape(centralize(seed[0]), atom_stamp, 0.5, 15)
        union_shape = union_shape + mol_shape
    union_shape[union_shape>1]=1

    flag = False
    for j in range(0, 122):
        large_union_shape = np.zeros((61*3, 61*3, 61*3))
        large_union_shape[j: j + 61, j: j + 61, j: j + 61] = union_shape
        inter_shape = large_cavity_shape * large_union_shape
        if inter_shape.sum() > 2400:
            flag = True
            break
    if flag:
        inter_idx = np.where(inter_shape > 0)
        x, y, z = inter_idx[0].mean(), inter_idx[1].mean(), inter_idx[2].mean()
        x, y, z = int(x.round()), int(y.round()), int(z.round())
        x_left, x_right = x - 13, x + 14 + 1
        y_left, y_right = y - 13, y + 14 + 1
        z_left, z_right = z - 13, z + 14 + 1
        inter_shape = inter_shape[x_left: x_right, y_left: y_right, z_left: z_right]
        inter_n_o_f = large_n_o_f_grid[x_left: x_right, y_left: y_right, z_left: z_right]
        sample_shapes.append(inter_shape)
        sample_n_o_f.append(inter_n_o_f)

sample_shapes.append(get_shape(cavity, atom_stamp, 0.5, 6.75))

protein_coords, protein_features = get_binary_features(protein, -1, False)
protein_grid, feature_dict = make_grid(protein_coords, protein_features, 0.5, 6.75)
protein_grid = protein_grid.squeeze()
n_o_f_grid = np.zeros(protein_grid.shape)
for xyz in feature_dict[(7.0,)] + feature_dict[(8.0,)] + feature_dict[(9.0,)]:
    x, y, z = xyz[0], xyz[1], xyz[2]
    
    x_left = x - 4 if x - 4 >=0 else 0
    x_right = x + 4 if x + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1
    
    y_left = y - 4 if y - 4 >=0 else 0
    y_right = y + 4 if y + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

    z_left = z - 4 if z - 4 >=0 else 0
    z_right = z + 4 if z + 4 < protein_grid.shape[0] else protein_grid.shape[0] - 1

    tmp = n_o_f_grid[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
    tmp += 1
sample_n_o_f.append(n_o_f_grid)

with open('--YOUR SAVE PATH--', 'wb') as fw:
    pkl.dump(sample_shapes, fw)

with open('--YOUR SAVE PATH--', 'wb') as fw:
    pkl.dump(sample_n_o_f, fw)