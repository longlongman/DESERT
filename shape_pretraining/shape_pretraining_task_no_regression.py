from bycha.tasks import register_task
from bycha.tasks.base_task import BaseTask
from bycha.models import create_model
from bycha.criteria import create_criterion
from bycha.generators import create_generator
from .utils import get_atom_stamp, get_shape, sample_augment, get_shape_patches, time_shift, get_grid_coords, get_rotation_bins, \
    get_atom_stamp_with_noise
import random
import numpy as np
import torch
from pytransform3d.rotations import quaternion_from_matrix
from math import ceil

@register_task
class ShapePretrainingTaskNoRegression(BaseTask):
    def __init__(self,
                 grid_resolution=1.0,
                 max_dist_stamp=3.0,
                 max_dist=10.0,
                 rotation_bin=24,
                 max_translation=1.0,
                 max_seq_len=20,
                 patch_size=3,
                 delta_input=False,
                 teacher_force_inference=False,
                 shape_noise_mu=0.0,
                 shape_noise_sigma=0.0,
                 rotation_bin_direction=11,
                 rotation_bin_angle=24,
                 **kwargs):
        super().__init__(**kwargs)
        self._grid_resolution = grid_resolution
        self._max_dist_stamp = max_dist_stamp
        self._max_dist = max_dist
        self._rotation_bin = rotation_bin # for data augmentation
        self._max_translation = max_translation
        self._atom_stamp = get_atom_stamp(grid_resolution, max_dist_stamp)
        self._max_seq_len = max_seq_len
        self._patch_size = patch_size
        self._delta_input = delta_input
        self._teacher_force_inference = teacher_force_inference
        self._box_size = ceil(2 * max_dist // grid_resolution + 1)
        self._rotation_bins = get_rotation_bins(rotation_bin_direction, rotation_bin_angle) # for mapping a fragment rotation of a bin
        self._shape_noise_mu = shape_noise_mu
        self._shape_noise_sigma = shape_noise_sigma
    
    def _build_datasets(self):
        super()._build_datasets()
        self._common_special_tokens = {'bos': self._datasets['train']._vocab['BOS'][2], # begining of sequence
                                       'eos': self._datasets['train']._vocab['EOS'][2], # end of sequence
                                       'bob': self._datasets['train']._vocab['BOB'][2], # begining of branch
                                       'eob': self._datasets['train']._vocab['EOB'][2], # end of branch
                                       'pad': self._datasets['train']._vocab['PAD'][2], # padding unit
                                       'unk': self._datasets['train']._vocab['UNK'][2]} # unknown unit
    
    def _build_models(self):
        self._model = create_model(self._model_configs)
        self._model.build(src_vocab_size=2,
                          tgt_vocab_size=len(self._datasets['train']._vocab),
                          src_special_tokens=self._common_special_tokens,
                          tgt_special_tokens=self._common_special_tokens)
    
    def _build_criterions(self):
        self._criterion = create_criterion(self._criterion_configs)
        self._criterion.build(self._model, padding_idx=self._datasets['train']._vocab['PAD'][2])
    
    def _build_generator(self):
        self._generator = create_generator(self._generator_configs)
        self._generator.build(self._model,
                              src_special_tokens=self._common_special_tokens,
                              tgt_special_tokens=self._common_special_tokens)
    
    def _collate(self, samples):
        shape = []
        shape_patches = []

        seq_len = []

        input_frag_idx = []
        output_frag_idx = []
        input_frag_idx_mask = []
        output_frag_idx_mask = []
        
        input_frag_trans = []
        output_frag_trans = []
        input_frag_trans_mask = []
        output_frag_trans_mask = []

        input_frag_r_mat = []
        output_frag_r_mat = []
        input_frag_r_mat_mask = []
        output_frag_r_mat_mask = []
        
        if not self._infering or (self._infering and self._teacher_force_inference):
            for sample in samples:
                if len(sample['tree_list']) == 0:
                    continue

                # ---------------------------- augment samples by rotation and translation -------------------------------------------------
                if not self._infering:
                    sample = sample_augment(sample, self._rotation_bin, self._max_translation)
                elif self._infering and self._teacher_force_inference:
                    sample = sample
                else:
                    raise Exception('please make sure [self._infering is False] or [(self._infering and self._teacher_force_inference) is True]!')
                
                # --------------------------------- calculate the shape of molecules -------------------------------------------------------
                curr_atom_stamp = get_atom_stamp_with_noise(self._grid_resolution, 
                                                            self._max_dist_stamp, 
                                                            self._shape_noise_mu,
                                                            self._shape_noise_sigma)
                curr_shape = get_shape(sample['mol'], 
                                       curr_atom_stamp, 
                                       self._grid_resolution, 
                                       self._max_dist)
                shape.append(curr_shape)
                
                # -------------------------------- calculate the patches of the shape -------------------------------------------------------
                curr_shape_patches = get_shape_patches(curr_shape, self._patch_size)
                curr_shape_patches = curr_shape_patches.reshape(curr_shape.shape[0] // self._patch_size,
                                                                curr_shape.shape[0] // self._patch_size,
                                                                curr_shape.shape[0] // self._patch_size, -1)
                curr_shape_patches = curr_shape_patches.reshape(-1, self._patch_size**3)
                shape_patches.append(curr_shape_patches)

                # ------------------------------------- create training sequence -------------------------------------------------------------
                random_tree = random.choice(sample['tree_list'])
                
                curr_idx = [] # fragment idx in vocab
                curr_idx_mask = []
                curr_trans = [] # fragment centroid position
                curr_trans_mask = []
                curr_r_mat = [] # fragment roation in quaternion
                curr_r_mat_mask = []
                curr_seq_len = 0
                
                for unit in random_tree:
                    curr_seq_len += 1
                    # ------------------ not a special token ------------------
                    if unit[0] not in ['BOS', 'EOS', 'BOB', 'EOB']:
                        curr_frag = sample['frag_list'][unit[0]]
                        curr_idx.append(curr_frag['vocab_id'])
                        curr_idx_mask.append(1)
                        # ------------------ known fragment ------------------
                        if curr_frag['vocab_id'] != self._datasets['train']._vocab['UNK'][2]:
                            curr_trans_grid_coords = get_grid_coords(curr_frag['trans_vec'], self._max_dist, self._grid_resolution)

                            # ------------------ map position to bin ------------------
                            # the center of current fragment not in the box
                            if (curr_trans_grid_coords[0] < 0 or curr_trans_grid_coords[0] >= self._box_size) or \
                               (curr_trans_grid_coords[1] < 0 or curr_trans_grid_coords[1] >= self._box_size) or \
                               (curr_trans_grid_coords[2] < 0 or curr_trans_grid_coords[2] >= self._box_size):
                                curr_trans.append(1) # out of the box
                                curr_trans_mask.append(1)
                            else:
                                pos_bin = curr_trans_grid_coords[0] * self._box_size**2 + \
                                          curr_trans_grid_coords[1] * self._box_size + \
                                          curr_trans_grid_coords[2] + 2 # plus 2, because 0 for not a fragment, 1 for out of box
                                curr_trans.append(pos_bin)
                                curr_trans_mask.append(1)

                            # ------------------ map rotation to bin ------------------
                            tmp = self._rotation_bins - curr_frag['rotate_mat']
                            tmp = abs(tmp).sum(axis=-1).sum(axis=-1)
                            min_index = np.argmin(tmp)
                            curr_r_mat.append(min_index + 1) # 0 for not a fragment
                            curr_r_mat_mask.append(1)
                        # ----------------- unknown fragment -----------------
                        else:
                            curr_trans.append(0)
                            curr_trans_mask.append(0)
                            curr_r_mat.append(0)
                            curr_r_mat_mask.append(0)
                    # --------------------- special tokens --------------------
                    else:
                        if unit[0] == 'BOS':
                            curr_idx.append(self._datasets['train']._vocab['BOS'][2]) # BOS
                        elif unit[0] == 'EOS':
                            curr_idx.append(self._datasets['train']._vocab['EOS'][2]) # EOS
                        elif unit[0] == 'BOB':
                            curr_idx.append(self._datasets['train']._vocab['BOB'][2]) # BOB
                        elif unit[0] == 'EOB':
                            curr_idx.append(self._datasets['train']._vocab['EOB'][2]) # EOB
                        
                        curr_idx_mask.append(1)
                        curr_trans.append(0)
                        curr_trans_mask.append(0)
                        curr_r_mat.append(0)
                        curr_r_mat_mask.append(0)
                
                # --------------------- create shifted sequence --------------------
                input_curr_idx, output_curr_idx = time_shift(curr_idx)
                input_curr_idx_mask, output_curr_idx_mask = time_shift(curr_idx_mask)

                # -------------------- create delta translation --------------------
                if self._delta_input:
                    delta = []
                    pre_trans = np.zeros(1)
                    for tr, tr_m in zip(curr_trans, curr_trans_mask):
                        if tr_m != 0:
                            delta.append(tr - pre_trans)
                            pre_trans = tr
                        else:
                            delta.append(np.zeros(1))
                    curr_trans = delta

                input_curr_trans, output_curr_trans = time_shift(curr_trans)
                input_curr_trans_mask, output_curr_trans_mask = time_shift(curr_trans_mask)

                input_curr_r_mat, output_curr_r_mat = time_shift(curr_r_mat)
                input_curr_r_mat_mask, output_curr_r_mat_mask = time_shift(curr_r_mat_mask)

                curr_seq_len -= 1
                
                # --------------------- create truncated sequence --------------------
                if self._training:
                    input_curr_idx = input_curr_idx[:self._max_seq_len]
                    output_curr_idx = output_curr_idx[:self._max_seq_len]
                    input_curr_idx_mask = input_curr_idx_mask[:self._max_seq_len]
                    output_curr_idx_mask = output_curr_idx_mask[:self._max_seq_len]

                    input_curr_trans = input_curr_trans[:self._max_seq_len]
                    output_curr_trans = output_curr_trans[:self._max_seq_len]
                    input_curr_trans_mask = input_curr_trans_mask[:self._max_seq_len]
                    output_curr_trans_mask = output_curr_trans_mask[:self._max_seq_len]

                    input_curr_r_mat = input_curr_r_mat[:self._max_seq_len]
                    output_curr_r_mat = output_curr_r_mat[:self._max_seq_len]
                    input_curr_r_mat_mask = input_curr_r_mat_mask[:self._max_seq_len]
                    output_curr_r_mat_mask = output_curr_r_mat_mask[:self._max_seq_len]

                    curr_seq_len = min(curr_seq_len, self._max_seq_len)
                
                input_frag_idx.append(np.array(input_curr_idx))
                input_frag_idx_mask.append(np.array(input_curr_idx_mask))

                output_frag_idx.append(np.array(output_curr_idx))
                output_frag_idx_mask.append(np.array(output_curr_idx_mask))

                input_frag_trans.append(np.array(input_curr_trans))
                input_frag_trans_mask.append(np.array(input_curr_trans_mask))

                output_frag_trans.append(np.array(output_curr_trans))
                output_frag_trans_mask.append(np.array(output_curr_trans_mask))

                input_frag_r_mat.append(np.array(input_curr_r_mat))
                input_frag_r_mat_mask.append(np.array(input_curr_r_mat_mask))

                output_frag_r_mat.append(np.array(output_curr_r_mat))
                output_frag_r_mat_mask.append(np.array(output_curr_r_mat_mask))

                seq_len.append(curr_seq_len)
            
            # --------------------- create padded sequence --------------------
            max_seq_len = max(seq_len)
            for i in range(len(input_frag_idx)):
                pad_input_frag_idx = np.zeros(max_seq_len) # pad 0
                pad_input_frag_idx[:len(input_frag_idx[i])] = input_frag_idx[i]
                pad_input_frag_idx_mask = np.zeros(max_seq_len) # pad 0
                pad_input_frag_idx_mask[:len(input_frag_idx_mask[i])] = input_frag_idx_mask[i]

                pad_output_frag_idx = np.zeros(max_seq_len) # pad 0
                pad_output_frag_idx[:len(output_frag_idx[i])] = output_frag_idx[i]
                pad_output_frag_idx_mask = np.zeros(max_seq_len) # pad 0
                pad_output_frag_idx_mask[:len(output_frag_idx_mask[i])] = output_frag_idx_mask[i]

                pad_input_frag_trans = np.zeros((max_seq_len,)) # pad 0
                pad_input_frag_trans[:len(input_frag_trans[i])] = input_frag_trans[i]
                pad_input_frag_trans_mask = np.zeros(max_seq_len) # pad 0
                pad_input_frag_trans_mask[:len(input_frag_trans_mask[i])] = input_frag_trans_mask[i]

                pad_output_frag_trans = np.zeros((max_seq_len, )) # pad 0
                pad_output_frag_trans[:len(output_frag_trans[i])] = output_frag_trans[i]
                pad_output_frag_trans_mask = np.zeros(max_seq_len) # pad 0
                pad_output_frag_trans_mask[:len(output_frag_trans_mask[i])] = output_frag_trans_mask[i]

                pad_input_frag_r_mat = np.zeros((max_seq_len, )) # pad 0
                pad_input_frag_r_mat[:len(input_frag_r_mat[i])] = input_frag_r_mat[i]
                pad_input_frag_r_mat_mask = np.zeros(max_seq_len) # pad 0
                pad_input_frag_r_mat_mask[:len(input_frag_r_mat_mask[i])] = input_frag_r_mat_mask[i]

                pad_output_frag_r_mat = np.zeros((max_seq_len, )) # pad 0
                pad_output_frag_r_mat[:len(output_frag_r_mat[i])] = output_frag_r_mat[i]
                pad_output_frag_r_mat_mask = np.zeros(max_seq_len) # pad 0
                pad_output_frag_r_mat_mask[:len(output_frag_r_mat_mask[i])] = output_frag_r_mat_mask[i]
                
                input_frag_idx[i] = pad_input_frag_idx
                input_frag_idx_mask[i] = pad_input_frag_idx_mask

                output_frag_idx[i] = pad_output_frag_idx
                output_frag_idx_mask[i] = pad_output_frag_idx_mask

                input_frag_trans[i] = pad_input_frag_trans
                input_frag_trans_mask[i] = pad_input_frag_trans_mask

                output_frag_trans[i] = pad_output_frag_trans
                output_frag_trans_mask[i] = pad_output_frag_trans_mask

                input_frag_r_mat[i] = pad_input_frag_r_mat
                input_frag_r_mat_mask[i] = pad_input_frag_r_mat_mask

                output_frag_r_mat[i] = pad_output_frag_r_mat
                output_frag_r_mat_mask[i] = pad_output_frag_r_mat_mask
            
            shape = torch.tensor(np.array(shape), dtype=torch.long)
            shape_patches = torch.tensor(np.array(shape_patches), dtype=torch.float)
            
            input_frag_idx = torch.tensor(np.array(input_frag_idx), dtype=torch.long)
            input_frag_idx_mask = torch.tensor(np.array(input_frag_idx_mask), dtype=torch.float)

            output_frag_idx = torch.tensor(np.array(output_frag_idx), dtype=torch.long)
            output_frag_idx_mask = torch.tensor(np.array(output_frag_idx_mask), dtype=torch.float)

            input_frag_trans = torch.tensor(np.array(input_frag_trans), dtype=torch.long)
            input_frag_trans_mask = torch.tensor(np.array(input_frag_trans_mask), dtype=torch.float)

            output_frag_trans = torch.tensor(np.array(output_frag_trans), dtype=torch.long)
            output_frag_trans_mask = torch.tensor(np.array(output_frag_trans_mask), dtype=torch.float)

            input_frag_r_mat = torch.tensor(np.array(input_frag_r_mat), dtype=torch.long)
            input_frag_r_mat_mask= torch.tensor(np.array(input_frag_r_mat_mask), dtype=torch.float)

            output_frag_r_mat = torch.tensor(np.array(output_frag_r_mat), dtype=torch.long)
            output_frag_r_mat_mask = torch.tensor(np.array(output_frag_r_mat_mask), dtype=torch.float)

            if not self._infering:
                batch = {
                    'net_input':{
                        'shape': shape, # [batch_size, box_size, box_size, box_size]
                        'shape_patches': shape_patches, # [batch_size, (box_size // patch_size)**3, patch_size**3]
                        
                        'input_frag_idx': input_frag_idx, # [batch_size, seq_len]
                        'input_frag_idx_mask': input_frag_idx_mask, # [batch_size, seq_len]
                        
                        'input_frag_trans': input_frag_trans, # [batch_size, seq_len, 3]
                        'input_frag_trans_mask': input_frag_trans_mask, # [batch_size, seq_len]
                        
                        'input_frag_r_mat': input_frag_r_mat, # [batch_size, seq_len, 4]
                        'input_frag_r_mat_mask': input_frag_r_mat_mask, # [batch_size, seq_len]
                    },
                    'net_output':{
                        'output_frag_idx': output_frag_idx, # [batch_size, seq_len]
                        'output_frag_idx_mask': output_frag_idx_mask, # [batch_size, seq_len]

                        'output_frag_trans': output_frag_trans, # [batch_size, seq_len, 3]
                        'output_frag_trans_mask': output_frag_trans_mask, # [batch_size, seq_len]

                        'output_frag_r_mat': output_frag_r_mat, # [batch_size, seq_len, 4]
                        'output_frag_r_mat_mask': output_frag_r_mat_mask, # [batch_size, seq_len]
                    }
                }
            elif self._infering and self._teacher_force_inference:
                net_input = {
                    'encoder': (shape_patches,),
                    'decoder': ((input_frag_idx, input_frag_trans, input_frag_r_mat, True),),
                }

                batch = {'net_input': net_input}
            else:
                raise Exception('please make sure [self._infering is False] or [(self._infering and self._teacher_force_inference) is True]!')
        else:
            for sample in samples:
                curr_atom_stamp = get_atom_stamp_with_noise(self._grid_resolution, 
                                                            self._max_dist_stamp, 
                                                            self._shape_noise_mu,
                                                            self._shape_noise_sigma)
                curr_shape = get_shape(sample['mol'], 
                                       curr_atom_stamp, 
                                       self._grid_resolution, 
                                       self._max_dist)
                shape.append(curr_shape)

                curr_shape_patches = get_shape_patches(curr_shape, self._patch_size)
                curr_shape_patches = curr_shape_patches.reshape(curr_shape.shape[0] // self._patch_size,
                                                                curr_shape.shape[0] // self._patch_size,
                                                                curr_shape.shape[0] // self._patch_size, -1)
                curr_shape_patches = curr_shape_patches.reshape(-1, self._patch_size**3)
                shape_patches.append(curr_shape_patches)

                input_frag_idx.append(np.array([self._datasets['train']._vocab['BOS'][2]])) # BOS
                input_frag_trans.append(np.zeros((1, )))
                input_frag_r_mat.append(np.zeros((1, )))

            
            shape = torch.tensor(np.array(shape), dtype=torch.long)
            
            shape_patches = torch.tensor(np.array(shape_patches), dtype=torch.float)
            
            input_frag_idx = torch.tensor(np.array(input_frag_idx), dtype=torch.long)
            input_frag_trans = torch.tensor(np.array(input_frag_trans), dtype=torch.long)
            input_frag_r_mat = torch.tensor(np.array(input_frag_r_mat), dtype=torch.long)
            
            net_input = {
                'encoder': (shape_patches,),
                'decoder': ((input_frag_idx, input_frag_trans, input_frag_r_mat, False),),
            }

            batch = {'net_input': net_input}
        return batch

    def _output_collate_fn(self, outputs, *args, **kwargs):
        processed_outputs = []
        for idxs, trs, rms in zip(outputs[0].cpu().tolist(), outputs[1].cpu().tolist(), outputs[2].cpu().tolist()):
            po = []
            for idx, tr, rm in zip(idxs, trs, rms):
                if isinstance(idx, list):
                    ppo = []
                    for i, t, r in zip(idx, tr, rm):
                        ppo.append((i, t, r))
                    po.append(ppo)
                else:
                    po.append((idx, tr, rm))
            processed_outputs.append(po)
        return processed_outputs
