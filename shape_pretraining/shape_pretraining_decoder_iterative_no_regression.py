from bycha.modules.decoders import register_decoder
from bycha.modules.decoders.transformer_decoder import TransformerDecoder
from bycha.modules.utils import create_time_mask
from bycha.modules.layers.feed_forward import FFN
import torch
import torch.nn as nn
from bycha.modules.encoders import create_encoder
from math import ceil

@register_decoder
class ShapePretrainingDecoderIterativeNoRegression(TransformerDecoder):
    def __init__(self, 
                 *args, 
                 iterative_block, 
                 iterative_num=1, 
                 max_dist=10.0, 
                 grid_resolution=1.0, 
                 rotation_bin_direction=11,
                 rotation_bin_angle=24,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._iterative_block_config = iterative_block
        self._iterative_num = iterative_num
        self._box_size = ceil(2 * max_dist // grid_resolution + 1)
        self._rotation_bin_direction = rotation_bin_direction
        self.rotation_bin_angle = rotation_bin_angle
    
    def build(self,
              embed,
              special_tokens,
              out_proj):
        super().build(embed, special_tokens, out_proj)
        from bycha.modules.layers.embedding import Embedding
        
        self._trans_emb = Embedding(vocab_size=self._box_size ** 3 + 2,
                                    d_model=self._d_model)
        self._rotat_emb = Embedding(vocab_size=self._rotation_bin_direction * self._rotation_bin_direction * 3 * (self.rotation_bin_angle - 1) + 1 + 1,
                                    d_model=self._d_model)
        
        self._trans_output_proj = nn.Linear(self._trans_emb.weight.shape[1],
                                            self._trans_emb.weight.shape[0],
                                            bias=False)
        self._trans_output_proj.weight = self._trans_emb.weight

        self._rotat_output_proj = nn.Linear(self._rotat_emb.weight.shape[1],
                                            self._rotat_emb.weight.shape[0],
                                            bias=False)
        self._rotat_output_proj.weight = self._rotat_emb.weight

        iterative_block_emb =  Embedding(vocab_size=embed.weight.shape[0], d_model=embed.weight.shape[1], padding_idx=embed.padding_idx)
        self._iterative_block = create_encoder(self._iterative_block_config)
        self._iterative_block.build(iterative_block_emb, special_tokens, self._trans_emb.weight.shape[0], self._rotat_emb.weight.shape[0])
    
    def forward(self, 
                input_frag_idx,
                input_frag_trans,
                input_frag_r_mat,
                memory,
                memory_padding_mask):
        tgt = input_frag_idx
        
        x = self._embed(tgt)
        
        input_frag_trans = self._trans_emb(input_frag_trans)

        input_frag_r_mat = self._rotat_emb(input_frag_r_mat)

        x = x + input_frag_trans

        x = x + input_frag_r_mat

        x = x * self._embed_scale

        if self._pos_embed is not None:
            x = x + self._pos_embed(tgt)
        x = self._embed_dropout(x)
        
        x = x.transpose(0, 1)

        tgt_mask = create_time_mask(tgt)
        tgt_padding_mask = tgt.eq(self._special_tokens['pad'])
        for layer in self._layers:
            x = layer(tgt=x,
                      memory=memory,
                      tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_padding_mask,
                      memory_key_padding_mask=memory_padding_mask,)
        if self._norm is not None:
            x = self._norm(x)
        x = x.transpose(0, 1)
        logits = self._out_proj(x)
        if self._out_proj_bias is not None:
            logits = logits + self._out_proj_bias
        
        trans = self._trans_output_proj(x)
        r_mat = self._rotat_output_proj(x)

        ret_logits = [logits]
        ret_trans = [trans]
        ret_r_mat = [r_mat]

        if self._mode != 'infer':
            for _ in range(self._iterative_num):
                logits, trans, r_mat = self._iterative_block(logits, trans, r_mat, tgt_padding_mask)
                ret_logits.append(logits)
                ret_trans.append(trans)
                ret_r_mat.append(r_mat)

        return ret_logits, ret_trans, ret_r_mat

    def reset(self, mode='train'):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        for layer in self._layers:
            layer.reset(mode)
        self._iterative_block.reset(mode)

    def iterative_infer(self, frag_idx, frag_trans, frag_r_mat):
        assert self._mode == 'infer'

        bz, sl = frag_idx.shape[0], frag_idx.shape[1]

        padding_mask = []
        for s in frag_idx:
            cnt = 0
            for idx in s:
                if idx != self._special_tokens['eos']:
                    cnt += 1
                else:
                    cnt += 1 # include an EOS token
                    break
            curr_mask = frag_idx.new_ones(sl)
            curr_mask[:cnt] = 0.0
            padding_mask.append(curr_mask.bool())
        padding_mask = torch.stack(padding_mask, dim=0)
        
        logits = frag_idx.new_zeros((bz * sl, self._embed.weight.shape[0]))
        tmp = frag_idx.new_ones((bz * sl, 1))
        frag_idx = frag_idx.contiguous().view(bz * sl, 1)
        logits = logits.scatter(-1, frag_idx, tmp)
        logits = logits.view(bz, sl, -1)

        trans = frag_trans.new_zeros((bz * sl, self._trans_emb.weight.shape[0]))
        tmp = frag_trans.new_ones((bz * sl, 1))
        frag_trans = frag_trans.contiguous().view(bz * sl, 1)
        trans = trans.scatter(-1, frag_trans, tmp)
        trans = trans.view(bz, sl, -1)

        r_mat = frag_r_mat.new_zeros((bz * sl, self._rotat_emb.weight.shape[0]))
        tmp = frag_r_mat.new_ones((bz * sl, 1))
        frag_r_mat = frag_r_mat.contiguous().view(bz * sl, 1)
        r_mat = r_mat.scatter(-1, frag_r_mat, tmp)
        r_mat = r_mat.view(bz, sl, -1)

        for _ in range(self._iterative_num):
            logits, trans, r_mat = self._iterative_block(logits, trans, r_mat, padding_mask)
        
        return logits, trans, r_mat
