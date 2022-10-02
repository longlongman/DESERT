from bycha.modules.search import register_search
from bycha.modules.search.greedy_search import GreedySearch
from bycha.modules.utils import create_init_scores
import torch

@register_search
class ShapePretrainingSearchIterativeNoRegression(GreedySearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self,
                prev_units,
                memory,
                memory_padding_mask,
                target_mask=None,
                prev_scores=None):
        batch_size = prev_units[0].size(0)
        scores = create_init_scores(prev_units[0], memory) if prev_scores is None else prev_scores
       
        # --------------------- normal inference --------------------
        if not prev_units[3]:
            for _ in range(int(memory.size(0) * self._maxlen_a + self._maxlen_b)):
                logits, trans, r_mat = self._decoder(prev_units[0], prev_units[1], prev_units[2], memory, memory_padding_mask)
                if isinstance(logits, list):
                    logits = logits[-1]
                    trans = trans[-1]
                    r_mat = r_mat[-1]
                logits = logits[:, -1, :]
                trans = trans[:, -1, :]
                r_mat = r_mat[:, -1, :]
                if target_mask is not None:
                    logits = logits.masked_fill(target_mask, float('-inf'))
                next_word_scores, words = logits.max(dim=-1)
                next_tran_scores, posit = trans.max(dim=-1)
                next_rota_scores, rotat = r_mat.max(dim=-1)
                eos_mask = words.eq(self._eos)
                if eos_mask.long().sum() == batch_size:
                    break
                scores = scores + next_word_scores.masked_fill_(eos_mask, 0.).view(-1) + \
                                  next_tran_scores.masked_fill_(eos_mask, 0.).view(-1) + \
                                  next_rota_scores.masked_fill_(eos_mask, 0.).view(-1)
                prev_tokens = torch.cat([prev_units[0], words.unsqueeze(dim=-1)], dim=-1)
                prev_trans = torch.cat([prev_units[1], posit.unsqueeze(dim=-1)], dim=-1)
                prev_r_mat = torch.cat([prev_units[2], rotat.unsqueeze(dim=-1)], dim=-1)
                prev_units = (prev_tokens, prev_trans, prev_r_mat)
            prev_units = (prev_units[0][:, 1:], prev_units[1][:, 1:], prev_units[2][:, 1:])
        # ------------------- teacher force inference ------------------
        else:
            output_tokens = []
            output_trans = []
            output_r_mat = []
            for i in range(prev_units[0].size(1)):
                logits, trans, r_mat = self._decoder(prev_units[0][:, :i + 1], prev_units[1][:, :i + 1], prev_units[2][:, :i + 1], memory, memory_padding_mask)
                if isinstance(logits, list):
                    logits = logits[-1]
                    trans = trans[-1]
                    r_mat = r_mat[-1]
                logits = logits[:, -1, :]
                trans = trans[:, -1, :]
                r_mat = r_mat[:, -1, :]
                if target_mask is not None:
                    logits = logits.masked_fill(target_mask, float('-inf'))
                next_word_scores, words = logits.max(dim=-1)
                next_tran_scores, posit = trans.max(dim=-1)
                next_rota_scores, rotat = r_mat.max(dim=-1)
                eos_mask = words.eq(self._eos)
                if eos_mask.long().sum() == batch_size:
                    break
                scores = scores + next_word_scores.masked_fill_(eos_mask, 0.).view(-1) + \
                                  next_tran_scores.masked_fill_(eos_mask, 0.).view(-1) + \
                                  next_rota_scores.masked_fill_(eos_mask, 0.).view(-1)
                output_tokens.append(words.unsqueeze(dim=-1))
                output_trans.append(posit.unsqueeze(dim=-1))
                output_r_mat.append(rotat.unsqueeze(dim=-1))
            output_tokens = torch.cat(output_tokens, dim=1)
            output_trans = torch.cat(output_trans, dim=1)
            output_r_mat = torch.cat(output_r_mat, dim=1)
            prev_units = (output_tokens, output_trans, output_r_mat)
        
        # ------------------- iterative inference ------------------
        output_tokens, output_trans, output_r_mat = self._decoder.iterative_infer(prev_units[0], prev_units[1], prev_units[2])
        output_tokens = output_tokens.argmax(-1)
        output_trans = output_trans.argmax(-1)
        output_r_mat = output_r_mat.argmax(-1)
        prev_units = (output_tokens, output_trans, output_r_mat)
        
        return scores, prev_units
