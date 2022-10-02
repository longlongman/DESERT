from bycha.modules.search import register_search
from bycha.modules.search.sequence_search import SequenceSearch
from bycha.modules.utils import create_init_scores
import torch
import torch.nn.functional as F
import torch.distributions as D
from .utils import get_dock_fast, get_dock_fast_with_smiles
import pickle

@register_search
class ShapePretrainingSearchForwardSamplingDockDedupIterativeNoRegression(SequenceSearch):
    def __init__(self, 
                 maxlen_coef=(1.2, 10), 
                 topk=1, 
                 ltopk=1, 
                 ttopk=1, 
                 rtopk=1,
                 ltopp=0.95,
                 ttopp=0.0,
                 rtopp=0.0,
                 ltemp=1.2,
                 ttemp=1.0,
                 rtemp=1.0,
                 num_return_sequence=2,
                 fnum_return_sequence=2,
                 keepdim=False,
                 for_protein_decode=False,
                 sampling_type='topp_independent'):
        super().__init__()

        self._maxlen_a, self._maxlen_b = maxlen_coef
        
        # topk sampling
        self._topk = topk
        self._ltopk = ltopk
        self._ttopk = ttopk
        self._rtopk = rtopk

        # topp sampling
        self._ttopp = ttopp
        self._ltopp = ltopp
        self._rtopp = rtopp
        
        self._num_return_sequence = num_return_sequence
        self._keepdim = keepdim
        self._sampling_type = sampling_type

        self._ltemp = ltemp
        self._ttemp = ttemp
        self._rtemp = rtemp

        self._fnum_return_sequence = fnum_return_sequence
        self._for_protein_decode = for_protein_decode
        if for_protein_decode:
            with open('/opt/tiger/shape_based_pretraining/data/vocab/vocab.h_nei_nof', 'rb') as fr:
                self._vocab_h_nei_nof = pickle.load(fr)
    
    def forward(self,
                units,
                memory,
                memory_padding_mask,
                target_mask=None,
                prev_scores=None):
        tokens = units[0]
        trans = units[1]
        rotat = units[2]
        if self._for_protein_decode:
            nof = units[4]

        bz, sl = tokens.size(0), tokens.size(1)
        tokens = tokens.unsqueeze(1).repeat(1, self._num_return_sequence, 1).view(bz * self._num_return_sequence, sl)
        trans = trans.unsqueeze(1).repeat(1, self._num_return_sequence, 1).view(bz * self._num_return_sequence, sl)
        rotat = rotat.unsqueeze(1).repeat(1, self._num_return_sequence, 1).view(bz * self._num_return_sequence, sl)

        # copy memory for 'num_return_sequence' times
        memory, memory_padding_mask = self._expand(memory, memory_padding_mask)

        scores = create_init_scores(tokens, memory) if prev_scores is None else prev_scores
        for _ in range(int(memory.size(0) * self._maxlen_a + self._maxlen_b)):
            logits, tlogits, rlogits = self._decoder(tokens, trans, rotat, memory, memory_padding_mask)
            
            if isinstance(logits, list):
                logits = logits[-1]
                tlogits = tlogits[-1]
                rlogits = rlogits[-1]

            logits = logits[:, -1, :]
            tlogits = tlogits[:, -1, :]
            rlogits = rlogits[:, -1, :]
            if target_mask is not None:
                logits = logits.masked_fill(target_mask, float('-inf'))
            
            logits = logits / self._ltemp
            tlogits = tlogits / self._ttemp
            rlogits = rlogits / self._rtemp
            
            if self._sampling_type == 'topk_joint':
                next_score, next_token, next_trans, next_rotat = self._sample_from_topk_joint(logits, tlogits, rlogits)
            elif self._sampling_type == 'topk_independent':
                next_score, next_token, next_trans, next_rotat = self._sample_from_topk_independent(logits, tlogits, rlogits)
            elif self._sampling_type == 'topp_independent':
                next_score, next_token, next_trans, next_rotat = self._sample_from_topp_independent(logits, tlogits, rlogits)
            elif self._sampling_type == 'topp_independent_for_protein_decode':
                assert self._for_protein_decode
                next_score, next_token, next_trans, next_rotat = self._sample_from_topp_independent_for_protein_decode(logits, tlogits, rlogits, nof)
            else:
                raise NotImplementedError
            
            eos_mask = next_token.eq(self._eos)
            scores = scores + next_score.masked_fill_(eos_mask, 0.).view(-1)
            tokens = torch.cat([tokens, next_token], dim=-1)
            trans = torch.cat([trans, next_trans], dim=-1)
            rotat = torch.cat([rotat, next_rotat], dim=-1)
        
        scores = scores.view(bz, self._num_return_sequence, -1)
        tokens = tokens.view(bz, self._num_return_sequence, -1)
        trans = trans.view(bz, self._num_return_sequence, -1)
        rotat = rotat.view(bz, self._num_return_sequence, -1)

        tokens, trans, rotat = self._get_top_dock_dedup(tokens, trans, rotat)

        if not self._keepdim and self._num_return_sequence == 1:
            tokens = tokens.squeeze(dim=1)
            trans = trans.squeeze(dim=1)
            rotat = rotat.squeeze(dim=1)

        return scores, (tokens, trans, rotat)
    
    def _get_top_dock_dedup(self, tokens, trans, rotat):
        curr_len = tokens.size(2)
        dock_results = get_dock_fast_with_smiles(tokens.cpu().tolist()[0],
                                                 trans.cpu().tolist()[0],
                                                 rotat.cpu().tolist()[0],
                                                 '--PDBQT PATH FOR DOCKING--',
                                                 '--LIGAND PATH FOR CALCULATE CENTER--')
        idxs = []
        smis = set()
        sorted_dock_results = sorted(dock_results, key=lambda x: x[0], reverse=True)
        for _, idx, smi in sorted_dock_results:
            if len(idxs) == self._fnum_return_sequence:
                break
            if smi in smis:
                continue
            if smi == '':
                continue
            idxs.append(idx)
            smis.add(smi)
        idxs = torch.tensor(idxs).to(tokens.device).unsqueeze(0)
        top_tokens = tokens.gather(1, index=idxs.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        top_trans = trans.gather(1, index=idxs.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        top_rotat = rotat.gather(1, index=idxs.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        return top_tokens, top_trans, top_rotat
    
    def _get_top_dock_fast(self, tokens, trans, rotat):
        mol_num = tokens.size(1)
        curr_len = tokens.size(2)
        mol_dock = [float('-inf') for _ in range(mol_num)]
        dock_results = get_dock_fast(tokens.cpu().tolist()[0],
                                     trans.cpu().tolist()[0],
                                     rotat.cpu().tolist()[0],
                                     '--PDBQT PATH FOR DOCKING--',
                                     '--LIGAND PATH FOR CALCULATE CENTER--')
        for ds, idx in dock_results:
            mol_dock[idx] = ds
        mol_dock = torch.tensor(mol_dock).to(tokens.device).unsqueeze(0)
        _, idx = mol_dock.topk(self._fnum_return_sequence, dim=1)
        top_tokens = tokens.gather(1, index=idx.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        top_trans = trans.gather(1, index=idx.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        top_rotat = rotat.gather(1, index=idx.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        return top_tokens, top_trans, top_rotat
    
    def _sample_from_topp_independent_for_protein_decode(self, logits, trans, r_mat, nof):
        position = trans.argmax(dim=-1) - 2
        flat_nof = nof.view(-1)
        nof_num = flat_nof[position]
        vocab_h_nei_nof = torch.tensor(self._vocab_h_nei_nof, dtype=torch.float).to(nof_num.device)
        match_score = nof_num.unsqueeze(-1) * vocab_h_nei_nof.unsqueeze(0)
        match_max = match_score.max(dim=-1)[0]
        match_min = match_score.min(dim=-1)[0]
        alpha = (match_score - match_min.unsqueeze(-1)) / (match_max - match_min + 1e-9).unsqueeze(-1)
        logits = alpha * logits + logits
        return self._sample_from_topp_independent(logits, trans, r_mat)

    def _sample_from_topp_independent(self, logits, trans, r_mat):
        logits = F.softmax(logits, dim=-1)
        t_logits = F.softmax(trans, dim=-1)
        r_logits = F.softmax(r_mat, dim=-1)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_t_logits, sorted_t_indices = torch.sort(t_logits, descending=True)
        sorted_r_logits, sorted_r_indices = torch.sort(r_logits, descending=True)

        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
        cumulative_t_probs = torch.cumsum(sorted_t_logits, dim=-1)
        cumulative_r_probs = torch.cumsum(sorted_r_logits, dim=-1)

        sorted_indices_to_remove = cumulative_probs > self._ltopp
        sorted_t_indices_to_remove = cumulative_t_probs > self._ttopp
        sorted_r_indices_to_remove = cumulative_r_probs > self._rtopp

        # make sure at least have one point to sample
        sorted_indices_to_remove[..., 0] = 0
        sorted_t_indices_to_remove[..., 0] = 0
        sorted_r_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        t_indices_to_remove = sorted_t_indices_to_remove.scatter(1, sorted_t_indices, sorted_t_indices_to_remove)
        r_indices_to_remove = sorted_r_indices_to_remove.scatter(1, sorted_r_indices, sorted_r_indices_to_remove)
        
        logits[indices_to_remove] = 0.0
        t_logits[t_indices_to_remove] = 0.0
        r_logits[r_indices_to_remove] = 0.0
        
        token_prob = logits / logits.sum(dim=-1, keepdim=True)
        trans_prob = t_logits / t_logits.sum(dim=-1, keepdim=True)
        rotat_prob = r_logits / r_logits.sum(dim=-1, keepdim=True)

        token_dist = D.Categorical(token_prob)
        trans_dist = D.Categorical(trans_prob)
        rotat_dist = D.Categorical(rotat_prob)

        next_token = token_dist.sample((1, )).permute(1, 0)
        next_trans = trans_dist.sample((1, )).permute(1, 0)
        next_rotat = rotat_dist.sample((1, )).permute(1, 0)

        next_token_score = logits.gather(-1, next_token)
        next_trans_score = t_logits.gather(-1, next_trans)
        next_rotat_score = r_logits.gather(-1, next_rotat)

        next_score = next_token_score * next_trans_score * next_rotat_score

        return next_score, next_token, next_trans, next_rotat

    def _sample_from_topk_independent(self, logits, trans, r_mat):
        logits = F.softmax(logits, dim=-1)
        t_logits = F.softmax(trans, dim=-1)
        r_logits = F.softmax(r_mat, dim=-1)

        topk_token_scores, topk_token = logits.topk(self._ltopk, dim=-1)
        topk_trans_scores, topk_trans = t_logits.topk(self._ttopk, dim=-1)
        topk_rotat_scores, topk_rotat = r_logits.topk(self._rtopk, dim=-1)

        token_prob = topk_token_scores / topk_token_scores.sum(dim=-1, keepdim=True)
        trans_prob = topk_trans_scores / topk_trans_scores.sum(dim=-1, keepdim=True)
        rotat_prob = topk_rotat_scores / topk_rotat_scores.sum(dim=-1, keepdim=True)

        token_dist = D.Categorical(token_prob)
        trans_dist = D.Categorical(trans_prob)
        rotat_dist = D.Categorical(rotat_prob)

        next_token_index = token_dist.sample((1, )).permute(1, 0)
        next_trans_index = trans_dist.sample((1, )).permute(1, 0)
        next_rotat_index = rotat_dist.sample((1, )).permute(1, 0)

        next_token = topk_token.gather(-1, next_token_index)
        next_trans = topk_trans.gather(-1, next_trans_index)
        next_rotat = topk_rotat.gather(-1, next_rotat_index)

        next_token_score = topk_token_scores.gather(-1, next_token_index)
        next_trans_score = topk_trans_scores.gather(-1, next_trans_index)
        next_rotat_score = topk_rotat_scores.gather(-1, next_rotat_index)

        next_score = next_token_score * next_trans_score * next_rotat_score

        return next_score, next_token, next_trans, next_rotat

    
    def _sample_from_topk_joint(self, logits, trans, r_mat):
        batch_size = logits.size(0)
        
        logits = F.softmax(logits, dim=-1)
        t_logits = F.softmax(trans, dim=-1)
        r_logits = F.softmax(r_mat, dim=-1)

        topk_token_scores, topk_token = logits.topk(self._ltopk, dim=-1)
        topk_trans_scores, topk_trans = t_logits.topk(self._ttopk, dim=-1)
        topk_rotat_scores, topk_rotat = r_logits.topk(self._rtopk, dim=-1)

        token_trans_scores = topk_token_scores.view(batch_size, self._ltopk, 1) * \
                             topk_trans_scores.view(batch_size, 1, self._ttopk)
        token_trans_rotat_scores = token_trans_scores.view(batch_size, self._ltopk * self._ttopk, 1) * \
                                   topk_rotat_scores.view(batch_size, 1, self._rtopk)
        
        next_token_trans_rotat_scores, next_token_trans_rotat = token_trans_rotat_scores.view(batch_size, -1).topk(self._topk, dim=-1)
        
        # prob = F.softmax(next_token_trans_rotat_scores, dim=-1)
        prob = next_token_trans_rotat_scores / next_token_trans_rotat_scores.sum(dim=-1, keepdim=True)
        dist = D.Categorical(prob)
        next_token_trans_rotat = dist.sample((1, ))
        next_token_trans_rotat = next_token_trans_rotat.permute(1, 0)

        next_rotat_index = next_token_trans_rotat % self._rtopk
        next_trans_index = ((next_token_trans_rotat - next_rotat_index) % (self._ttopk * self._rtopk)) // self._rtopk
        next_token_index = (next_token_trans_rotat - next_rotat_index - next_trans_index * self._rtopk) // (self._ttopk * self._rtopk)

        next_token = topk_token.gather(-1, next_token_index)
        next_trans = topk_trans.gather(-1, next_trans_index)
        next_rotat = topk_rotat.gather(-1, next_rotat_index)

        next_score = next_token_trans_rotat_scores.gather(-1, next_token_trans_rotat)

        return next_score, next_token, next_trans, next_rotat
    
    def _expand(self, memory, memory_padding_mask):
        batch_size, memory_size = memory_padding_mask.size()
        memory = memory.unsqueeze(dim=2).repeat(1, 1, self._num_return_sequence, 1)
        memory = memory.view(memory_size, batch_size * self._num_return_sequence, -1)
        memory_padding_mask = memory_padding_mask.unsqueeze(dim=1).repeat(1, self._num_return_sequence, 1)
        memory_padding_mask = memory_padding_mask.view(batch_size * self._num_return_sequence, memory_size)
        return memory, memory_padding_mask
