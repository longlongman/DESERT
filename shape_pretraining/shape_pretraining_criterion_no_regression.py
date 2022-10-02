from bycha.criteria import register_criterion
from bycha.criteria.cross_entropy import CrossEntropy
from torch.nn import MSELoss
import torch
import torch.nn.functional as F
import torch.nn as nn

@register_criterion
class ShapePretrainingCriterionNoRegression(CrossEntropy):
    def __init__(self, weight=None, logging_metric='acc', trans=1.0, rotat=1.0):
        super().__init__(weight=weight, logging_metric=logging_metric)
        self._nll = nn.NLLLoss(ignore_index=0)
        self._trans = trans
        self._rotat = rotat
    
    def compute_loss(self,
                     lprobs, 
                     output_frag_idx,
                     output_frag_idx_mask,
                     output_frag_trans,
                     output_frag_trans_mask,
                     output_frag_r_mat,
                     output_frag_r_mat_mask):
        predict_frag_idx = lprobs[0]
        predict_frag_trans = lprobs[1]
        predict_frag_r_mat = lprobs[2]
        
        if isinstance(predict_frag_idx, list):
            tmp_nll_loss, tmp_acc = [], []
            for i in range(len(predict_frag_idx)):
                curr_nll_loss, curr_logging_states = super().compute_loss(predict_frag_idx[i], output_frag_idx)
                tmp_nll_loss.append(curr_nll_loss)
                tmp_acc.append(curr_logging_states['acc'])
            nll_loss = sum(tmp_nll_loss) / len(tmp_nll_loss)
            logging_states = {
                'loss': nll_loss.data.item(),
                'acc': tmp_acc[-1] # use the prediction at last layer as the final prediction
            }
        else:
            nll_loss, logging_states = super().compute_loss(predict_frag_idx, output_frag_idx)
        
        if isinstance(predict_frag_trans, list):
            tmp_trans_nll_loss, tmp_trans_lprobs = [], []
            trans_target = output_frag_trans.view(-1)
            for i in range(len(predict_frag_trans)):
                curr_trans_lprobs = F.log_softmax(predict_frag_trans[i], dim=-1)
                curr_trans_lprobs = curr_trans_lprobs.view(-1, curr_trans_lprobs.size(-1))
                
                curr_trans_nll_loss = self._nll(curr_trans_lprobs, trans_target)
                curr_trans_nll_loss = self._trans * curr_trans_nll_loss

                tmp_trans_lprobs.append(curr_trans_lprobs)
                tmp_trans_nll_loss.append(curr_trans_nll_loss)
            trans_nll_loss = sum(tmp_trans_nll_loss) / len(tmp_trans_nll_loss)
            trans_lprobs = tmp_trans_lprobs[-1] # use the prediction at last layer as the final prediction
        else:
            trans_lprobs = F.log_softmax(predict_frag_trans, dim=-1)
            trans_lprobs = trans_lprobs.view(-1, trans_lprobs.size(-1))
            trans_target = output_frag_trans.view(-1)
            trans_nll_loss = self._nll(trans_lprobs, trans_target)
            trans_nll_loss = self._trans * trans_nll_loss

        if isinstance(predict_frag_r_mat, list):
            tmp_rotat_nll_loss, tmp_rotat_lprobs = [], []
            rotat_target = output_frag_r_mat.view(-1)
            for i in range(len(predict_frag_r_mat)):
                curr_rotat_lprobs = F.log_softmax(predict_frag_r_mat[i], dim=-1)
                curr_rotat_lprobs = curr_rotat_lprobs.view(-1, curr_rotat_lprobs.size(-1))

                curr_rotat_nll_loss = self._nll(curr_rotat_lprobs, rotat_target)
                curr_rotat_nll_loss = self._rotat * curr_rotat_nll_loss

                tmp_rotat_nll_loss.append(curr_rotat_nll_loss)
                tmp_rotat_lprobs.append(curr_rotat_lprobs)
            rotat_nll_loss = sum(tmp_rotat_nll_loss) / len(tmp_rotat_nll_loss)
            rotat_lprobs = tmp_rotat_lprobs[-1] # use the prediction at last layer as the final prediction
        else:
            rotat_lprobs = F.log_softmax(predict_frag_r_mat, dim=-1)
            rotat_lprobs = rotat_lprobs.view(-1, rotat_lprobs.size(-1))
            rotat_target = output_frag_r_mat.view(-1)
            rotat_nll_loss = self._nll(rotat_lprobs, rotat_target)
            rotat_nll_loss = self._rotat * rotat_nll_loss

        total_loss = nll_loss + trans_nll_loss + rotat_nll_loss

        logging_states['nll_trans'] = trans_nll_loss.item()
        logging_states['nll_rotat'] = rotat_nll_loss.item()

        # ------------------- fix the wrong acc here ---------------------
        if self._logging_metric == 'acc':
            if isinstance(predict_frag_idx, list):
                lprobs = F.log_softmax(predict_frag_idx[-1], dim=-1)
            else:
                lprobs = F.log_softmax(predict_frag_idx, dim=-1)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = output_frag_idx.view(-1)
            correct = ((lprobs.max(dim=-1)[1] == target) * output_frag_idx_mask.view(-1)).sum().data.item()
            tot = output_frag_idx_mask.sum().data.item()
            logging_states['acc'] = correct / tot

            correct = ((trans_lprobs.max(dim=-1)[1] == trans_target) * output_frag_trans_mask.view(-1)).sum().data.item()
            tot = output_frag_trans_mask.sum().data.item()
            logging_states['acc_trans'] = correct / tot

            correct = ((rotat_lprobs.max(dim=-1)[1] == rotat_target) * output_frag_r_mat_mask.view(-1)).sum().data.item()
            tot = output_frag_r_mat_mask.sum().data.item()
            logging_states['acc_rotat'] = correct / tot
        
        return total_loss, logging_states
