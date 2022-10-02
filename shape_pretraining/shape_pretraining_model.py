from bycha.models import register_model
from bycha.models.encoder_decoder_model import EncoderDecoderModel
import torch

@register_model
class ShapePretrainingModel(EncoderDecoderModel):
    def __init__(self,
                 encoder,
                 decoder,
                 d_model,
                 share_embedding=None,
                 path=None,
                 no_shape=False,
                 no_trans=False,
                 no_rotat=False):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         d_model=d_model,
                         share_embedding=share_embedding,
                         path=path)
        self._no_shape = no_shape
        self._no_trans = no_trans
        self._no_rotat = no_rotat
    
    def forward(self, 
                shape,
                shape_patches, 
                input_frag_idx,
                input_frag_idx_mask,
                input_frag_trans,
                input_frag_trans_mask,
                input_frag_r_mat,
                input_frag_r_mat_mask):
        memory, memory_padding_mask = self._encoder(src=shape_patches)
        if self._no_shape:
            memory = torch.zeros_like(memory)
        if self._no_trans:
            input_frag_trans = torch.zeros_like(input_frag_trans)
        if self._no_rotat:
            input_frag_r_mat = torch.zeros_like(input_frag_r_mat)
        logits, trans, r_mat = self._decoder(input_frag_idx=input_frag_idx,
                                             input_frag_trans=input_frag_trans,
                                             input_frag_r_mat=input_frag_r_mat,
                                             memory=memory,
                                             memory_padding_mask=memory_padding_mask)
        return (logits, trans, r_mat)
