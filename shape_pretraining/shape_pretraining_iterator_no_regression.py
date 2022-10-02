from bycha.modules.encoders import register_encoder
from bycha.modules.encoders.transformer_encoder import TransformerEncoder
from bycha.modules.layers.feed_forward import FFN
import torch
import torch.nn as nn

@register_encoder
class ShapePretrainingIteratorNoRegression(TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build(self,
              embed, 
              special_tokens,
              trans_size,
              rotat_size):
        super().build(embed, special_tokens)
        from bycha.modules.layers.embedding import Embedding

        self._trans_emb = Embedding(vocab_size=trans_size,
                                    d_model=embed.weight.shape[1])
        self._rotat_emb = Embedding(vocab_size=rotat_size,
                                    d_model=embed.weight.shape[1])
        
        self._logits_output_proj = nn.Linear(embed.weight.shape[1],
                                             embed.weight.shape[0],
                                             bias=False)
        self._logits_output_proj.weight = embed.weight
        self._trans_output_proj = nn.Linear(self._trans_emb.weight.shape[1],
                                            self._trans_emb.weight.shape[0],
                                            bias=False)
        self._trans_output_proj.weight = self._trans_emb.weight
        self._rotat_output_proj = nn.Linear(self._rotat_emb.weight.shape[1],
                                            self._rotat_emb.weight.shape[0],
                                            bias=False)
        self._rotat_output_proj.weight = self._rotat_emb.weight
    
    def _forward(self, logits, trans, r_mat, padding_mask):
        bz, sl = logits.size(0), logits.size(1)
        logits_pred = logits.argmax(-1)
        trans_pred = trans.argmax(-1)
        r_mat_pred = r_mat.argmax(-1)
        
        x = self._embed(logits_pred)
        x = x + self._trans_emb(trans_pred)
        x = x + self._rotat_emb(r_mat_pred)

        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            pos = torch.arange(sl).unsqueeze(0).repeat(bz, 1).to(x.device)
            x = x + self._pos_embed(pos)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        src_padding_mask = padding_mask
        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x, src_key_padding_mask=src_padding_mask)
        
        if self._norm is not None:
            x = self._norm(x)
        
        x = x.transpose(0, 1)
        logits = self._logits_output_proj(x)
        trans = self._trans_output_proj(x)
        r_mat = self._rotat_output_proj(x)

        return logits, trans, r_mat
