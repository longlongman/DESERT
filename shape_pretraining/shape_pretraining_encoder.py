from bycha.modules.encoders import register_encoder
from bycha.modules.encoders.transformer_encoder import TransformerEncoder
from bycha.modules.layers.feed_forward import FFN
import torch

@register_encoder
class ShapePretrainingEncoder(TransformerEncoder):
    def __init__(self, patch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patch_size = patch_size
    
    def build(self,
              embed, 
              special_tokens):
        super().build(embed, special_tokens)
        self._patch_ffn = FFN(self._patch_size**3, self._d_model, self._d_model)
    
    def _forward(self, src):
        bz, sl = src.size(0), src.size(1)
        
        x = self._patch_ffn(src)
        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            pos = torch.arange(sl).unsqueeze(0).repeat(bz, 1).to(x.device)
            x = x + self._pos_embed(pos)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        src_padding_mask = torch.zeros((bz, sl), dtype=torch.bool).to(x.device)
        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x, src_key_padding_mask=src_padding_mask)
        
        if self._norm is not None:
            x = self._norm(x)
        
        if self._return_seed:
            encoder_out = x[1:], src_padding_mask[:, 1:], x[0]
        else:
            encoder_out = x, src_padding_mask

        return encoder_out
