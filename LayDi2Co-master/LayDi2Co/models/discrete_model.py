import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from models.utils import PositionalEncoding, TimestepEmbedder

class DiscreteModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, categories_num, latent_dim=256, num_layers=4, num_heads=4, dropout_r=0., activation="gelu",
                 cond_emb_size=224,cat_emb_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.categories_num = categories_num
        self.cat_emb = nn.Parameter(torch.randn(self.categories_num, cat_emb_size))
        self.cond_mask_cat_emb = nn.Parameter(torch.randn(2, cat_emb_size))
        self.seq_pos_enc = PositionalEncoding(self.latent_dim, self.dropout_r)


        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.seq_pos_enc)
        self.cond_emb_size = cond_emb_size

        self.token_proj = nn.Linear(cat_emb_size, latent_dim)


        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=self.latent_dim * 2,
            dropout=dropout_r,
            activation=activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)


        self.output_cls = nn.Sequential(
            nn.Linear(self.latent_dim, categories_num))

    def forward(self, sample, noisy_sample, timesteps):

        cat_input = noisy_sample['cat'] * sample['mask_cat'] + (1 - sample['mask_cat']) * sample['cat']
        cat_input_flat = rearrange(cat_input, 'b c -> (b c)')


        elem_cat_emb = self.cat_emb[cat_input_flat, :]
        elem_cat_emb = rearrange(elem_cat_emb, '(b c) d -> b c d', b=cat_input.shape[0])


        mask_cl = sample['mask_cat']
        mask_flat = rearrange(mask_cl, 'b c -> (b c)').long()
        emb_mask_cl = self.cond_mask_cat_emb[mask_flat, :]

        emb_mask_cl = rearrange(emb_mask_cl, '(b c) d -> b c d', b=sample['mask_cat'].shape[0])

        elem_cat_emb = elem_cat_emb + emb_mask_cl
        tokens_emb = self.token_proj(elem_cat_emb)
        tokens_emb = rearrange(tokens_emb, 'b c d -> c b d')

        t_emb = self.embed_timestep(timesteps)


        xseq = torch.cat([t_emb, tokens_emb], dim=0)


        xseq = self.seq_pos_enc(xseq)
        xseq = self.seqTransEncoder(xseq)
        output = xseq[1:]
        output = rearrange(output, 'c b d -> b c d')


        output_cls = self.output_cls(output)
        return output_cls




