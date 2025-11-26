import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from models.utils import PositionalEncoding, TimestepEmbedder

class ContinuousModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, categories_num, latent_dim=256, num_layers=4, num_heads=4, dropout_r=0., activation="gelu",
                 cond_emb_size=224,cat_emb_size=64):
        super().__init__()
        self.categories_num = categories_num
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.cat_emb_size = cat_emb_size
        self.seq_pos_enc = PositionalEncoding(self.latent_dim, self.dropout_r)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.seq_pos_enc)

        # transformer
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=self.latent_dim * 2,
            dropout=dropout_r,
            activation=activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)
        self.size_emb = nn.Linear(2, cond_emb_size)
        self.loc_emb = nn.Linear(2, cond_emb_size)
        self.cond_mask_box_emb = nn.Parameter(torch.randn(2, cond_emb_size))
        self.output_box = nn.Linear(self.latent_dim, 4)
        self.cat_emb = nn.Parameter(torch.randn(self.categories_num, cat_emb_size))


        self.null_label_emb = nn.Parameter(torch.zeros(cat_emb_size))


    def forward(self, sample, noisy_sample, timesteps, cond_label_emb):
        sample_tensor = sample['mask_box'] * noisy_sample + (1 - sample['mask_box']) * sample['box_cond']

        xy = sample_tensor[:, :, :2]
        wh = sample_tensor[:, :, 2:]

        cond_label_flat = rearrange(cond_label_emb, 'b c -> (b c)')

        elem_cat_emb_flat = torch.zeros((cond_label_flat.shape[0], self.cat_emb_size), device=cond_label_flat.device)

        cond_mask = cond_label_flat != -1
        elem_cat_emb_flat[cond_mask] = self.cat_emb[cond_label_flat[cond_mask]]

        elem_cat_emb_flat[~cond_mask] = self.null_label_emb

        elem_cat_emb = rearrange(elem_cat_emb_flat, '(b c) d -> b c d', b=cond_label_emb.shape[0])

        mask_wh = sample['mask_box'][:, :, 2]
        mask_xy = sample['mask_box'][:, :, 0]

        def mask_to_emb(mask, cond_mask_emb):
            mask_flat = rearrange(mask, 'b c -> (b c)').long()
            mask_all_emb = cond_mask_emb[mask_flat, :]
            mask_all_emb = rearrange(mask_all_emb, '(b c) d -> b c d', b=mask.shape[0])
            return mask_all_emb

        emb_mask_wh = mask_to_emb(mask_wh, self.cond_mask_box_emb)
        emb_mask_xy = mask_to_emb(mask_xy, self.cond_mask_box_emb)

        size_emb = self.size_emb(wh) + emb_mask_wh
        loc_emb = self.loc_emb(xy) + emb_mask_xy


        t_emb = self.embed_timestep(timesteps)
        tokens_emb = torch.cat([size_emb, loc_emb, elem_cat_emb], dim=-1)
        tokens_emb = rearrange(tokens_emb, 'b c d -> c b d')
        xseq = torch.cat((t_emb, tokens_emb), dim=0)
        xseq = self.seq_pos_enc(xseq)


        output = self.seqTransEncoder(xseq)[1:]
        output = rearrange(output, 'c b d -> b c d')


        output_box = self.output_box(output)
        return output_box
