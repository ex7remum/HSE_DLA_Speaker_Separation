import torch
import torch.nn as nn
import torch.nn.functional as F
from hw_ss.model.layers import TCNBlock, TCNBlock_Spk, ResBlock

# modified version of https://github.com/xuchenglin28/speaker_extraction_SpEx
class SpexPlus(nn.Module):
    def __init__(self,
                 short_kernel_size=20,
                 middle_kernel_size=80,
                 long_kernel_size=160,
                 num_feats=256,
                 num_blocks=8,
                 n_proj_channels=256,
                 hidden_dim=512,
                 tcn_kernel_size=3,
                 num_spks=101,
                 spk_embed_dim=256):
        super(SpexPlus, self).__init__()
        self.short_kernel_size = short_kernel_size
        self.middle_kernel_size = middle_kernel_size
        self.long_kernel_size = long_kernel_size

        self.encoder_short = nn.Conv1d(1, num_feats, short_kernel_size, stride=short_kernel_size // 2)
        self.encoder_middle = nn.Conv1d(1, num_feats, middle_kernel_size, stride=short_kernel_size // 2)
        self.encoder_long = nn.Conv1d(1, num_feats, long_kernel_size, stride=short_kernel_size // 2)
        self.layer_norm = nn.LayerNorm(3 * num_feats)

        self.proj = nn.Conv1d(3 * num_feats, n_proj_channels, 1)
        self.conv_block_1 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim,
                                         in_channels=n_proj_channels,
                                         conv_channels=hidden_dim,
                                         kernel_size=tcn_kernel_size)

        self.conv_block_1_other = self._build_stacks(num_blocks=num_blocks,
                                                     in_channels=n_proj_channels,
                                                     conv_channels=hidden_dim,
                                                     kernel_size=tcn_kernel_size)

        self.conv_block_2 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim,
                                         in_channels=n_proj_channels,
                                         conv_channels=hidden_dim,
                                         kernel_size=tcn_kernel_size)

        self.conv_block_2_other = self._build_stacks(num_blocks=num_blocks,
                                                     in_channels=n_proj_channels,
                                                     conv_channels=hidden_dim,
                                                     kernel_size=tcn_kernel_size)

        self.conv_block_3 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim,
                                         in_channels=n_proj_channels,
                                         conv_channels=hidden_dim,
                                         kernel_size=tcn_kernel_size)

        self.conv_block_3_other = self._build_stacks(num_blocks=num_blocks,
                                                     in_channels=n_proj_channels,
                                                     conv_channels=hidden_dim,
                                                     kernel_size=tcn_kernel_size)

        self.conv_block_4 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim,
                                         in_channels=n_proj_channels,
                                         conv_channels=hidden_dim,
                                         kernel_size=tcn_kernel_size)

        self.conv_block_4_other = self._build_stacks(num_blocks=num_blocks,
                                                     in_channels=n_proj_channels,
                                                     conv_channels=hidden_dim,
                                                     kernel_size=tcn_kernel_size)

        self.mask1 = nn.Conv1d(n_proj_channels, num_feats, 1)
        self.mask2 = nn.Conv1d(n_proj_channels, num_feats, 1)
        self.mask3 = nn.Conv1d(n_proj_channels, num_feats, 1)

        self.decoder_short = nn.ConvTranspose1d(num_feats, 1, kernel_size=short_kernel_size,
                                                stride=short_kernel_size // 2)
        self.decoder_middle = nn.ConvTranspose1d(num_feats, 1, kernel_size=middle_kernel_size,
                                                 stride=short_kernel_size // 2)
        self.decoder_long = nn.ConvTranspose1d(num_feats, 1, kernel_size=long_kernel_size,
                                               stride=short_kernel_size // 2)
        self.num_spks = num_spks

        self.layer_norm_spk = nn.LayerNorm(3 * num_feats)
        self.spk_encoder = nn.Sequential(
            nn.Conv1d(3 * num_feats, n_proj_channels, 1),
            ResBlock(n_proj_channels, n_proj_channels),
            ResBlock(n_proj_channels, hidden_dim),
            ResBlock(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, spk_embed_dim, 1),
        )

        self.class_head = nn.Linear(spk_embed_dim, num_spks)

    def _build_stacks(self, num_blocks, **block_kwargs):
        blocks = [
            TCNBlock(**block_kwargs, dilation=(2 ** b))
            for b in range(1, num_blocks)
        ]
        return nn.Sequential(*blocks)

    def forward(self, audio_mix, audio_ref, len_ref, *args, **kwargs):
        mix = audio_mix
        ref = audio_ref
        ref_len = len_ref
        w1 = F.relu(self.encoder_short(mix))
        T = w1.shape[-1]
        xlen1 = mix.shape[-1]
        xlen2 = (T - 1) * (self.short_kernel_size // 2) + self.middle_kernel_size
        xlen3 = (T - 1) * (self.short_kernel_size // 2) + self.long_kernel_size
        w2 = F.relu(self.encoder_middle(F.pad(mix, (0, xlen2 - xlen1), "constant", 0)))
        w3 = F.relu(self.encoder_long(F.pad(mix, (0, xlen3 - xlen1), "constant", 0)))

        y = torch.cat([w1, w2, w3], 1)
        y = torch.transpose(y, 1, 2)
        y = self.layer_norm(y)
        y = torch.transpose(y, 1, 2)
        y = self.proj(y)

        ref_w1 = F.relu(self.encoder_short(ref))
        ref_T_shape = ref_w1.shape[-1]
        ref_len1 = ref.shape[-1]
        ref_len2 = (ref_T_shape - 1) * (self.short_kernel_size // 2) + self.middle_kernel_size
        ref_len3 = (ref_T_shape - 1) * (self.short_kernel_size // 2) + self.long_kernel_size
        ref_w2 = F.relu(self.encoder_middle(F.pad(ref, (0, ref_len2 - ref_len1), "constant", 0)))
        ref_w3 = F.relu(self.encoder_long(F.pad(ref, (0, ref_len3 - ref_len1), "constant", 0)))

        ref = torch.cat([ref_w1, ref_w2, ref_w3], 1)
        ref = torch.transpose(ref, 1, 2)
        ref = self.layer_norm_spk(ref)
        ref = torch.transpose(ref, 1, 2)
        ref = self.spk_encoder(ref)
        ref_T = (ref_len - self.short_kernel_size) // (self.short_kernel_size // 2) + 1
        ref_T = ((ref_T // 3) // 3) // 3
        ref = torch.sum(ref, -1) / ref_T.view(-1, 1).float().to(ref.device)

        y = self.conv_block_1(y, ref)
        y = self.conv_block_1_other(y)
        y = self.conv_block_2(y, ref)
        y = self.conv_block_2_other(y)
        y = self.conv_block_3(y, ref)
        y = self.conv_block_3_other(y)
        y = self.conv_block_4(y, ref)
        y = self.conv_block_4_other(y)

        m1 = F.relu(self.mask1(y))
        m2 = F.relu(self.mask2(y))
        m3 = F.relu(self.mask3(y))
        S1 = w1 * m1
        S2 = w2 * m2
        S3 = w3 * m3

        short_res = self.decoder_short(S1).squeeze(1)
        short_res = F.pad(short_res, (0, T - short_res.shape[-1]), "constant", 0)
        middle_res = self.decoder_middle(S2).squeeze(1)[:, :xlen1]
        middle_res = F.pad(middle_res, (0, T - middle_res.shape[-1]), "constant", 0)
        long_res = self.decoder_long(S3).squeeze(1)[:, :xlen1]
        long_res = F.pad(long_res, (0, T - long_res.shape[-1]), "constant", 0)
        logits_spkrs = self.class_head(ref)
        return {
            's1': short_res,
            's2': middle_res,
            's3': long_res,
            'logits': logits_spkrs
        }
