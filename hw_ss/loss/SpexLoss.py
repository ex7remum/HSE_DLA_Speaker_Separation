import torch
from torch import Tensor
import torch.nn as nn


class SpexPlusLoss(nn.Module):
    def __init__(self):
        super(SpexPlusLoss, self).__init__()

    def sisdr(self, x, s, eps=1e-8):

        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))

        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def mask_by_length(self, xs, lengths, fill=0):
        assert xs.size(0) == len(lengths)
        ret = xs.data.new(*xs.size()).fill_(fill)
        for i, l in enumerate(lengths):
            ret[i, :l] = xs[i, :l]
        return ret

    def forward(self, pred_values, batch):
        logits = pred_values['logits']
        s1 = pred_values['s1'].squeeze(1)
        s2 = pred_values['s2'].squeeze(1)
        s3 = pred_values['s3'].squeeze(1)
        valid_len = batch['len_tgt']
        tgt_id = batch['target_id']
        ests = self.mask_by_length(s1, valid_len)
        ests2 = self.mask_by_length(s2, valid_len)
        ests3 = self.mask_by_length(s3, valid_len)
        tgt = self.mask_by_length(batch['audio_tgt'].squeeze(1), valid_len)
        snr1 = self.sisdr(ests, tgt)
        snr2 = self.sisdr(ests2, tgt)
        snr3 = self.sisdr(ests3, tgt)
        snr_loss = (-0.8 * torch.sum(snr1) - 0.1 * torch.sum(snr2) - 0.1 * torch.sum(snr3)) / logits.shape[0]
        ce = torch.nn.CrossEntropyLoss()
        ce_loss = ce(logits, tgt_id)
        pred = logits.argmax(dim=-1)
        correct = (pred == tgt_id).sum().item() / logits.shape[0]

        return {
            'loss': snr_loss + 0.5 * ce_loss,
            'ce_loss': ce_loss,
            'si_sdr': snr_loss,
            'correct_speakers': correct
        }

