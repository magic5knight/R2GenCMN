import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def count_parameters(self):
        """Count total / trainable parameters of each component (shared tensors counted once)."""
        def count(params):
            params = list({id(p): p for p in params}.values())
            total = sum(p.numel() for p in params)
            trainable = sum(p.numel() for p in params if p.requires_grad)
            return total, trainable

        ed = self.encoder_decoder
        encoder_params = list(ed.att_embed.parameters()) + list(ed.model.encoder.parameters())
        encoder_ids = set(map(id, encoder_params))
        cmn_decoder_params = [p for p in ed.parameters() if id(p) not in encoder_ids]

        stats = {
            'ResNet (visual_extractor)': count(self.visual_extractor.parameters()),
            'Transformer encoder (+att_embed)': count(encoder_params),
            'CMN + Decoder': count(cmn_decoder_params),
        }
        stats['TOTAL'] = count(self.parameters())
        return stats

    def log_parameters(self, logger):
        stats = self.count_parameters()
        logger.info('=' * 78)
        logger.info('{:35s} {:>14s} {:>14s} {:>10s}'.format('Component', 'Total', 'Trainable', '% model'))
        logger.info('-' * 78)
        total_trainable = stats['TOTAL'][1]
        for name, (total, trainable) in stats.items():
            if name == 'TOTAL':
                logger.info('-' * 78)
            logger.info('{:35s} {:>14,} {:>14,} {:>9.2f}%'.format(
                name, total, trainable, 100.0 * trainable / max(total_trainable, 1)))
        logger.info('=' * 78)
        return stats

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
