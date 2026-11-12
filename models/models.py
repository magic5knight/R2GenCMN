import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor
from modules.vmamba import Backbone_VSSM


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        # ResNet stem output [B,64,112,112] -> VSSM stage 2 [B,384,7,7].
        self.visual_backbone = Backbone_VSSM(
            in_chans=64,
            patch_size=4,
            out_indices=(2,),
            forward_type="v04",
        )
        self.visual_ffn = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, args.d_model),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model, args.d_model),
        )
        self.encoder_decoder = BaseCMN(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def _encode_image(self, images):
        feature_map, pooled = self.visual_extractor(images)
        feature_map = self.visual_backbone(feature_map)[0]  # [B,384,7,7]
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        tokens = self.visual_ffn(tokens)  # [B,49,d_model]
        return tokens, pooled

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        att_feats_0, fc_feats_0 = self._encode_image(images[:, 0])
        att_feats_1, fc_feats_1 = self._encode_image(images[:, 1])
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
        att_feats, fc_feats = self._encode_image(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
