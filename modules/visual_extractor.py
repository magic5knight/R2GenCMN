import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        # Keep the ResNet stem before maxpool: for 224x224 input this is
        # [B, 64, 112, 112].
        modules = [model.conv1, model.bn1, model.relu]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).flatten(1)
        return patch_feats, avg_feats
