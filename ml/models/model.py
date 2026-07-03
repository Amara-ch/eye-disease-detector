import torch
import torch.nn as nn
import timm


class DualHeadNet(nn.Module):
    '''
    One backbone, two heads:
      - binary head: disease present or not (1 output)
      - disease head: multi-label over confident diseases (N outputs)
    '''

    def __init__(self, backbone='tf_efficientnetv2_s',
                 num_diseases=10, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features

        self.binary_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 1),
        )
        self.disease_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_diseases),
        )

    def forward(self, x):
        feats = self.backbone(x)
        binary_logit = self.binary_head(feats).squeeze(1)   # (B,)
        disease_logits = self.disease_head(feats)            # (B, N)
        return binary_logit, disease_logits
