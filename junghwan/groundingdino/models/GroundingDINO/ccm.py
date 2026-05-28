import torch
import torch.nn as nn


class CategoricalCounting(nn.Module):
    """Categorical Counting Module (CCM) from DQ-DETR.

    Predicts object density class from encoder features to dynamically
    determine the number of decoder queries (top-k selection count).

    Args:
        cls_num: Number of density classes (default 4).

    Input:
        features: [bs, hw, 256]  — visual encoder output (memory)
        spatial_shapes: Tensor[num_levels, 2]  — (H, W) per feature level

    Output:
        out: [bs, cls_num]  — density class logits
        x:   [bs, 256, h, w]  — intermediate spatial feature map
    """

    def __init__(self, cls_num: int = 4):
        super().__init__()
        self.ccm_cfg = [512, 512, 512, 256, 256, 256]
        self.in_channels = 512
        self.conv1 = nn.Conv2d(256, self.in_channels, kernel_size=1)
        self.ccm = _make_layers(self.ccm_cfg, in_channels=self.in_channels, d_rate=2)
        self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(256, cls_num)

    def forward(self, features: torch.Tensor, spatial_shapes: torch.Tensor):
        features = features.transpose(1, 2)                   # [bs, 256, hw]
        bs = features.shape[0]
        # spatial_shapes is a Tensor in Grounding DINO (vs list of tuples in DQ-DETR)
        h = spatial_shapes[0][0].item()
        w = spatial_shapes[0][1].item()
        v_feat = features[:, :, 0:h * w].view(bs, 256, h, w)
        x = self.conv1(v_feat)                                # [bs, 512, h, w]
        x = self.ccm(x)                                       # [bs, 256, h, w]
        out = self.output(x).squeeze(3).squeeze(2)            # [bs, 256]
        out = self.linear(out)                                # [bs, cls_num]
        return out, x


def _make_layers(cfg, in_channels: int = 3, d_rate: int = 1) -> nn.Sequential:
    layers = []
    for v in cfg:
        layers += [
            nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate),
            nn.ReLU(inplace=True),
        ]
        in_channels = v
    return nn.Sequential(*layers)
