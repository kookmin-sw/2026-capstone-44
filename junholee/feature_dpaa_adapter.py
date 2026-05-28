import torch
import torch.nn as nn


class FeatureEnhancerDPAA(nn.Module):
    """
    Feature Enhancer 위치용 DPAA.
    입력은 transformer / feature enhancer로 들어가기 직전의 feature map:
    x: [B, C, H, W]
    """
    def __init__(self, dim, mid_dim=64, kernel_l=15, kernel_s=3, scale=1.0):
        super().__init__()

        self.down = nn.Conv2d(dim, mid_dim, kernel_size=1)

        self.conv_large = nn.Conv2d(
            mid_dim,
            mid_dim,
            kernel_size=kernel_l,
            padding=kernel_l // 2,
            groups=mid_dim,
        )

        self.conv_small = nn.Conv2d(
            mid_dim,
            mid_dim,
            kernel_size=kernel_s,
            padding=kernel_s // 2,
            groups=mid_dim,
        )

        self.act = nn.GELU()
        self.up = nn.Conv2d(mid_dim, dim, kernel_size=1)
        self.scale = scale

        # 초기에는 원래 GroundingDINO 출력을 거의 유지하되,
        # Feature-DPAA 출력이 완전히 막히지 않도록 small normal init 사용
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x

        x = self.down(x)
        x = self.conv_large(x) + self.conv_small(x)
        x = self.act(x)
        x = self.up(x)

        return residual + self.scale * x


def _get_proj_out_channels(proj_module):
    """
    input_proj 모듈의 출력 channel 수를 자동으로 찾는다.
    보통 GroundingDINO SwinT에서는 256.
    """
    last_out = None

    for m in proj_module.modules():
        if isinstance(m, nn.Conv2d):
            last_out = m.out_channels

    if last_out is None:
        last_out = 256

    return int(last_out)


def insert_feature_dpaa(
    model,
    mid_dim=64,
    kernel_l=15,
    kernel_s=3,
    scale=1.0,
    verbose=True,
):
    """
    model.input_proj 이후 feature map에 적용할 DPAA ModuleList를 추가.
    실제 forward 적용은 groundingdino.py 안에서
    if hasattr(self, "feature_dpaa") 로 수행한다.
    """
    if not hasattr(model, "input_proj"):
        raise AttributeError("model has no input_proj. Cannot insert FeatureEnhancer-DPAA.")

    adapters = []

    for i, proj in enumerate(model.input_proj):
        dim = _get_proj_out_channels(proj)

        adapters.append(
            FeatureEnhancerDPAA(
                dim=dim,
                mid_dim=mid_dim,
                kernel_l=kernel_l,
                kernel_s=kernel_s,
                scale=scale,
            )
        )

        if verbose:
            print(
                f"[Feature-DPAA] inserted after input_proj[{i}], "
                f"dim={dim}, mid_dim={mid_dim}, "
                f"k_large={kernel_l}, k_small={kernel_s}"
            )

    model.feature_dpaa = nn.ModuleList(adapters)

    print(f"[Feature-DPAA] total inserted adapters: {len(adapters)}")
    return model


def freeze_except_feature_dpaa(model):
    """
    GroundingDINO 전체 freeze.
    오직 feature_dpaa 파라미터만 학습 허용.
    """
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "feature_dpaa" in name:
            param.requires_grad = True

    return model


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    print("\n[Trainable parameter names]")
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print(name)

    ratio = 100 * trainable / total if total > 0 else 0.0

    print("\n[Params]")
    print(f"trainable: {trainable:,}")
    print(f"total: {total:,}")
    print(f"trainable ratio: {ratio:.4f}%")
