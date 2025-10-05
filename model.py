import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(module: nn.Module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class CNNcifar(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 64, dropout: float = 0.0):
        super().__init__()
        w1, w2, w3 = width, width * 2, width * 4

        # Stem (32x32 유지)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),

            nn.Conv2d(w1, w1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
        )

        # Stage 2 (32 -> 16)
        self.stage2 = nn.Sequential(
            nn.Conv2d(w1, w2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),

            nn.Conv2d(w2, w2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
        )

        # Stage 3 (16 -> 8)
        self.stage3 = nn.Sequential(
            nn.Conv2d(w2, w3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),

            nn.Conv2d(w3, w3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(w3, 1, bias=True)

        
        self.apply(kaiming_init) 

    @property
    def feature_dim(self):
        return self.head.in_features
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                 # (N, w1, 32, 32)
        x = self.stage2(x)               # (N, w2, 16, 16)
        x = self.stage3(x)               # (N, w3, 8, 8)
        x = F.adaptive_avg_pool2d(x, 1)  # (N, w3, 1, 1)
        x = torch.flatten(x, 1)          # (N, w3)
        return x         



def build_model(arch, **kwargs):
        return CNNcifar(**kwargs)
