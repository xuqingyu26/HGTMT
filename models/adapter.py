import torch.nn as nn
import torch.nn.functional as F

class AdapterModule(nn.Module):
    def __init__(self,
                 in_feature,
                 ADAPTER_BOTTLENECK
    ):
        super().__init__()

        self.proj_down = nn.Linear(in_features=in_feature, out_features=ADAPTER_BOTTLENECK)
        self.proj_up = nn.Linear(in_features=ADAPTER_BOTTLENECK, out_features=in_feature)

    def forward(self, x):
        input = x.clone()

        x = self.proj_down(x)
        x = F.relu(x)
        return self.proj_up(x) + input # Skip Connection