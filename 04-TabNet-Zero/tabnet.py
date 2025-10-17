import torch, torch.nn as nn
class TabNet(nn.Module):
    def __init__(self,inp,out): super().__init__(); self.bn=nn.BatchNorm1d(inp); self.fc=nn.Linear(inp,out)
    def forward(self,x): return self.fc(self.bn(x))

