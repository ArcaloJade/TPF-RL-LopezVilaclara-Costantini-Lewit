import torch
import torch.nn as nn

class ValueNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLu(),
            nn.Linear(256, 64),
            nn.ReLu(),
            nn.Linear(64, output_dim)

        )
    
    def forward(self, x):
        return self.fc(x)