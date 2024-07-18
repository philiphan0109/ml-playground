import torch
import torch.nn as nn


class relumodel(nn.Module):
    def __init__(self):
        super(relumodel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        
    
    def forward(self, x):
        return self.layers(x)
