import torch
import torch.nn as nn


class approximator(nn.Module):
    def __init__(self):
        super(approximator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
        
    
    def forward(self, x):
        return self.layers(x)
