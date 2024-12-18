import torch.nn as nn

class CompressionModel(nn.Module):
    def __init__(self):
        super(CompressionModel, self).__init__()
        self.fc = nn.Linear(2048, 768)

    def forward(self, x):
        return self.fc(x.float().unsqueeze(0)).squeeze(0)
