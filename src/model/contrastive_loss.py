import utils
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, x, y):
        return 1 - utils.get_sim(x, y)
