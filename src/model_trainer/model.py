import torch.nn as nn
from model_trainer import utils
from torch.utils.data import Dataset

class CompressionDataset(Dataset):
    def __init__(self, files):
        self.files = files
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_data = utils.get_file_data(self.files[index], 2048)
        title_embedding = utils.get_word_embedding(utils.get_title(self.files[index]))
        return file_data, title_embedding

class CompressionModel(nn.Module):
    def __init__(self):
        super(CompressionModel, self).__init__()
        self.fc = nn.Linear(2048, 768)

    def forward(self, x):
        return self.fc(x.float().unsqueeze(0)).squeeze(0)
        
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, x, y):
        return 1 - utils.get_sim(x, y)
