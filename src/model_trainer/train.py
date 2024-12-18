import torch
import torch.optim as optim

from tqdm import tqdm
from model_trainer import utils
from torch.utils.data import DataLoader
from model_trainer.model import CompressionModel, ContrastiveLoss, CompressionDataset

def train():
    files = utils.get_files()[:1000]
    compression_model = CompressionModel()
    criterion = ContrastiveLoss()

    for epoch in tqdm(range(50), desc = 'Epoch'):
        dataset = CompressionDataset(files[:1000])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for batch in tqdm(dataloader, desc='Batch'):
            file_data, title_embeddings = batch
            compression_vectors = compression_model(file_data.float().requires_grad_(True))
            loss = criterion(compression_vectors, title_embeddings.float().requires_grad_(True))
            optimizer = optim.Adam(compression_model.parameters(), lr=0.001)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

    torch.save(compression_model.state_dict(), './model/compression_model.pth')
