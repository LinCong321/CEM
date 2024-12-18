import utils
import torch
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from model.contrastive_loss import ContrastiveLoss
from model.compression_model import CompressionModel
from model.compression_dataset import CompressionDataset

def train(device):
    compression_model = CompressionModel().to(device)
    criterion = ContrastiveLoss().to(device)
    optimizer = optim.Adam(compression_model.parameters(), lr=0.001)

    files = utils.get_files()[:1000]
    dataset = CompressionDataset(files[:1000], device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in tqdm(range(20), desc = 'Epoch'):
        for batch in tqdm(dataloader, desc='Batch'):
            # 将数据迁移到设备
            file_data, title_embeddings = batch
            file_data = file_data.float().to(device)
            title_embeddings = title_embeddings.float().to(device)
            # 前向传播
            compression_vectors = compression_model(file_data)
            loss = criterion(compression_vectors, title_embeddings)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

    torch.save(compression_model.state_dict(), './compression_model/compression_model.pth')
    print("Model saved to './compression_model/compression_model.pth'")
