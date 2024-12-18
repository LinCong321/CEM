import torch
import utils

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model.contrastive_loss import ContrastiveLoss
from model.compression_model import CompressionModel
from model.compression_dataset import CompressionDataset

def test(device):
    compression_model = CompressionModel().to(device)
    compression_model.load_state_dict(torch.load('./compression_model/compression_model.pth', map_location=device))
    print("Model loaded from './compression_model/compression_model.pth'")

    files = utils.get_files()[:1000]
    dataset = CompressionDataset(files, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    criterion = ContrastiveLoss().to(device)

    test_accuracy = []
    test_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Test Batch'):
            file_data, title_embeddings = batch
            file_data = file_data.to(device)
            title_embeddings = title_embeddings.to(device)
            compression_embeddings = compression_model(file_data)

            predictions = utils.get_sim(title_embeddings, compression_embeddings)
            accuracy = accuracy_score([1 if prediction > 0.8 else 0 for prediction in predictions],
                                      [1] * len(predictions))
            loss = criterion(title_embeddings, compression_embeddings)

            test_accuracy.append(accuracy)
            test_losses.append(loss.mean().item())

    print(f'Test Accuracy: {torch.mean(torch.tensor(test_accuracy)):.4f}')
    print(f'Test Loss: {torch.mean(torch.tensor(test_losses)):.4f}')