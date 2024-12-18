import torch

from tqdm import tqdm
from model_trainer import utils
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model_trainer.model import CompressionModel, CompressionDataset

def test():
    compression_model = CompressionModel()
    compression_model.load_state_dict(torch.load('./model/compression_model.pth'))

    files = utils.get_files()[:1000]
    dataset = CompressionDataset(files)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    test_accuracy = []
    test_losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Test Batch'):
            file_data, title_embeddings = batch
            compression_embeddings = compression_model(file_data)
            predictions = get_sim(title_embeddings, compression_embeddings)
            accuracy = accuracy_score([1 if prediction > 0.8 else 0 for prediction in predictions],
                                      [1] * len(predictions))
            loss = criterion(title_embeddings, compression_embeddings)
            test_accuracy.append(accuracy)
            test_losses.append(loss.mean())

    print(f'Test Accuracy:{torch.mean(torch.tensor(test_accuracy)):.4f}')
    print(f'Test Loss:{torch.mean(torch.tensor(test_losses)):.4f}')
