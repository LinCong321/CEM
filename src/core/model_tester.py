import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from model.contrastive_loss import ContrastiveLoss
from model.compression_model import CompressionModel
from model.compression_dataset import CompressionDataset

from helpers import utils
from helpers.embedding_extractor import EmbeddingExtractor

def test(device):
    compression_model = load_model(device)
    criterion = ContrastiveLoss().to(device)

    files = utils.get_files()[:1000]
    dataset = CompressionDataset(files, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    test_accuracy = []
    test_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Test Batch'):
            # 将数据迁移到设备
            file_data, title_embeddings = batch
            file_data = file_data.to(device)
            title_embeddings = title_embeddings.to(device)
            # 计算准确度和损失
            compression_embeddings = compression_model(file_data)
            predictions = utils.get_sim(title_embeddings, compression_embeddings)
            accuracy = accuracy_score([1 if prediction > 0.8 else 0 for prediction in predictions],
                                      [1] * len(predictions))
            loss = criterion(title_embeddings, compression_embeddings)
            # 存储结果
            test_accuracy.append(accuracy)
            test_losses.append(loss.mean().item())

    print(f'Test Accuracy: {torch.mean(torch.tensor(test_accuracy)):.4f}')
    print(f'Test Loss: {torch.mean(torch.tensor(test_losses)):.4f}')

def load_model(device):
    compression_model = CompressionModel().to(device)
    compression_model.load_state_dict(torch.load('./compression_model/compression_model.pth', map_location=device))
    return compression_model

def ann(word):
    device = utils.get_device()
    embedding_extractor = EmbeddingExtractor(device)
    embedding = embedding_extractor.get_embedding(word)
    compression_model = load_model(device)
    files = utils.get_files()[:1000]
    dataset = CompressionDataset(files, device)

    result = []
    for i in range(1000):
        sim = utils.get_sim(embedding, compression_model(dataset[i][0].to(device)))
        result.append((utils.get_title(files[i]), sim))
    sorted_result = sorted(result, key=lambda x : x[1], reverse=True)
    
    for i in range(10):
        print(sorted_result[i])