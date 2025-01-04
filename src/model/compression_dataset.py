import helpers.utils as utils
from torch.utils.data import Dataset
from helpers.embedding_extractor import EmbeddingExtractor

class CompressionDataset(Dataset):
    def __init__(self, files, device, batch_size=32):
        self.files = files
        self.embedding_extractor = EmbeddingExtractor(device)
        self.title_embeddings = self._compute_embeddings(batch_size)

    def _compute_embeddings(self, batch_size):
        embeddings = []
        for i in range(0, len(self.files), batch_size):
            batch_files = self.files[i:i + batch_size]
            batch_titles = [utils.get_title(file) for file in batch_files]
            batch_embeddings = self.embedding_extractor.get_embeddings(batch_titles)
            embeddings.extend(batch_embeddings)
        return embeddings

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_data = utils.get_file_data(self.files[index], 2048)
        title_embedding = self.title_embeddings[index]
        return file_data, title_embedding
