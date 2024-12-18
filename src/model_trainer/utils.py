import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

bert_tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
bert_model = BertModel.from_pretrained('./bert-base-chinese')

def get_files():
    folder_paths = [f'./output/{i:03d}' for i in range(10)]
    files = []
    for folder_path in tqdm(folder_paths):
        files.extend([os.path.join(folder_path, file) for file in os.listdir(folder_path)])
    return files

def get_file_data(file_path, max_length):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    file_data = torch.tensor([c for c in file_data] + [0] * (max_length - len(file_data)))
    return file_data[:max_length]

def get_title(file_path):
    file_name = os.path.basename(file_path)
    return file_name.split('.')[0]

def get_word_embedding(word):
    inputs = bert_tokenizer(word, return_tensors='pt')
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

def get_sim(x, y):
    return torch.cosine_similarity(x, y, dim=0)
