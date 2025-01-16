import os
import base64
import random
from tqdm import tqdm
from datasets import Dataset, DatasetDict

def load_compressed_files(folder_path):
    file_names = os.listdir(folder_path)
    compressed_files = []
    
    # 遍历文件夹中的所有文件
    for file_name in tqdm(file_names, desc=f"Processing files in {folder_path}"):
        if file_name.endswith('.zip'):  # 只处理.zip文件
            file_path = os.path.join(folder_path, file_name)
            
            # 读取压缩包内容并进行Base64编码
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # 将二进制数据编码为Base64字符串
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            compressed_files.append((encoded_content, file_name))
    
    return compressed_files

def load_text_files(folder_path):
    file_names = os.listdir(folder_path)
    text_files = []
    
    # 遍历文件夹中的所有文件
    for file_name in tqdm(file_names):
        if file_name.endswith('.txt'):  # 只处理.txt文件
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as f:
                text_files.append((f.read(), file_name))
            
    return text_files

def get_dataset(files):
    # 创建数据集
    data = []
    for idx, (question1, question2) in tqdm(enumerate(files, start=1)):
        data.append({
            "id": idx,
            "question1": question1,
            "question2": os.path.splitext(question2)[0],  # 去掉.zip后缀
            "is_duplicate": 1
        })

    for _ in tqdm(range(len(data))):
        # 随机选择两个不同的样本
        q1, q2 = random.sample(data, 2)  # 保证选出的样本不相同
        new_entry = {
            "id": len(data) + 1,  # 新的ID
            "question1": q1["question1"],
            "question2": q2["question2"],
            "is_duplicate": 0  # 设置为0，表示不重复
        }
        data.append(new_entry)

    # 打乱数据
    random.shuffle(data)

    train_data = data[len(data)//20:]
    validation_data = data[:len(data)//20]

    # 创建 Dataset 对象
    train_dataset = Dataset.from_dict({key: [d[key] for d in train_data] for key in train_data[0].keys()})
    validation_dataset = Dataset.from_dict({key: [d[key] for d in validation_data] for key in validation_data[0].keys()})

    # 创建 DatasetDict 对象
    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })

def get_dataset_from_zip():
    base_folder_path = '../dataset'
    all_files = []

    for i in range(10):  # 遍历000到009文件夹
        folder_path = os.path.join(base_folder_path, f'{i:03}')
        all_files.extend(load_compressed_files(folder_path))

    return get_dataset(all_files)

def get_dataset_from_txt():
    folder_path = '../dataset/200'
    return get_dataset(load_text_files(folder_path))