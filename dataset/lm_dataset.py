import json
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer: AutoTokenizer, max_length: int=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)
    
    def load_data(self, path):
        samples = []
        
        # total lines
        with open(path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        
        # read line
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total, desc="Loading pretrain data"):
                data = json.loads(line.strip())
                samples.append(data)  
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # tokenizer sample
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        
        loss_mask = input_ids != self.tokenizer.pad_token_id # 1 for valid tokens
        
        # autoregressive
        # [x1, x2, x3, x4] -> [x2, x3, x4, x5]
        x = torch.tensor(input_ids[:-1], dtype=torch.long)  # ignore the last token
        y = torch.tensor(input_ids[1:], dtype=torch.long)   # ignore the first token
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)   # ignore padding
        
        return x, y, loss_mask
        
        
