import json
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_data(path):
    samples = []
    
    # total lines
    with open(path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
        
    # read line
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="Load training data"):
            data = json.loads(line.strip())
            samples.append(data)  
    return samples


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer: AutoTokenizer, max_length: int=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = _load_data(data_path)
    
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
        
        input_ids = encoding["input_ids"].squeeze()  # (1, L_max) -> (L_max)
    
        # autoregressive
        # [x1, x2, x3, x4] -> [x2, x3, x4, x5]
        x = input_ids[:-1].to(dtype=torch.long)  # ignore the last token
        y = input_ids[1:].to(dtype=torch.long)   # ignore the first token
        loss_mask = encoding["attention_mask"].squeeze()[1:].to(dtype=torch.long)   # ignore padding
         
        return x, y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer: AutoTokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = _load_data(data_path)
        
        # bos and eos
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids  # <|im_start|>assistant
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids     # <|im_end|>
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # build prompts
        prompt = self._create_chat_prompt(sample['conversations'])      # apply chate template
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]  # convert to token_id and chunk
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))  # padding

        # build mask
        loss_mask = self._generate_loss_mask(input_ids)
        
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask
        
    def _create_chat_prompt(self, chat: list[dict[str, str]]):
        messages = chat.copy()
        
        # parse tools
        if chat and chat[0]["role"] == "system" and chat[0].get("functions"):
            tools = chat[0]["functions"]
        else:
            tools = None
        
        # apply chat tempalte
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
    
    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)    # 0: discount
        i = 0
        
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                # find tokens that within eos and bos
                start = i + len(self.bos_id)
                end = start
                
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # update
                slice_start = start + 1
                slice_end = min(end + len(self.eos_id) + 1, self.max_length)
                loss_mask[slice_start: slice_end] = [1] * (slice_end - slice_start)    

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)  # next
            else:
                i += 1
        
        return loss_mask