import sys
import os

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import json
import torch
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset
from model.model_olm import MiniMindOLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OLMDataset(Dataset):
    def __init__(
            self,
            parquet_path,
            tokenizer,
            preprocess=None,
            max_length=512,
            image_special_token='@' * 196,
            speech_special_token='#' * 100,
    ):
        super().__init__()
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.speech_token = speech_special_token
        self.columns = set(self.table.column_names)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            content = turn['content'].replace('<image>', self.image_token).replace('<speech>', self.speech_token)
            messages.append({"role": role, "content": content})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def _load_image_tensor(self, index):
        if 'image_bytes' not in self.columns:
            return torch.zeros((1, 1, 3, 224, 224), dtype=torch.float32)

        image_bytes = self.table['image_bytes'][index].as_py()
        if image_bytes is None:
            return torch.zeros((1, 1, 3, 224, 224), dtype=torch.float32)

        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]

        image_tensors = []
        for img in image_bytes[:1]:
            image_tensors.append(
                MiniMindOLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess)
            )
        return torch.stack(image_tensors)

    def _load_speech_tensor(self, index):
        if 'speech_path' not in self.columns:
            zeros = torch.zeros((1, 3000, 128), dtype=torch.float32)
            lengths = torch.LongTensor([3000])
            return zeros, lengths

        speech_path = self.table['speech_path'][index].as_py()
        if speech_path is None:
            zeros = torch.zeros((1, 3000, 128), dtype=torch.float32)
            lengths = torch.LongTensor([3000])
            return zeros, lengths

        if isinstance(speech_path, list):
            speech_path = speech_path[0]

        speech_tensor = MiniMindOLM.speech2tensor(speech_path)
        speech_tensor = speech_tensor.unsqueeze(0)
        speech_lengths = torch.LongTensor([speech_tensor.size(1)])
        return speech_tensor, speech_lengths

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        prompt = self.create_chat_prompt(conversations)

        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        image_tensor = self._load_image_tensor(index)
        speech_tensor, speech_lengths = self._load_speech_tensor(index)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            image_tensor,
            speech_tensor,
            speech_lengths,
        )
