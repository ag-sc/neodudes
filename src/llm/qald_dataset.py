import csv
import os
import sys
from typing import List, Optional, Dict

import lightning as L
import torch
import torch.nn.functional as F
import transformers
import lemon

from torch.utils.data import random_split, DataLoader, Dataset


class QALDDataset(Dataset):
    def __init__(self, data: List, pad_token_id: int, pad_multiplier: int = 512):
        self.data = data
        self.pad_token_id = pad_token_id
        self.pad_multiplier = pad_multiplier
        input_max_length = max([len(de["input"]) for de in data])
        self.input_target_length = 512 * (input_max_length // self.pad_multiplier + 1)
        output_max_length = max([len(de["target"]) for de in data])
        self.output_target_length = 512 * (output_max_length // self.pad_multiplier + 1)
        raw_input_max_length = max([len(list(bytes(de["raw_input"], 'utf8'))) for de in data])
        self.input_target_length_raw = 512 * (raw_input_max_length // self.pad_multiplier + 1)
        raw_output_max_length = max([len(list(bytes(de["raw_target"], 'utf8'))) for de in data])
        self.output_target_length_raw = 512 * (raw_output_max_length // self.pad_multiplier + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        de: Dict[str, str] = self.data[idx]
        input: torch.Tensor = torch.LongTensor(de["input"])
        input = F.pad(input, (0, self.input_target_length - len(de["input"])), value=self.pad_token_id)
        output: torch.Tensor = torch.LongTensor(de["target"])
        output = F.pad(output, (0, self.output_target_length - len(de["target"])), value=self.pad_token_id)
        raw_input: torch.Tensor = torch.ByteTensor(list(bytes(de["raw_input"], 'utf8')))
        raw_input = F.pad(raw_input, (0, self.input_target_length_raw - len(raw_input)), value=32)#32 is space
        raw_target: torch.Tensor = torch.ByteTensor(list(bytes(de["raw_target"], 'utf8')))
        raw_target = F.pad(raw_target, (0, self.output_target_length_raw - len(raw_target)), value=32)

        return {
            "input_token_ids": input,
            "output_token_ids": output,
            "raw_input": raw_input,
            "raw_target": raw_target,
        }


class QALDDataModule(L.LightningDataModule):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 train_csv_path: Optional[str] = None,
                 test_csv_path: Optional[str] = None,
                 batch_size: int = 32,
                 seed: Optional[int] = None):
        super().__init__()
        self.qald_train: Optional[Dataset] = None
        self.qald_val: Optional[Dataset] = None
        self.qald_test: Optional[Dataset] = None
        self.train_csv_path = train_csv_path if train_csv_path is not None else os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "QALD9_train-dataset-raw.csv")
        self.test_csv_path = test_csv_path if test_csv_path is not None else os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "QALD9_test-dataset-raw.csv")
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seed = seed

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            with open(self.train_csv_path) as csv_file:
                csv_dict = csv.DictReader(csv_file, delimiter=',')
                train_data_raw = [{
                    "input": self.tokenizer.encode(de["question"]),
                    "target": self.tokenizer.encode(de["sparql"]),
                    "raw_input": de["question"],
                    "raw_target": de["sparql"],
                } for de in list(csv_dict)]
                full_dataset = QALDDataset(train_data_raw, self.tokenizer.pad_token_id)

                if self.seed is None:
                    self.qald_train, self.qald_val = random_split(
                        full_dataset, [0.9, 0.1]
                    )
                else:
                    self.qald_train, self.qald_val = random_split(
                        full_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(self.seed)
                    )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            with open(self.test_csv_path) as csv_file:
                csv_dict = csv.DictReader(csv_file, delimiter=',')
                test_data_raw = [{
                    "input": self.tokenizer.encode(de["question"]),
                    "target": self.tokenizer.encode(de["sparql"]),
                    "raw_input": de["question"],
                    "raw_target": de["sparql"],
                } for de in list(csv_dict)]
                self.qald_test = QALDDataset(test_data_raw, self.tokenizer.pad_token_id)


    def train_dataloader(self):
        assert self.qald_train is not None
        return DataLoader(self.qald_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        assert self.qald_val is not None
        return DataLoader(self.qald_val, shuffle=True, batch_size=self.batch_size)

    def test_dataloader(self):
        assert self.qald_test is not None
        return DataLoader(self.qald_test, shuffle=True, batch_size=self.batch_size)

    def predict_dataloader(self):
        assert self.qald_test is not None
        return DataLoader(self.qald_test, shuffle=True, batch_size=self.batch_size)
