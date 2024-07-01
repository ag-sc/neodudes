import csv
import os
import compress_pickle as cpl
import random
import sys
from typing import List, Optional, Dict, Set

import lightning as L
import more_itertools
import torch
import torch.nn.functional as F
import transformers
import lemon
import pandas as pd

from torch.utils.data import random_split, DataLoader, Dataset

from dudes import utils, consts


class QueryScoreDataset(Dataset):
    def __init__(self, data: List, pad_token_id: int, pad_multiplier: int = 512):
        self.data = data
        self.pad_token_id = pad_token_id
        self.pad_multiplier = pad_multiplier
        input_max_length = max([len(de["input1"]) for de in data] + [len(de["input2"]) for de in data])
        self.input_target_length = 512 * (input_max_length // self.pad_multiplier + (0 if input_max_length % self.pad_multiplier == 0 else 1))
        print("self.input_target_length", self.input_target_length, flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        de: Dict[str, str] = self.data[idx]
        input1: torch.Tensor = torch.LongTensor(de["input1"])
        input1 = F.pad(input1, (0, self.input_target_length - len(de["input1"])), value=self.pad_token_id)
        input2: torch.Tensor = torch.LongTensor(de["input2"])
        input2 = F.pad(input2, (0, self.input_target_length - len(de["input2"])), value=self.pad_token_id)
        output: torch.Tensor = torch.Tensor(de["target"])

        return {
            "input_token_ids": torch.cat((input1, input2)),
            "output_values": output,
        }

class QueryScoreDataModule(L.LightningDataModule):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 train_csv_path: Optional[str] = None,
                 test_csv_path: Optional[str] = None,
                 dataset_type: str = "f1",
                 batch_size: int = 32,
                 combs_per_question: int = 100):
        super().__init__()
        self.query_score_train: Optional[Dataset] = None
        self.query_score_val: Optional[Dataset] = None
        self.query_score_test: Optional[Dataset] = None
        self.train_csv_path = train_csv_path if train_csv_path is not None else os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "all-queries-train_unique3.csv.zst")
        self.test_csv_path = test_csv_path if test_csv_path is not None else os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "all-queries-test_unique.csv.zst")
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.combs_per_question = combs_per_question
        self.query_score_train = None
        self.query_score_val = None
        self.query_score_test = None
        self.dataset_type = dataset_type
        assert self.dataset_type.lower() in ["f1", "f1-novalid", "clampfp", "clampfp-novalid"]

        # self.input_template = "Question: {question}" + self.tokenizer.pad_token
        # self.input_template += "SPARQL: {sparql1} DUDES: {dudes1} Number of results: {numres1}" + self.tokenizer.pad_token
        # self.input_template += "SPARQL: {sparql2} DUDES: {dudes2} Number of results: {numres2}"
    def prepare_data(self):
        if self.dataset_type.lower() in ["f1", "f1-novalid"]:
            float_func = self._get_f1
        elif self.dataset_type.lower() in ["clampfp", "clampfp-novalid"]:
            float_func = self._clamp_fprate
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        if not os.path.isfile(self.train_csv_path+f"-{self.dataset_type}.cpkl") or not os.path.isfile(self.train_csv_path+f"-{self.dataset_type}-valid.cpkl"):
            df = pd.read_csv(self.train_csv_path)
            train_ids = df['id'].unique().tolist()
            valid_ids = set(random.sample(train_ids, len(train_ids) // 10))
            train_ids = set(train_ids) - valid_ids
            df1 = df[df["F1"] >= 0.01]
            df2 = df[df["F1"] < 0.01]


            train_data_raw = []
            query_set = set()
            td, qs = self._gen_data(df1, train_ids, float_func=float_func)
            train_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data(df2, train_ids, float_func=float_func)
            train_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data_comb(df1, df2, train_ids, float_func=float_func)
            train_data_raw += td
            query_set.update(qs)

            if self.dataset_type.lower().endswith("-novalid"):
                td = self._gen_not_valid_data(df, query_set)
                train_data_raw += td

            with open(self.train_csv_path+f"-{self.dataset_type}.cpkl", "wb") as f:
                cpl.dump(train_data_raw, f, compression="gzip")

            valid_data_raw = []
            query_set = set()
            td, qs = self._gen_data(df1, valid_ids, float_func=float_func)
            valid_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data(df2, valid_ids, float_func=float_func)
            valid_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data_comb(df1, df2, valid_ids, float_func=float_func)
            valid_data_raw += td
            query_set.update(qs)

            if self.dataset_type.lower().endswith("-novalid"):
                td = self._gen_not_valid_data(df, query_set)
                valid_data_raw += td

            with open(self.train_csv_path+f"-{self.dataset_type}-valid.cpkl", "wb") as f:
                cpl.dump(valid_data_raw, f, compression="gzip")

        if not os.path.isfile(self.test_csv_path+f"-{self.dataset_type}.cpkl"):
            df = pd.read_csv(self.test_csv_path)
            df1 = df[df["F1"] >= 0.01]
            df2 = df[df["F1"] < 0.01]
            test_data_raw = []
            query_set = set()
            td, qs = self._gen_data(df1, float_func=float_func)
            test_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data(df2, float_func=float_func)
            test_data_raw += td
            query_set.update(qs)
            td, qs = self._gen_data_comb(df1, df2, float_func=float_func)
            test_data_raw += td
            query_set.update(qs)

            if self.dataset_type.lower().endswith("-novalid"):
                td = self._gen_not_valid_data(df, query_set)
                test_data_raw += td

            with open(self.test_csv_path+f"-{self.dataset_type}.cpkl", "wb") as f:
                cpl.dump(test_data_raw, f, compression="gzip")

    @staticmethod
    def _input_string(de1: Dict[str, str]):
        return """Question: {question}
        Number of results: {numres1}
        SPARQL: 
        {sparql1} 
        DUDES: 
        {dudes1}""".format(
            question=de1["question"],
            numres1=int(de1['True Positive']) + int(de1['False Positive']),
            sparql1=utils.replace_namespaces_dirty(utils.remove_prefix(de1["Generated SPARQL"])),
            dudes1=de1["DUDES"],
        )


    def _gen_not_valid_data(self, df: pd.DataFrame, used_queries: Set[int]):
        train_data_raw = []

        # for q, indices in df.groupby('question').groups.items():
        #     if len(indices) < 2 or (valid_ids is not None and df.loc[indices[0]]["id"] not in valid_ids):
        #         continue
        for qid in used_queries:
            de = df.loc[qid]
            ede = self.tokenizer.encode(self._input_string(de), max_length=512, truncation=True)
            invalid_elem = self.tokenizer.encode(self._input_string({
                "question": de["question"],
                "True Positive": "0",
                "False Positive": "0",
                "Generated SPARQL": "No valid result yet.",
                "DUDES": "None"
            }), max_length=512, truncation=True)
            train_data_raw.append({
                "input1": ede,
                "input2": invalid_elem,
                "target": [0.0, 1.0] if self._has_high_fprate(de) else [1.0, 0.0],
                # "target": [(float(de1["F1"]) / 2.0) - (float(de2["F1"]) / 2.0) + 0.5],
                # norm to 0-1 with 0 = left, 1 = right
            })
            train_data_raw.append({
                "input1": invalid_elem,
                "input2": ede,
                "target": [1.0, 0.0] if self._has_high_fprate(de) else [0.0, 1.0]
                # "target": [(float(de2["F1"]) / 2.0) - (float(de1["F1"]) / 2.0) + 0.5],
                # norm to 0-1 with 0 = left, 1 = right
            })

        return train_data_raw

    @staticmethod
    def _has_high_fprate(de) -> bool:
        return (float(de["False Positive"]) + 0.001) / (float(de["True Positive"]) + 0.001) > consts.fp_ratio
    @staticmethod
    def _clamp_fprate(de) -> float:
        #(curr_stats.fp + 0.001) / (curr_stats.tp + 0.001) <= consts.fp_ratio
        return 0.0 if QueryScoreDataModule._has_high_fprate(de) else float(de["F1"])

    @staticmethod
    def _get_f1(de) -> float:
        return float(de["F1"])

    def _gen_data(self, df: pd.DataFrame, valid_ids: Optional[set] = None, float_func=None):
        if float_func is None:
            float_func = self._get_f1
        train_data_raw = []
        query_ids = set()
        for q, indices in df.groupby('question').groups.items():
            if len(indices) < 2 or (valid_ids is not None and df.loc[indices[0]]["id"] not in valid_ids):
                continue

            for i in range(self.combs_per_question):
                row_id1, row_id2 = more_itertools.random_combination(indices, 2)
                query_ids.add(row_id1)
                query_ids.add(row_id2)
                de1 = df.loc[row_id1]
                de2 = df.loc[row_id2]
                ede1 = self.tokenizer.encode(self._input_string(de1), max_length=512, truncation=True)
                ede2 = self.tokenizer.encode(self._input_string(de2), max_length=512, truncation=True)

                train_data_raw.append({
                    "input1": ede1,
                    "input2": ede2,
                    "target": [float_func(de1), float_func(de2)],
                    #"target": [(float(de1["F1"]) / 2.0) - (float(de2["F1"]) / 2.0) + 0.5],
                    # norm to 0-1 with 0 = left, 1 = right
                })
                train_data_raw.append({
                    "input1": ede2,
                    "input2": ede1,
                    "target": [float_func(de2), float_func(de1)],
                    #"target": [(float(de2["F1"]) / 2.0) - (float(de1["F1"]) / 2.0) + 0.5],
                    # norm to 0-1 with 0 = left, 1 = right
                })
        return train_data_raw, query_ids



    def _gen_data_comb(self, df1: pd.DataFrame, df2: pd.DataFrame, valid_ids: Optional[set] = None, float_func=None):
        if float_func is None:
            float_func = self._get_f1
        train_data_raw = []
        query_ids = set()
        for q, indices1 in df1.groupby('question').groups.items():
            if len(indices1) < 2 or (valid_ids is not None and df1.loc[indices1[0]]["id"] not in valid_ids):
                continue
            gr2 = df2.groupby('question').groups
            if q not in gr2:
                continue
            indices2 = gr2[q]
            for i in range(2*self.combs_per_question):
                #row_id1, row_id2 = more_itertools.random_combination(indices, 2)
                row_id1 = random.choice(indices1)
                row_id2 = random.choice(indices2)
                query_ids.add(row_id1)
                query_ids.add(row_id2)
                de1 = df1.loc[row_id1]
                de2 = df2.loc[row_id2]
                ede1 = self.tokenizer.encode(self._input_string(de1), max_length=512, truncation=True)
                ede2 = self.tokenizer.encode(self._input_string(de2), max_length=512, truncation=True)

                train_data_raw.append({
                    "input1": ede1,
                    "input2": ede2,
                    "target": [float_func(de1), float_func(de2)],
                    #"target": [(float(de1["F1"]) / 2.0) - (float(de2["F1"]) / 2.0) + 0.5],
                    # norm to 0-1 with 0 = left, 1 = right
                })
                train_data_raw.append({
                    "input1": ede2,
                    "input2": ede1,
                    "target": [float_func(de2), float_func(de1)],
                    #"target": [(float(de2["F1"]) / 2.0) - (float(de1["F1"]) / 2.0) + 0.5],
                    # norm to 0-1 with 0 = left, 1 = right
                })
        return train_data_raw, query_ids

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        #if stage == "fit" or stage is None:
        if self.query_score_train is None or self.query_score_val is None:
            with open(self.train_csv_path+f"-{self.dataset_type}.cpkl", "rb") as f:
                train_data_raw = cpl.load(f, compression="gzip")

            train_dataset = QueryScoreDataset(train_data_raw, self.tokenizer.pad_token_id)

            self.query_score_train = train_dataset

            with open(self.train_csv_path+f"-{self.dataset_type}-valid.cpkl", "rb") as f:
                valid_data_raw = cpl.load(f, compression="gzip")

            valid_dataset = QueryScoreDataset(valid_data_raw, self.tokenizer.pad_token_id)

            self.query_score_val = valid_dataset

            # if self.seed is None:
            #     self.query_score_train, self.query_score_val = random_split(
            #         full_dataset, [0.9, 0.1]
            #     )
            # else:
            #     self.query_score_train, self.query_score_val = random_split(
            #         full_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(self.seed)
            #     )

                # Assign test dataset for use in dataloader(s)
        #if stage == "test" or stage == "predict":
        if self.query_score_test is None:
            with open(self.test_csv_path+f"-{self.dataset_type}.cpkl", "rb") as f:
                test_data_raw = cpl.load(f, compression="gzip")

            test_dataset = QueryScoreDataset(test_data_raw, self.tokenizer.pad_token_id)

            self.query_score_test = test_dataset


    def train_dataloader(self):
        assert self.query_score_train is not None
        return DataLoader(self.query_score_train, shuffle=True, batch_size=self.batch_size)#, num_workers=8)

    def val_dataloader(self):
        assert self.query_score_val is not None
        return DataLoader(self.query_score_val, shuffle=False, batch_size=self.batch_size)#, num_workers=8)

    def test_dataloader(self):
        assert self.query_score_test is not None
        return DataLoader(self.query_score_test, shuffle=False, batch_size=self.batch_size)#, num_workers=8)

    def predict_dataloader(self):
        assert self.query_score_test is not None
        return DataLoader(self.query_score_test, shuffle=False, batch_size=self.batch_size)#, num_workers=8)
