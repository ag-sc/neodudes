import torch
from lightning import LightningModule
from transformers import T5ForConditionalGeneration, T5Tokenizer

from llm.qald_models.qald_model import QaldModel


class QaldT5(QaldModel):
    def __init__(self, model_name: str = "google-t5/t5-base", learning_rate: float = 1e-5, ld: float = 0.9, sparql_endpoint=None):
        super().__init__(learning_rate=learning_rate, ld=ld, sparql_endpoint=sparql_endpoint)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
