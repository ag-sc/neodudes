import torch
from lightning import LightningModule
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, T5EncoderModel

from llm.query_score_models.query_score_model import QueryScoreModel


class QueryScoreT5(QueryScoreModel):
    def __init__(self, model_name: str = "google/flan-t5-small", learning_rate: float = 1e-5, ld: float = 0.9):
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        config = T5Config.from_pretrained(model_name)
        super().__init__(encoder_size=config.d_model, learning_rate=learning_rate, ld=ld)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)

class QueryScoreT5Base(QueryScoreModel):
    def __init__(self, model_name: str = "google/flan-t5-base", learning_rate: float = 1e-5, ld: float = 0.9):
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        config = T5Config.from_pretrained(model_name)
        super().__init__(encoder_size=config.d_model, learning_rate=learning_rate, ld=ld)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
