import statistics
from typing import Optional

import torch
from lightning import LightningModule
from transformers import PreTrainedTokenizer, PreTrainedModel

from dudes import utils
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint


class QaldModel(LightningModule):
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    sparql_endpoint: Optional[SPARQLEndpoint]

    def __init__(self, learning_rate: float = 1e-5, ld: float = 0.9, sparql_endpoint: Optional[SPARQLEndpoint] = None):
        super().__init__()
        self.learning_rate = learning_rate
        self.ld = ld
        self.sparql_endpoint = sparql_endpoint

    def forward(self, inputs, target):
        return self.model(input_ids=inputs, labels=target)

    def training_step(self, batch, batch_idx):
        inputs = batch["input_token_ids"]
        target = batch["output_token_ids"]
        loss = self(inputs, target).loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        print("\n", loss.item())
        return loss

    def _do_eval(self, metric_name, batch):
        if self.sparql_endpoint is None:
            print("Warning: No SPARQL endpoint provided, only returning loss.", metric_name)
            inputs = batch["input_token_ids"]
            target = batch["output_token_ids"]
            loss = self(inputs, target).loss
            self.log(metric_name, loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss.item()
        else:
            #inputs = batch["input_token_ids"].tolist()
            target = batch["raw_target"].tolist()
            sparql_golds = ["".join([chr(i) for i in elems]).strip() for elems in target]#self.tokenizer.batch_decode(target, skip_special_tokens=True)
            preds = self.model.generate(batch["input_token_ids"])
            sparql_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            f1s = []

            for i, (gold, pred) in enumerate(zip(sparql_golds, sparql_preds)):
                try:
                    gold_res = self.sparql_endpoint.get_results_query(gold)
                    pred_res = self.sparql_endpoint.get_results_query(pred)

                    tstats = utils.compare_results(gold_res=gold_res, sys_res=pred_res)
                    f1s.append(tstats.f1 if tstats.f1 is not None else 0.0)
                    self.log(metric_name, 1.0 - tstats.f1 if tstats.f1 is not None else 1.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                    print(tstats)
                except Exception as e:
                    print(e, i, gold, pred)
                    self.log(metric_name, 1.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return (1.0 - statistics.mean(f1s)) if len(f1s) > 0 else 1.0 # invert to imitate loss, i.e. hyperparameter optimization into same direction, minimizing
    def validation_step(self, batch, batch_idx):
        self._do_eval(metric_name="val_loss", batch=batch)

    def test_step(self, batch, batch_idx):
        self._do_eval(metric_name="test_loss", batch=batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        lambda1 = lambda epoch: self.ld ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }