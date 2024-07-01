import statistics
from typing import Optional

import torch
from lightning import LightningModule
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoConfig

from dudes import utils
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint


class QueryScoreModel(LightningModule):
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    def __init__(self, encoder_size, learning_rate: float = 1e-5, ld: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.ld = ld
        self.encoder_size = encoder_size

        self.decision = nn.Sequential(
            nn.Linear(2*self.encoder_size, 2),
            #nn.Sigmoid()
        )

    def forward(self, inputs):
        bat_inputs = inputs.reshape((inputs.shape[0]*2, inputs.shape[1]//2))
        mo = self.model(input_ids=bat_inputs)
        mohs = mo.last_hidden_state.sum(axis=1)
        mohs2 = mohs.reshape((mohs.shape[0]//2, mohs.shape[1]*2))
        return self.decision(mohs2)

    def training_step(self, batch, batch_idx):
        self.train()
        inputs = batch["input_token_ids"]
        target = batch["output_values"]
        loss = nn.BCEWithLogitsLoss()
        res = self(inputs)#.mean(axis=1)
        #if any(res > 1.0) or any(res < 0.0) or any(target > 1.0) or any(target < 0.0):
        #    print("res", res)
        #    print("target", target, flush=True)
        loss_val = loss(res, target)
        self.log("train_loss", loss_val, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        #print("\n", loss_val.item())
        return loss_val

    def validation_step(self, batch, batch_idx):
        self.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            inputs = batch["input_token_ids"]
            target = batch["output_values"]
            loss = nn.BCEWithLogitsLoss()
            res = self(inputs)#.mean(axis=1)
            #if any(res > 1.0) or any(res < 0.0) or any(target > 1.0) or any(target < 0.0):
            #    print("res", res)
            #    print("target", target, flush=True)
            loss_val = loss(res, target)
            self.log("val_loss", loss_val, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            #print("\n", loss_val.item())
            #return loss_val#may lead to oom?

    def test_step(self, batch, batch_idx):
        self.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            inputs = batch["input_token_ids"]
            target = batch["output_values"]
            loss = nn.BCEWithLogitsLoss()
            res = self(inputs)#.mean(axis=1)
            #if any(res > 1.0) or any(res < 0.0) or any(target > 1.0) or any(target < 0.0):
            #    print("res", res)
            #    print("target", target, flush=True)
            loss_val = loss(res, target)
            if batch_idx % 100 == 0:
                print("Test loss: ", loss_val)
            self.log("test_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            #print("\n", loss_val.item())
            #return loss_val#may lead to oom?

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
