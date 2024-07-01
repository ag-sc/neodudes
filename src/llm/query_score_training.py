import os.path
import sys
import traceback
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import optuna
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from optuna import Trial
from optuna.storages import JournalStorage, JournalFileStorage

from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from llm.qald_models.t5 import QaldT5
from llm.qald_dataset import QALDDataModule
from llm.query_score_dataset import QueryScoreDataModule
from llm.query_score_models.t5 import QueryScoreT5


def objective(trial: Trial,
              batch_size=None,
              learning_rate=None,
              ld=None,
              epochs=None,
              model_name="google/flan-t5-small",
              dataset_type="clampfp"):
    try:
        if batch_size is None:
            batch_size = 1
            # trial.suggest_int('batch_size', 1, 10)
        if learning_rate is None:
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
        if ld is None:
            ld = trial.suggest_float('ld', 0.9, 1.0, log=True)
        if epochs is None:
            epochs = trial.suggest_int('epochs', 1, 5)

        model = QueryScoreT5(model_name=model_name, learning_rate=learning_rate, ld=ld)
        dm = QueryScoreDataModule(tokenizer=model.tokenizer, batch_size=batch_size, dataset_type=dataset_type)
        dm.prepare_data()
        dm.setup("train")

        print("Batch size: ", batch_size, flush=True)
        print("Learning rate: ", learning_rate, flush=True)
        print("Lambda: ", ld, flush=True)
        print("Epochs: ", epochs, flush=True)
        print("Model: ", model_name, flush=True)
        print("Dataset: ", dataset_type, flush=True)

        logger = TensorBoardLogger("tb_logs",
                                   name="query_score_llm",
                                   version=None if 'SLURM_ARRAY_TASK_ID' not in os.environ else os.environ['SLURM_ARRAY_TASK_ID'])

        trainer = Trainer(enable_checkpointing=True,
                          logger=logger,
                          #accelerator="cpu",
                          reload_dataloaders_every_n_epochs=1,
                          log_every_n_steps=1,
                          max_epochs=epochs,
                          min_epochs=epochs,
                          strategy="ddp")
        trainer.fit(model, datamodule=dm)
        trainer.save_checkpoint(f"query_score_llm_{dataset_type}_{learning_rate}_{ld}_{batch_size}_{epochs}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.ckpt")
        print(trainer.callback_metrics, flush=True)
        ret_val = trainer.callback_metrics["val_loss"]
        torch.cuda.empty_cache()
        trainer.test(model, datamodule=dm)
        print(trainer.callback_metrics, flush=True)
        torch.cuda.empty_cache()
        return ret_val
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(e, flush=True)
        raise optuna.TrialPruned()

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--model", type=str, default="google/flan-t5-small")
    argparser.add_argument("--batchsize", type=int, default=80)
    argparser.add_argument("--epochs", type=int, default=None)
    argparser.add_argument("--lr", type=float, default=None)
    argparser.add_argument("--ld", type=float, default=None)
    argparser.add_argument("--trials", type=int, default=10)
    argparser.add_argument("--optunafile", type=str, default=f"gen_optuna_{datetime.now().strftime('%Y-%m-%d')}.log")
    argparser.add_argument("--studyname", type=str, default=f"Query Score {datetime.now().strftime('%Y-%m-%d')}")
    argparser.add_argument("--dataset", type=str, default="f1")

    arguments = argparser.parse_args()

    batch_size = arguments.batchsize #64 + 16
    learning_rate = arguments.lr
    ld = arguments.ld
    epochs = arguments.epochs #5
    n_trials = arguments.trials  #10
    storage = JournalStorage(JournalFileStorage(f"{arguments.optunafile}"))
    study = optuna.create_study(direction='minimize',
                                study_name=f"{arguments.studyname}",
                                storage=storage, load_if_exists=True)
    study.optimize(lambda trial: objective(trial=trial,
                                           batch_size=batch_size,
                                           epochs=int(epochs),
                                           model_name=arguments.model,
                                           dataset_type=arguments.dataset,
                                           learning_rate=float(learning_rate),
                                           ld=float(ld)),
                   n_trials=n_trials,
                   n_jobs=1)
    print(study.best_params)
