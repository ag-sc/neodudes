import os.path
import sys
from datetime import datetime
from typing import Optional

import optuna
from lightning import Trainer
from optuna import Trial
from optuna.storages import JournalStorage, JournalFileStorage

from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from llm.qald_models.t5 import QaldT5
from llm.qald_dataset import QALDDataModule

def objective(trial: Trial,
              batch_size=None,
              learning_rate=None,
              ld=None,
              epochs=50):
    if batch_size is None:
        batch_size = 1
        # trial.suggest_int('batch_size', 1, 10)
    if learning_rate is None:
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    if ld is None:
        ld = trial.suggest_float('ld', 0.9, 1.0, log=True)

    se: Optional[SPARQLEndpoint] = None
    try:
        se = SPARQLEndpoint()
    except Exception as e:
        print(e)
    model = QaldT5(learning_rate=learning_rate, ld=ld, sparql_endpoint=se)
    dm = QALDDataModule(tokenizer=model.tokenizer, batch_size=batch_size)

    trainer = Trainer(enable_checkpointing=False,
                      reload_dataloaders_every_n_epochs=1,
                      log_every_n_steps=1,
                      max_epochs=epochs,
                      min_epochs=epochs,
                      strategy="ddp")
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(f"qald_llm_{learning_rate}_{ld}_{batch_size}_{epochs}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.ckpt")
    print(trainer.callback_metrics, flush=True)
    ret_val = trainer.callback_metrics["val_loss"]
    trainer.test(model, datamodule=dm)
    print(trainer.callback_metrics, flush=True)
    return ret_val

def eval_model():
    se = SPARQLEndpoint()
    model = QaldT5.load_from_checkpoint(os.path.join(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "..", "..", "qald_llm_2.8720744151668305e-05_0.9899229588262145_24_50_2024-05-02_18-30-58.ckpt")))
    model.sparql_endpoint = se
    dm = QALDDataModule(tokenizer=model.tokenizer, batch_size=1)
    trainer = Trainer(enable_checkpointing=False,
                      reload_dataloaders_every_n_epochs=1,
                      log_every_n_steps=1,
                      max_epochs=50,
                      min_epochs=50,
                      strategy="ddp")
    trainer.test(model, datamodule=dm)
    print(trainer.callback_metrics)

if __name__ == '__main__':
    eval_model()
    exit(0)


    batch_size = 24
    learning_rate = 1e-5
    ld = 0.9
    epochs = 50
    n_trials = 30
    storage = JournalStorage(JournalFileStorage(f"gen_optuna_{datetime.now().strftime('%Y-%m-%d')}.log"))
    study = optuna.create_study(direction='minimize',
                                study_name=f"QALD LLM {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                                storage=storage, load_if_exists=True)
    study.optimize(lambda trial: objective(trial=trial,
                                           batch_size=batch_size,
                                           epochs=epochs),
                                           #learning_rate=learning_rate,
                                           #ld=ld),
                   n_trials=n_trials,
                   n_jobs=1)
    print(study.best_params)
