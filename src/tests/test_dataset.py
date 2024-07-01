import math

from lightning import Trainer
from transformers import BertTokenizerFast, T5Tokenizer

from dudes.qa.sparql_selection.llm_query_selector import LLMQuerySelector
from llm.qald_dataset import *
import lemon
import pandas as pd

from llm.query_score_dataset import QueryScoreDataModule
from llm.query_score_models.t5 import QueryScoreT5, QueryScoreT5Base


def test_dataset_creation():
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
    dm = QALDDataModule(tokenizer=tokenizer)
    dm.prepare_data()
    dm.setup(stage="fit")
    td = [d for d in dm.train_dataloader()]
    pass

def test_dataset_creation2():
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
    for dt in ["f1", "f1-novalid", "clampfp", "clampfp-novalid"]:
        print(dt, flush=True)
        dm = QueryScoreDataModule(tokenizer=tokenizer, dataset_type=dt)
        dm.prepare_data()
        dm.setup(stage="fit")
    #td = [d for d in dm.train_dataloader()]
    # same = [v.item() for d in td for v in d["output_values"] if abs(v.item() - 0.5) < 0.01]
    # nonsame = [v.item() for d in td for v in d["output_values"] if abs(v.item() - 0.5) >= 0.01]
    # print("Same:", len(same), "Non-same:", len(nonsame))
    #for d in td:
    #    if d["output_values"]
    pass

def test_all_queries_csv():
    path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "all-queries-test.csv.zst")
    path_out = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "all-queries-test_unique.csv.zst")
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['id', 'Generated SPARQL'])
    df = df[df['True Positive'] + df['False Positive'] > 0]
    df.to_csv(path_out, index=False)
    pass

def test_all_queries_csv2():
    path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "all-queries-train_unique3.csv.zst")
    df = pd.read_csv(path)
    pass

def test_model():
    # models = [
    #     #"query_score_llm_0.00010562737540666712_0.9818562079077777_64_5_2024-06-17_07-13-07.ckpt",
    #     #TODO{'val_loss': tensor(0.0641)} {'test_loss': tensor(0.8147), 'test_loss_epoch': tensor(0.8147)}
    #     #"query_score_llm_0.00011535062829921682_0.9247780438698548_80_5_2024-06-16_22-24-45.ckpt",
    #     #TODO{'val_loss': tensor(0.0604)} {'test_loss': tensor(0.8998), 'test_loss_epoch': tensor(0.8998)}
    #     #"query_score_llm_0.00011814405794547551_0.9557844515766813_64_5_2024-06-17_16-06-55.ckpt",
    #     #TODO{'val_loss': tensor(0.0647)} {'test_loss': tensor(0.8184), 'test_loss_epoch': tensor(0.8184)}
    #     #"query_score_llm_0.00015662119441657414_0.9647471965700974_80_10_2024-06-17_00-52-41.ckpt",
    #     #TODO{'val_loss': tensor(0.0559)} {'test_loss': tensor(1.0992), 'test_loss_epoch': tensor(1.0992)}
    #     #"query_score_llm_0.00015887151191173668_0.9025580194299236_80_10_2024-06-17_00-57-15.ckpt",
    #     #TODO{'val_loss': tensor(0.0592)} {'test_loss': tensor(1.0500), 'test_loss_epoch': tensor(1.0500)}
    #     #"query_score_llm_0.00017423904072302543_0.923393129664633_64_10_2024-06-17_14-13-53.ckpt",
    #     #TODO{'val_loss': tensor(0.0551)} {'test_loss': tensor(1.2515), 'test_loss_epoch': tensor(1.2515)}
    #     #"query_score_llm_0.00025003848134042667_0.9638188875074791_80_5_2024-06-16_22-24-55.ckpt",
    #     #TODO{'val_loss': tensor(0.0583)} {'test_loss': tensor(1.0816), 'test_loss_epoch': tensor(1.0816)}
    #     "query_score_llm_0.00027467810650825354_0.9064407760748068_28_5_2024-06-17_10-53-03.ckpt",
    #     #TODO{'val_loss': tensor(0.0560)} {'test_loss': tensor(1.4101), 'test_loss_epoch': tensor(1.4101)}
    #     #"query_score_llm_0.0003726426425378459_0.9458877416739538_64_5_2024-06-17_05-52-04.ckpt",
    #     #TODO{'val_loss': tensor(0.0592)} {'test_loss': tensor(0.9985), 'test_loss_epoch': tensor(0.9985)}
    #     "query_score_llm_0.0008564850238242047_0.9381625504397031_28_5_2024-06-17_10-50-11.ckpt",
    #     #TODO{'val_loss': tensor(0.0621)} {'test_loss': tensor(1.0995), 'test_loss_epoch': tensor(1.0995)}
    #     "query_score_llm_0.0009400619187059821_0.911971764537041_80_5_2024-06-16_22-23-30.ckpt",
    #     "query_score_llm_0.0009883521703547417_0.9129423603361809_28_5_2024-06-17_10-51-23.ckpt",
    #     "query_score_llm_1.0228270810882938e-05_0.9093453991596947_64_5_2024-06-17_07-13-48.ckpt",
    #     "query_score_llm_1.3013124885549776e-05_0.943911489002152_64_5_2024-06-17_11-44-22.ckpt",
    #     "query_score_llm_1.3803890819433259e-05_0.973940286843353_80_10_2024-06-17_00-50-34.ckpt",
    #     "query_score_llm_1.4559908418368885e-05_0.9520103660529491_80_5_2024-06-16_22-25-54.ckpt",
    #     "query_score_llm_1.5210299766219246e-05_0.9598071150918308_64_5_2024-06-17_13-07-45.ckpt",
    #     "query_score_llm_1.983126152435975e-05_0.9829173382159417_40_5_2024-06-16_23-35-31.ckpt",
    #     "query_score_llm_2.4564828752149007e-05_0.9682011839514554_64_10_2024-06-17_12-56-03.ckpt",
    #     "query_score_llm_4.1664144064597064e-05_0.9151322956490647_80_10_2024-06-17_00-51-20.ckpt",
    #     "query_score_llm_6.235288076728704e-05_0.9202516265354788_64_10_2024-06-17_12-56-38.ckpt",
    #     "query_score_llm_7.426002981560885e-05_0.9718038975302306_28_5_2024-06-17_10-47-24.ckpt",
    # ]
    models = [
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-29-16-536758.ckpt"),
        #{'val_loss': tensor(0.3758)}
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-30-21-434619.ckpt"),
        #{'val_loss': tensor(0.3812)}
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-30-49-282346.ckpt"),
        #{'val_loss': tensor(0.5294)}
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-30-59-969590.ckpt"),
        #{'val_loss': tensor(0.7222)}
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-31-30-134770.ckpt"),
        #{'val_loss': tensor(0.4879)}
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-31-54-743125.ckpt"),
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-32-25-476961.ckpt"),
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-32-30-917349.ckpt"),
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-33-02-776942.ckpt"),
        os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_21-21-50-580922.ckpt"),
        #{'val_loss': tensor(0.3828)}
    ]
    dm = QueryScoreDataModule(tokenizer=T5Tokenizer.from_pretrained("google/flan-t5-small"), batch_size=200, dataset_type="clampfp")
    #dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    # dm2 = QueryScoreDataModule(tokenizer=T5Tokenizer.from_pretrained("google/flan-t5-base"), batch_size=100)
    # dm2.prepare_data()
    # dm2.setup("fit")
    # dm2.setup("test")
    for mp in models:
        try:
            model = QueryScoreT5.load_from_checkpoint(mp)#query_score_llm_1.4559908418368885e-05_0.9520103660529491_80_5_2024-06-16_22-25-54.ckpt

            #td = [d for d in dm.test_dataloader()]

            trainer = Trainer(enable_checkpointing=True,
                              # accelerator="cpu",
                              reload_dataloaders_every_n_epochs=1,
                              log_every_n_steps=1,
                              max_epochs=1,
                              min_epochs=1,
                              strategy="ddp")
            torch.cuda.empty_cache()
            print(mp, flush=True)
            trainer.validate(model, datamodule=dm)
            print(trainer.callback_metrics, flush=True)
            # trainer.test(model, datamodule=dm)
            # print(trainer.callback_metrics, flush=True)
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(e, flush=True)


def test_query_selection():
    qs = LLMQuerySelector(model_path="/home/dvs23/projects/neodudes/resources/qald/query_score_models/query_score_llm_1.4559908418368885e-05_0.9520103660529491_80_5_2024-06-16_22-25-54.ckpt")
    question = "What is Angela Merkel’s birth name?"
    query1 = """SELECT DISTINCT ?v0
WHERE {
   <http://dbpedia.org/resource/Angela_Merkel> <http://dbpedia.org/property/birthName> ?v0 . 
} """
    dudes1 = """[x6-1 | l10]
l10: {
[x6-1, x10]
is(x6-1) == True
what(x6-1) == True
dbp:birthName(x10, x6-1) == True
x10 == "dbr:Angela_Merkel"
}
---- 
(x6-1, l10)
---- 

--------"""
    numres1 = 1

    query2 = """SELECT DISTINCT ?v0
WHERE {
   <http://dbpedia.org/resource/Angela_Merkel> <http://dbpedia.org/property/name> ?v0 . 
} """
    dudes2 = """
[x6-1 | l10]
l10: {
[x6-1, x13, x10]
is(x6-1) == True
dbp:name(x10, x6-1) == True
x13 == "dbr:Birth"
what(x6-1) == True
x10 == "dbr:Angela_Merkel"
}
---- 
(x6-1, l10)
---- 

--------
    """
    numres2 = 1


    query3 = """SELECT DISTINCT ?v4
WHERE {
   ?v4 ?v5 ?v1 . 
   ?v0 <http://dbpedia.org/ontology/child> ?v1 . 
} 
    """
    numres3 = 10000
    dudes3 = """[None | l6]
    l6: {
    [x13, x6-0, x6-1, x1, x11]
    local:with(x1, x6-1) == True
    did(x11) == True
    how_many(x6-1) == True
    dbo:child(x6-0, x6-1) == True
    x13 == "dbr:Jacques_Cousteau"
    }
    ---- 
    (x1, l6)
    (x6-0, l6, ['of', 'of'])
    (x6-1, l6)
    (x11, l6)
    ---- 
    
    --------"""
    res = qs.compare_queries(question, query1, query2, dudes1, dudes2, numres1, numres2)
    res2 = qs.compare_queries(question, query1, query3, dudes1, dudes3, numres1, numres3)
    res3 = qs.compare_queries(question, query2, query3, dudes2, dudes3, numres2, numres3)
    print(res, res2, res3)
    res = qs.compare_queries(question, query2, query1, dudes2, dudes1, numres2, numres1)
    res2 = qs.compare_queries(question, query3, query1, dudes3, dudes1, numres3, numres1)
    res3 = qs.compare_queries(question, query3, query2, dudes3, dudes2, numres3, numres2)
    print(res, res2, res3)

