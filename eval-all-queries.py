import csv
import itertools
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict

import pandas as pd
import rpyc
import lemon

from dudes import utils, consts
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from dudes.qa.sparql_selection.llm_query_selector import MultiLLMQuerySelector
from dudes.qa.sparql_selection.query_evaluation_strategies import LLMMostWinsEval, LLMAccumEval, BestScoreEval
from dudes.qa.sparql_selection.query_evaluator import QueryEvaluator
from dudes.utils import EvalStats

fieldnames = ['id', 'question', "Strategy", "Gold SPARQL", "Generated SPARQL", "Generated SPARQL Full", "True Positive",
              "False Positive", "False Negative", "Precision", "Recall", "F1", "Exact matches", "Runtime",
              "Combinations", "DUDES"]

nstats_results: Dict[str, Dict[str, EvalStats]] = defaultdict(dict)
gold_res = dict()
query_evals = dict()

def print_eval():
    for rid, qe in query_evals.items():
        for strat, stats in qe.best_stats.items():
            nstats_results[strat][rid] = stats

    if len(query_evals) > 0:
        for strat in nstats_results.keys():
            print(f"Intermediate eval:", strat,
                  utils.prettier_print(sum(nstats_results[strat].values(), EvalStats()).to_dict()),
                  "Macro:", utils.prettier_print(utils.macro_stats(nstats_results[strat].values())),
                  "Processed:", len(nstats_results[strat]),
                  flush=True)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--path", type=str, required=True)

    arguments = argparser.parse_args()

    df = pd.read_csv(arguments.path)
    df = df.reset_index()

    se = SPARQLEndpoint()

    # rpc_conn = rpyc.connect(consts.rpc_host,
    #                         consts.rpc_port,
    #                         config={
    #                             "allow_public_attrs": True,
    #                             "allow_pickle": True,
    #                             "sync_request_timeout": 300
    #                         })

    query_score_model_path = [
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-29-16-536758.ckpt"),
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-30-21-434619.ckpt"),
        os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-30-49-282346.ckpt"),
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-30-59-969590.ckpt"),
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-31-30-134770.ckpt"),
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-31-54-743125.ckpt"),
        os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-32-25-476961.ckpt"),
        os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-32-30-917349.ckpt"),
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-33-02-776942.ckpt"),
        #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_21-21-50-580922.ckpt"),
    ]
    query_selector = MultiLLMQuerySelector.from_paths(query_score_model_path)



    #for index, row in df.iterrows():
    index = 0
    for row_idx in utils.roundrobin(*df.groupby(["id"]).groups.values()):
        index += 1
        row = df.loc[row_idx]
        print(index, row["id"], row["question"], flush=True)
        if not row["id"] in gold_res:
            raw_gold = None
            try:
                raw_gold = se.get_results_query(row["Gold SPARQL"])
                gold = utils.sanitize_sparql_result(raw_gold)
                gold_res[row["id"]] = gold
            except Exception as e:
                print("Error for gold of ID:", row["id"], "Question:", row["question"], e)

        if not row["id"] in query_evals:
            gold = gold_res[row["id"]]
            question = row["question"]
            rpc_conn = None,
            query_scorer=query_selector

            strategies = [
                BestScoreEval(gold),
                #LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.0, rpc_conn=rpc_conn,
                #                query_scorer=query_scorer),
                #LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.1, rpc_conn=rpc_conn,
                #                query_scorer=query_scorer),
                #LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.25, rpc_conn=rpc_conn,
                #                query_scorer=query_scorer),
                LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.5, rpc_conn=rpc_conn,
                                query_scorer=query_scorer),
                LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.75, rpc_conn=rpc_conn,
                                query_scorer=query_scorer),
                LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.9, rpc_conn=rpc_conn,
                                query_scorer=query_scorer),
                #LLMAccumEval(gold, question, model_id=None, use_sigmoid=True, rpc_conn=rpc_conn,
                #             query_scorer=query_scorer),
                #LLMAccumEval(gold, question, model_id=None, use_sigmoid=False, rpc_conn=rpc_conn,
                #             query_scorer=query_scorer)
            ]
            query_evals[row["id"]] = QueryEvaluator(strategies)

            # query_evals[row["id"]] = QueryEvaluator.default(gold=gold_res[row["id"]],
            #                                                 question=row["question"],
            #                                                 rpc_conn=None,
            #                                                 query_scorer=query_selector)

        query_evals[row["id"]].eval(
            curr_stats=EvalStats(tp=row["True Positive"], fp=row["False Positive"], fn=row["False Negative"]),
            query=row["Generated SPARQL"],
            dudes=row["DUDES"],
            full_query=row["Generated SPARQL Full"]
        )
        print_eval()


    pass
