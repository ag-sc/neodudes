import traceback
from typing import Iterable, List

import rpyc

from dudes import consts
from dudes.qa.sparql_selection.query_evaluation_strategies import PEvalStrategy, BestScoreEval, LowResultsNonzeroEval, \
    FrequencyEval, FrequencyLowResultNonzeroEval, LLMEval, LLMMostWinsEval, LLMAccumEval


class QueryEvaluator:

    def __init__(self, strategies: Iterable[PEvalStrategy]):
        self.strategies = list(strategies)

    @classmethod
    def default(cls, gold, question, rpc_conn=None, query_scorer=None):
        if rpc_conn is None and query_scorer is None:
            rpc_conn = rpyc.connect(consts.rpc_host,
                                    consts.rpc_port,
                                    config={
                                        "allow_public_attrs": True,
                                        "allow_pickle": True,
                                        "sync_request_timeout": 300
                                    })

        #strategies: List = sum([
        #    [LLMMostWinsEval(gold, question, model_id=i, win_threshold=0.0, rpc_conn=rpc_conn, query_scorer=query_scorer),
        #    LLMMostWinsEval(gold, question, model_id=i, win_threshold=0.1, rpc_conn=rpc_conn, query_scorer=query_scorer),
        #    LLMMostWinsEval(gold, question, model_id=i, win_threshold=0.25, rpc_conn=rpc_conn, query_scorer=query_scorer),
        #    LLMMostWinsEval(gold, question, model_id=i, win_threshold=0.5, rpc_conn=rpc_conn, query_scorer=query_scorer),
        #    LLMMostWinsEval(gold, question, model_id=i, win_threshold=0.75, rpc_conn=rpc_conn, query_scorer=query_scorer),
        #    LLMMostWinsEval(gold, question, model_id=i, win_threshold=0.9, rpc_conn=rpc_conn, query_scorer=query_scorer),
        #    LLMAccumEval(gold, question, model_id=i, use_sigmoid=True, rpc_conn=rpc_conn, query_scorer=query_scorer),
        #    LLMAccumEval(gold, question, model_id=i, use_sigmoid=False, rpc_conn=rpc_conn, query_scorer=query_scorer)]
        #    for i in range(10)
        #], [])
        strategies = [
            #LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.0, rpc_conn=rpc_conn, query_scorer=query_scorer),
            #LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.1, rpc_conn=rpc_conn, query_scorer=query_scorer),
            #LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.25, rpc_conn=rpc_conn, query_scorer=query_scorer),
            #LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.5, rpc_conn=rpc_conn, query_scorer=query_scorer),
            LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.75, rpc_conn=rpc_conn, query_scorer=query_scorer),
            LLMMostWinsEval(gold, question, model_id=None, win_threshold=0.9, rpc_conn=rpc_conn, query_scorer=query_scorer),
            #LLMAccumEval(gold, question, model_id=None, use_sigmoid=True, rpc_conn=rpc_conn, query_scorer=query_scorer),
            #LLMAccumEval(gold, question, model_id=None, use_sigmoid=False, rpc_conn=rpc_conn, query_scorer=query_scorer)
        ]
        strategies = [BestScoreEval(gold)] + strategies
        return cls(strategies)

    def eval(self, curr_stats, query, dudes, full_query):
        for strategy in self.strategies:
            try:
                strategy.eval(curr_stats, query, dudes, full_query)
            except Exception as e:
                print(f"Error evaluating {strategy.strategy_name}: {e}")
                print(traceback.format_exc())
                continue
    @property
    def best_stats(self):
        return {strategy.strategy_name: strategy.best_stats for strategy in self.strategies}

    @property
    def best_query(self):
        return {strategy.strategy_name: strategy.best_query for strategy in self.strategies}

    @property
    def best_dudes(self):
        return {strategy.strategy_name: strategy.best_dudes for strategy in self.strategies}

    @property
    def best_full_query(self):
        return {strategy.strategy_name: strategy.best_full_query for strategy in self.strategies}

    @property
    def best_changed(self):
        return {strategy.strategy_name: strategy.best_changed for strategy in self.strategies}

    def __str__(self):
        return str({
            "best_stats": self.best_stats,
            "best_query": self.best_query,
            "best_dudes": self.best_dudes,
            "best_full_query": self.best_full_query,
            "best_changed": self.best_changed
        })
