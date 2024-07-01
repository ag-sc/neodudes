from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional, Dict, Tuple

import torch

from dudes import consts
from dudes.dudes import DUDES
from dudes.qa.dudes_rpc_service import LLMQuerySelectorWrapper
from dudes.utils import EvalStats

from torch.nn import functional as F


class PEvalStrategy(ABC):
    gold: Any
    best_stats: EvalStats
    best_query: str
    best_dudes: Optional[DUDES]
    best_full_query: str
    best_changed: int

    def __init__(self, gold):
        self.gold = gold
        self.best_changed = 1
    @abstractmethod
    def eval(self, curr_stats, query, dudes, full_query):#, sys_res, gold_res
        pass

    @property
    def strategy_name(self):
        return self.__class__.__name__


class EvalStrategy(PEvalStrategy):
    gold: Any
    best_stats: EvalStats
    best_query: str
    best_dudes: Optional[DUDES]
    best_full_query: str
    best_changed: int

    def __init__(self, gold):
        #if self.gold.fset is not None:
        super().__init__(gold)
        self.best_stats = EvalStats(fn=len(gold))
        self.best_query = "No valid result yet."
        self.best_dudes = None
        self.best_full_query = "No valid result yet."
        self.best_changed = 1
    @abstractmethod
    def eval(self, curr_stats, query, dudes, full_query):#, sys_res, gold_res
        pass


class BestScoreEval(EvalStrategy):

    def __init__(self, gold):
        super().__init__(gold)

    def _best_score_eval(self, curr_stats):
        return ((curr_stats > self.best_stats or
                 (curr_stats >= self.best_stats and self.best_dudes is None))
                and curr_stats.tp + curr_stats.fp <= 4267 + 0.1 * 4267  # TODO: derive value from training data automatically
                and (curr_stats.fp + 0.001) / (curr_stats.tp + 0.001) <= consts.fp_ratio)
    def eval(self, curr_stats, query, dudes, full_query):
        if self._best_score_eval(curr_stats):
            self.best_stats = curr_stats
            self.best_query = query
            self.best_dudes = dudes
            self.best_full_query = full_query
            self.best_changed += 1


class LLMEval(EvalStrategy):

    def __init__(self, gold, question, model_id=None, rpc_conn=None, query_scorer=None):
        super().__init__(gold)
        self.question = question
        if query_scorer is None:
            self.query_scorer = LLMQuerySelectorWrapper(conn=rpc_conn)
        else:
            self.query_scorer = query_scorer
        self.best_stats = EvalStats(fn=len(gold))
        self.best_query = "No valid result yet."
        self.best_dudes = None
        self.best_full_query = "No valid result yet."
        self.model_id = model_id

    def _basic_filter(self, curr_stats):
        return (4267 + 0.1 * 4267 >= curr_stats.tp + curr_stats.fp > 0)

    def eval(self, curr_stats, query, dudes, full_query):
        if self._basic_filter(curr_stats):
            res = self.query_scorer.compare_queries(
                question=self.question,
                query1=self.best_query,
                query2=query,
                dudes1=self.best_dudes,
                dudes2=dudes,
                numresults1=self.best_stats.tp + self.best_stats.fp,
                numresults2=curr_stats.tp + curr_stats.fp
            )
            if self.model_id is not None:
                best_val = res[self.model_id][0]
                curr_val = res[self.model_id][1]
            else:
                best_val = sum([r[0] for r in res])
                curr_val = sum([r[1] for r in res])
            res2 = self.query_scorer.compare_queries(
                question=self.question,
                query2=self.best_query,
                query1=query,
                dudes2=self.best_dudes,
                dudes1=dudes,
                numresults2=self.best_stats.tp + self.best_stats.fp,
                numresults1=curr_stats.tp + curr_stats.fp
            )
            if self.model_id is not None:
                best_val += res[self.model_id][0]
                curr_val += res[self.model_id][1]
            else:
                best_val += sum([r[0] for r in res])
                curr_val += sum([r[1] for r in res])

            if curr_val > best_val:
                self.best_stats = curr_stats
                self.best_query = query
                self.best_dudes = dudes
                self.best_full_query = full_query
                self.best_changed += 1

    @property
    def strategy_name(self):
        return self.__class__.__name__+"_"+str(self.model_id)


class LLMMostWinsEval(PEvalStrategy):

    _query_win_counter: Dict[str, int]
    _query_data: Dict[str, Tuple]

    def __init__(self, gold, question, model_id=None, win_threshold=0.0, rpc_conn=None, query_scorer=None):
        super().__init__(gold)
        self.question = question
        self.model_id = model_id
        self.win_threshold = win_threshold
        if query_scorer is None:
            self.query_scorer = LLMQuerySelectorWrapper(conn=rpc_conn)
        else:
            self.query_scorer = query_scorer
        self._query_win_counter = defaultdict(int)
        self._query_data = dict()
        self._query_win_counter["No valid result yet."] = 0
        self._query_data["No valid result yet."] = (EvalStats(fn=len(gold)), str(None), "No valid result yet.")

    def _basic_filter(self, curr_stats):
        return (4267 + 0.1 * 4267 >= curr_stats.tp + curr_stats.fp > 0)

    def _most_wins_query(self):
        max_num = max(self._query_win_counter.values(), default=None)
        best_res = [query for query, count in self._query_win_counter.items() if count == max_num]
        if len(best_res) == 0:
            return None
        return best_res[0]

    def _update_wins(self, curr_stats, query, dudes):
        for old_query, d in self._query_data.items():
            old_stats, old_dudes, old_full_query = d
            res = self.query_scorer.compare_queries(
                question=self.question,
                query1=old_query,
                query2=query,
                dudes1=old_dudes,
                dudes2=dudes,
                numresults1=old_stats.tp + old_stats.fp,
                numresults2=curr_stats.tp + curr_stats.fp
            )
            if self.model_id is not None:
                old_val = res[self.model_id][0]
                curr_val = res[self.model_id][1]
                diff_frac = abs(curr_val - old_val) / abs(curr_val + old_val + 1e-6)
                if diff_frac > self.win_threshold:
                    if curr_val > old_val:
                        self._query_win_counter[query] += 1
                    elif old_val > curr_val:
                        self._query_win_counter[old_query] += 1
            else:
                for r in res:
                    old_val = r[0]
                    curr_val = r[1]
                    diff_frac = abs(curr_val - old_val) / abs(curr_val + old_val + 1e-6)
                    if diff_frac > self.win_threshold:
                        if curr_val > old_val:
                            self._query_win_counter[query] += 1
                        elif old_val > curr_val:
                            self._query_win_counter[old_query] += 1

            # old_val = res[0]
            # curr_val = res[1]
            # if curr_val > old_val:
            #     self._query_win_counter[query] += 1
            # elif old_val > curr_val:
            #     self._query_win_counter[old_query] += 1

            res = self.query_scorer.compare_queries(
                question=self.question,
                query2=old_query,
                query1=query,
                dudes2=old_dudes,
                dudes1=dudes,
                numresults2=old_stats.tp + old_stats.fp,
                numresults1=curr_stats.tp + curr_stats.fp
            )

            if self.model_id is not None:
                old_val = res[self.model_id][1]
                curr_val = res[self.model_id][0]
                diff_frac = abs(curr_val - old_val) / abs(curr_val + old_val + 1e-6)
                if diff_frac > self.win_threshold:
                    if curr_val > old_val:
                        self._query_win_counter[query] += 1
                    elif old_val > curr_val:
                        self._query_win_counter[old_query] += 1
            else:
                for r in res:
                    old_val = r[1]
                    curr_val = r[0]
                    diff_frac = abs(curr_val - old_val) / abs(curr_val + old_val + 1e-6)
                    if diff_frac > self.win_threshold:
                        if curr_val > old_val:
                            self._query_win_counter[query] += 1
                        elif old_val > curr_val:
                            self._query_win_counter[old_query] += 1
            # old_val = res[1]
            # curr_val = res[0]
            # if curr_val > old_val:
            #     self._query_win_counter[query] += 1
            # elif old_val > curr_val:
            #     self._query_win_counter[old_query] += 1
    def eval(self, curr_stats, query, dudes, full_query):
        if self._basic_filter(curr_stats):
            old_q = self._most_wins_query()
            #self._query_win_counter[query] += 1
            self._update_wins(curr_stats, query, str(dudes))

            if query not in self._query_data:
                self._query_data[query] = (curr_stats, str(dudes), full_query)

            new_q = self._most_wins_query()
            if old_q != new_q:
                self.best_changed += 1
    @property
    def best_query(self):
        mq = self._most_wins_query()
        return mq if mq is not None else "No valid result yet."

    @property
    def best_stats(self):
        mq = self._most_wins_query()
        return self._query_data[mq][0] if mq is not None else EvalStats(fn=len(self.gold))

    @property
    def best_dudes(self):
        mq = self._most_wins_query()
        return self._query_data[mq][1] if mq is not None else None

    @property
    def best_full_query(self):
        mq = self._most_wins_query()
        return self._query_data[mq][2] if mq is not None else "No valid result yet."

    @property
    def strategy_name(self):
        return self.__class__.__name__ + "_" + str(self.model_id) + "_" + str(round(self.win_threshold, 2))


class LLMAccumEval(PEvalStrategy):

    _query_logit_sum: Dict[str, int]
    _query_data: Dict[str, Tuple]

    def __init__(self, gold, question, model_id=None, use_sigmoid=True, rpc_conn=None, query_scorer=None):
        super().__init__(gold)
        self.question = question
        self.model_id = model_id
        self.use_sigmoid = use_sigmoid
        if query_scorer is None:
            self.query_scorer = LLMQuerySelectorWrapper(conn=rpc_conn)
        else:
            self.query_scorer = query_scorer
        self._query_logit_sum = defaultdict(int)
        self._query_data = dict()
        self._query_logit_sum["No valid result yet."] = 0
        self._query_data["No valid result yet."] = (EvalStats(fn=len(gold)), str(None), "No valid result yet.")

    def _basic_filter(self, curr_stats):
        return (4267 + 0.1 * 4267 >= curr_stats.tp + curr_stats.fp > 0)

    def _most_wins_query(self):
        max_num = max(self._query_logit_sum.values(), default=None)
        best_res = [query for query, count in self._query_logit_sum.items() if count == max_num]
        if len(best_res) == 0:
            return None
        return best_res[0]

    def _update_wins(self, curr_stats, query, dudes):
        for old_query, d in self._query_data.items():
            old_stats, old_dudes, old_full_query = d
            res = self.query_scorer.compare_queries(
                question=self.question,
                query1=old_query,
                query2=query,
                dudes1=old_dudes,
                dudes2=dudes,
                numresults1=old_stats.tp + old_stats.fp,
                numresults2=curr_stats.tp + curr_stats.fp
            )
            if self.model_id is not None:
                old_val = res[self.model_id][0]
                curr_val = res[self.model_id][1]
                self._query_logit_sum[query] += F.sigmoid(torch.Tensor([curr_val])).item() if self.use_sigmoid else curr_val
                self._query_logit_sum[old_query] += F.sigmoid(torch.Tensor([old_val])).item() if self.use_sigmoid else old_val
            else:
                for r in res:
                    old_val = r[0]
                    curr_val = r[1]
                    self._query_logit_sum[query] += F.sigmoid(torch.Tensor([curr_val])).item() if self.use_sigmoid else curr_val
                    self._query_logit_sum[old_query] += F.sigmoid(torch.Tensor([old_val])).item() if self.use_sigmoid else old_val

            # old_val = res[0]
            # curr_val = res[1]
            # if curr_val > old_val:
            #     self._query_win_counter[query] += 1
            # elif old_val > curr_val:
            #     self._query_win_counter[old_query] += 1

            res = self.query_scorer.compare_queries(
                question=self.question,
                query2=old_query,
                query1=query,
                dudes2=old_dudes,
                dudes1=dudes,
                numresults2=old_stats.tp + old_stats.fp,
                numresults1=curr_stats.tp + curr_stats.fp
            )

            if self.model_id is not None:
                old_val = res[self.model_id][1]
                curr_val = res[self.model_id][0]
                self._query_logit_sum[query] += F.sigmoid(torch.Tensor([curr_val])).item() if self.use_sigmoid else curr_val
                self._query_logit_sum[old_query] += F.sigmoid(torch.Tensor([old_val])).item() if self.use_sigmoid else old_val
            else:
                for r in res:
                    old_val = r[1]
                    curr_val = r[0]
                    self._query_logit_sum[query] += F.sigmoid(torch.Tensor([curr_val])).item() if self.use_sigmoid else curr_val
                    self._query_logit_sum[old_query] += F.sigmoid(torch.Tensor([old_val])).item() if self.use_sigmoid else old_val
            # old_val = res[1]
            # curr_val = res[0]
            # if curr_val > old_val:
            #     self._query_win_counter[query] += 1
            # elif old_val > curr_val:
            #     self._query_win_counter[old_query] += 1
    def eval(self, curr_stats, query, dudes, full_query):
        if self._basic_filter(curr_stats):
            old_q = self._most_wins_query()
            #self._query_win_counter[query] += 1
            self._update_wins(curr_stats, query, str(dudes))

            if query not in self._query_data:
                self._query_data[query] = (curr_stats, str(dudes), full_query)

            new_q = self._most_wins_query()
            if old_q != new_q:
                self.best_changed += 1
    @property
    def best_query(self):
        mq = self._most_wins_query()
        return mq if mq is not None else "No valid result yet."

    @property
    def best_stats(self):
        mq = self._most_wins_query()
        return self._query_data[mq][0] if mq is not None else EvalStats(fn=len(self.gold))

    @property
    def best_dudes(self):
        mq = self._most_wins_query()
        return self._query_data[mq][1] if mq is not None else None

    @property
    def best_full_query(self):
        mq = self._most_wins_query()
        return self._query_data[mq][2] if mq is not None else "No valid result yet."

    @property
    def strategy_name(self):
        return self.__class__.__name__ + "_" + str(self.model_id) + "_" + ("sigmoid" if self.use_sigmoid else "logits")


class LowResultsNonzeroEval(EvalStrategy):

    def _low_results_nonzero_eval(self, curr_stats):
        return ((curr_stats.tp + curr_stats.fp < self.best_stats.tp + self.best_stats.fp or self.best_dudes is None)
                and 4267 + 0.1 * 4267 >= curr_stats.tp + curr_stats.fp > 0)

    def eval(self, curr_stats, query, dudes, full_query):
        if self._low_results_nonzero_eval(curr_stats):
            self.best_stats = curr_stats
            self.best_query = query
            self.best_dudes = dudes
            self.best_full_query = full_query
            self.best_changed += 1


class FrequencyEval(PEvalStrategy):

    _query_counter: Dict[str, int]
    _query_data: Dict[str, Tuple]

    def __init__(self, gold):
        super().__init__(gold)
        self._query_counter = defaultdict(int)
        self._query_data = dict()

    def _basic_filter(self, curr_stats):
        return (4267 + 0.1 * 4267 >= curr_stats.tp + curr_stats.fp > 0)

    def _most_frequent_query(self):
        max_num = max(self._query_counter.values(), default=None)
        best_res = [query for query, count in self._query_counter.items() if count == max_num]
        if len(best_res) == 0:
            return None
        return best_res[0]
    def eval(self, curr_stats, query, dudes, full_query):
        if self._basic_filter(curr_stats):
            old_q = self._most_frequent_query()
            self._query_counter[query] += 1

            if query not in self._query_data:
                self._query_data[query] = (curr_stats, dudes, full_query)

            new_q = self._most_frequent_query()
            if old_q != new_q:
                self.best_changed += 1
    @property
    def best_query(self):
        mq = self._most_frequent_query()
        return mq if mq is not None else "No valid result yet."

    @property
    def best_stats(self):
        mq = self._most_frequent_query()
        return self._query_data[mq][0] if mq is not None else EvalStats(fn=len(self.gold))

    @property
    def best_dudes(self):
        mq = self._most_frequent_query()
        return self._query_data[mq][1] if mq is not None else None

    @property
    def best_full_query(self):
        mq = self._most_frequent_query()
        return self._query_data[mq][2] if mq is not None else "No valid result yet."


class FrequencyLowResultNonzeroEval(PEvalStrategy):

    _query_counter: Dict[str, int]
    _query_data: Dict[str, Tuple]

    def __init__(self, gold):
        super().__init__(gold)
        self._query_counter = defaultdict(int)
        self._query_data = dict()

    def _basic_filter(self, curr_stats):
        return (4267 + 0.1 * 4267 >= curr_stats.tp + curr_stats.fp > 0)

    def _most_frequent_query(self):
        max_num = max(self._query_counter.values(), default=None)
        best_res = [query for query, count in self._query_counter.items() if count == max_num]
        if len(best_res) == 0:
            return None
        return best_res[0]
    def eval(self, curr_stats, query, dudes, full_query):
        if self._basic_filter(curr_stats):
            if curr_stats.tp + curr_stats.fp < self.best_stats.tp + self.best_stats.fp:
                #Reset as lower total number has been found
                self._query_counter: Dict[str, int] = defaultdict(int)
                self._query_data: Dict[str, Tuple] = dict()
                self._query_counter[query] += 1
                self._query_data[query] = (curr_stats, dudes, full_query)
            elif curr_stats.tp + curr_stats.fp == self.best_stats.tp + self.best_stats.fp or self.best_dudes is None:
                old_q = self._most_frequent_query()
                self._query_counter[query] += 1

                if query not in self._query_data:
                    self._query_data[query] = (curr_stats, dudes, full_query)

                new_q = self._most_frequent_query()
                if old_q != new_q:
                    self.best_changed += 1
    @property
    def best_query(self):
        mq = self._most_frequent_query()
        return mq if mq is not None else "No valid result yet."

    @property
    def best_stats(self):
        mq = self._most_frequent_query()
        return self._query_data[mq][0] if mq is not None else EvalStats(fn=len(self.gold))

    @property
    def best_dudes(self):
        mq = self._most_frequent_query()
        return self._query_data[mq][1] if mq is not None else None

    @property
    def best_full_query(self):
        mq = self._most_frequent_query()
        return self._query_data[mq][2] if mq is not None else "No valid result yet."
