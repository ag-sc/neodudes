# z3.z3util.get_vars(expr)
# z3.z3util.is_const(expr)
# https://github.com/Z3Prover/z3/blob/master/src/api/python/z3/z3util.py
import itertools
import pickle
import re
import sqlite3
import statistics
import zlib
from dataclasses import dataclass
from queue import Queue
from typing import Iterable, Set, Optional, Tuple, Dict
from urllib.parse import urlparse, quote_plus

import more_itertools
import xxhash
import zstandard
from rdflib import Graph
from rdflib.namespace import NamespaceManager, Namespace
from treelib import Node
from Levenshtein import distance as levenshtein_distance

from dudes import consts
from dudes.dudes_token import DUDESToken
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from lemon.namespaces import default_namespaces


# Make list with non-hashable items only contain distinct values
def make_distinct(items: Iterable) -> list:
    res: list = []
    for i in items:
        if i not in res:
            res.append(i)
    return res


def make_distinct_ordered(items: Iterable) -> list:
    return list(dict.fromkeys(items))

def sanitize_url(url: str):
    try:
        parsed_url = urlparse(url)

        sanitized_url = parsed_url.scheme + "://" + parsed_url.netloc

        path_parts = parsed_url.path.split("/")

        for idx in range(len(path_parts)):
            path_parts[idx] = quote_plus(path_parts[idx])

        sanitized_url += "/".join(path_parts)
        if len(parsed_url.query) > 0:
            sanitized_url += "?" + parsed_url.query
        if len(parsed_url.fragment) > 0:
            sanitized_url += "#" + parsed_url.fragment
        return sanitized_url

        # url = url.replace("%", "%25")
        # slash_index = [i for i, n in enumerate(url) if n == '/']
        # if len(slash_index) < 3:
        #     if len(url) >= 7 and "http://" == url[:7].lower():
        #         url = url[:7] + url[7:].replace(":", "%3A")
        #     elif len(url) >= 8 and "https://" == url[:8].lower():
        #         url = url[:8] + url[8:].replace(":", "%3A")
        # else:
        #     url = url[:slash_index[2]] + url[slash_index[2]:].replace(":", "%3A")
        #
        # # https://docs.microfocus.com/OMi/10.62/Content/OMi/ExtGuide/ExtApps/URL_encoding.htm
        #
        # url = url.replace("'", "%27")
        # url = url.replace(",", "%2C")
        # url = url.replace(";", "%3B")
        # url = url.replace(" ", "%20")
        # url = url.replace("<", "%3C")
        # url = url.replace(">", "%3E")
        # url = url.replace("+", "%2B")
        # url = url.replace("{", "%7B")
        # url = url.replace("}", "%7D")
        # url = url.replace("|", "%7C")
        # url = url.replace("\\", "%5C")
        # url = url.replace("^", "%5E")
        # url = url.replace("~", "%7E")
        # url = url.replace("[", "%5B")
        # url = url.replace("]", "%5D")
        # url = url.replace("?", "%3F")
        # url = url.replace("@", "%40")
        # url = url.replace("=", "%3D")
        # url = url.replace("&", "%26")
        # url = url.replace("$", "%24")
        # url = url.replace("!", "%21")
        #
        # # .replace(".", "%2E") -> does not really work, just nothing found then
        #
        # return url
    except Exception as e:
        print("Error sanitizing URL:", e, url)
        return url


def sqlitedict_encode(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))


def sqlitedict_decode(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))


def sqlitedict_encode_zstd(obj):
    cctx = zstandard.ZstdCompressor()
    return sqlite3.Binary(cctx.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))


def sqlitedict_decode_zstd(obj):
    dctx = zstandard.ZstdDecompressor()
    return pickle.loads(dctx.decompress(bytes(obj)))

def any_in_list(strs: Iterable[str], lst: Iterable[str], lower=False) -> bool:
    return any([(s.lower() if lower else s) in lst for s in strs])

def unique_based_on_str(items: Iterable) -> list:
    return list({str(dds): dds for dds in items}.values())
def levenshtein_sim_normalized(string1, string2):
    return 1 - levenshtein_distance(string1, string2) / max(len(string1), len(string2), 1) # avoid division by 0

def show_tree(tree):
    tree.show(line_type="ascii-em", data_property="desc")

def hash_str(input: str, xh=None) -> int:
    if xh is None:
        xh = xxhash.xxh32()
    else:
        xh.reset()
    xh.update(input)  # .encode()
    return xh.intdigest()

def is_comp_phrase(token: DUDESToken):
    return (any_in_list(token.text_, [v + " than" for v in consts.comp_gt_keywords + consts.comp_lt_keywords], lower=True)
            or any_in_list(token.text_, consts.comp_gt_keywords_no_than + consts.comp_lt_keywords_no_than, lower=True))


def is_comp_token(token: DUDESToken):
    return token.text.lower() in consts.comp_keywords + ["than"]


def rem_quotes(val: str, all_border_quotes: bool = False):
    if all_border_quotes:
        return re.sub(r'^["\']+(.*)["\']+$', r'\1', val)
    else:
        return re.sub(r'^["\'](.*)["\']$', r'\1', val)


def tree_map(func, tree):
    q: Queue[Node] = Queue()
    q.put(tree.get_node(tree.root))

    res = []

    while not q.empty():
        node: Node
        node = q.get()

        # if len(node.data.token.candidate_uris) == 0:# and len(node.data.lex_candidates) == 0:
        # if self.exec_condition(node, tree):
        ret_val = func(node, tree)
        res.append(ret_val)

        for c in tree.children(node.identifier):
            q.put(c)

    return res
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


@dataclass
class EvalStats:
    """Class for storing evaluation statistics"""
    tp: int = 0
    """True positives"""
    fp: int = 0
    """False positives"""
    fn: int = 0
    """False negatives"""

    emc: int = 1 if fp == 0 and fn == 0 and tp > 0 else 0
    """Exact match count"""

    # @property
    # def exact_match(self):
    #     return self.emc
    #
    # @property
    # def emc(self):
    #     """Exact match count"""
    #     return 1 if self.fp == 0 and self.fn == 0 and self.tp > 0 else 0

    @property
    def f1(self) -> Optional[float]:
        """F1 score"""
        return (2 * self.precision * self.recall) / (self.precision + self.recall) if self.precision is not None and self.recall is not None and (self.precision + self.recall) > 0 else None

    @property
    def precision(self) -> Optional[float]:
        return self.prec

    @property
    def prec(self) -> Optional[float]:
        """Precision"""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else None

    @property
    def recall(self) -> Optional[float]:
        return self.rec

    @property
    def rec(self) -> Optional[float]:
        """Recall"""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else None

    def to_dict(self):
        return {
            "True Positives": self.tp,
            "False Positives": self.fp,
            "False Negatives": self.fn,
            "Exact matches": self.emc,
            "F1": self.f1,
            "Precision": self.precision,
            "Recall": self.recall
        }

    @staticmethod
    def comp_val(obj):
        f1 = 0.0 if obj.f1 is None else obj.f1
        false_rate = (obj.fp + obj.fn) / (obj.tp + obj.fp + obj.fn) if (obj.tp + obj.fp + obj.fn) > 0 else 0.0
        return (2.0*f1 + (1.0 - false_rate)) / 3.0

    def __gt__(self, other):
        return self.comp_val(self) > self.comp_val(other)

    def __lt__(self, other):
        return self.comp_val(self) < self.comp_val(other)

    def __ge__(self, other):
        return self.comp_val(self) >= self.comp_val(other)

    def __le__(self, other):
        return self.comp_val(self) <= self.comp_val(other)

    def __add__(self, other):
        return EvalStats(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn, emc=self.emc + other.emc)

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

def macro_stats(stats: Iterable[EvalStats], full=False) -> Dict:
    if full:
        return {
            "True Positives": statistics.mean([s.tp for s in stats]),
            "False Positives": statistics.mean([s.fp for s in stats]),
            "False Negatives": statistics.mean([s.fn for s in stats]),
            "Exact matches": statistics.mean([s.emc for s in stats]),
            "F1": statistics.mean([s.f1 if s.f1 is not None else 0.0 for s in stats]),
            "Precision": statistics.mean([s.precision if s.precision is not None else 0.0 for s in stats]),
            "Recall": statistics.mean([s.recall if s.recall is not None else 0.0 for s in stats])
        }
    else:
        return {
            "F1": statistics.mean([s.f1 if s.f1 is not None else 0.0 for s in stats]),
            "Precision": statistics.mean([s.precision if s.precision is not None else 0.0 for s in stats]),
            "Recall": statistics.mean([s.recall if s.recall is not None else 0.0 for s in stats])
        }

def sanitize_sparql_result(res):
    if True not in res and False not in res:
        return set([item["value"] for r in res for item in list(r.values())])
    else:
        return set(res)

def compare_results(gold_res, sys_res) -> EvalStats:
    assert gold_res is not None

    gold: Set = sanitize_sparql_result(gold_res)

    if sys_res is None:
        return EvalStats(tp=0, fp=0, fn=len(gold))
    else:
        sys: Set = sanitize_sparql_result(sys_res)
        tp = len(gold.intersection(sys))
        fp = len(sys.difference(gold))
        fn = len(gold.difference(sys))
        return EvalStats(tp=tp, fp=fp, fn=fn, emc=(1 if fp == 0 and fn == 0 and tp > 0 else 0))


# def calc_eval_stats(stats: EvalStats) -> EvalStats:
#     tp = stats.tp
#     fp = stats.fp
#     fn = stats.fn
#     precision = tp / (tp + fp) if (tp + fp) > 0 else None
#     recall = tp / (tp + fn) if (tp + fn) > 0 else None
#     f1 = (2 * precision * recall) / (precision + recall) if precision is not None and recall is not None and (
#             precision + recall) > 0 else None
#     if fp == 0 and fn == 0 and tp > 0:
#         stats.emc = 1
#     stats.f1 = f1
#     stats.prec = precision
#     stats.rec = recall
#     return stats

def eval_queries(gold, pred, sparql_endpoint=None, debug=True):
    if sparql_endpoint is None:
        sparql_endpoint = SPARQLEndpoint()

    gold_res = None
    pred_res = None
    try:
        gold_res = sparql_endpoint.get_results_query(gold)
    except Exception as e:
        if debug:
            print("Error:", e, gold, pred)
        raise e
    try:
        pred_res = sparql_endpoint.get_results_query(pred)
    except Exception as e:
        if debug:
            print("Error:", e, gold, pred)

    estats = compare_results(gold_res=gold_res, sys_res=pred_res)
    stats = estats.to_dict()
    stats["Gold SPARQL"] = remove_prefix(gold)
    stats["Generated SPARQL"] = remove_prefix(pred)

    return estats, stats

def remove_prefix(q: str) -> str:
    if isinstance(q, str):
        return re.sub(r"PREFIX .*?: <.*?>[\n ]?", "", q)
    else:
        return q

def replace_namespaces_dirty(q: str) -> str:
    if isinstance(q, str):
        for p, ns in default_namespaces:
            q = re.sub(f"<{str(ns)}(.*?)>", p + ":\\1", q)
        q = re.sub("<http://www.w3.org/1999/02/22-rdf-syntax-ns#(.*?)>", "rdf:\\1", q)
        return q
    else:
        return q

def create_namespace_manager(
        namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
        namespace_manager: Optional[NamespaceManager] = None
) -> NamespaceManager:
    if namespace_manager is not None:
        if namespaces is not None:
            for name, ns in namespaces:
                namespace_manager.bind(name, ns, override=True, replace=True)
        return namespace_manager
    else:
        if namespaces is None:
            namespaces = default_namespaces
        assert namespaces is not None

        namespace_manager = NamespaceManager(Graph())
        for name, ns in namespaces:
            namespace_manager.bind(name, ns, override=True, replace=True)

        return namespace_manager


def roundrobin(*iterables):
    "Visit input iterables in a cycle until each is exhausted."
    # roundrobin('ABC', 'D', 'EF') → A D E B F C
    # Algorithm credited to George Sakkis
    #try:
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = itertools.cycle(itertools.islice(iterators, num_active))
        yield from map(next, iterators)
    # except Exception as e:
    #     print(e)
    #     print(traceback.format_exc())
    #     raise e

def weighted_roundrobin(iterables, weights):
    assert len(iterables) == len(weights)
    iterators = sum([[iter(it)] * w for it, w in zip(iterables, weights)], [])
    for num_active in range(len(iterators), 0, -1):
        iterators = itertools.cycle(itertools.islice(iterators, num_active))
        yield from map(next, iterators)

def partition(n, m, prefix):
    if n == 0:
        yield prefix

    for i in range(1, min(m, n)+1):
        yield from partition(n-i, i, prefix + [i])

def diagonal_generator(candidates):
    #n = 2
    lens = [len(c) for c in candidates]
    seen = set()
    for n in range(1, sum(lens)):
        for part in (p for p in partition(n, n, []) if len(p) <= len(lens)):
            padded_part = [max(0, p-1) for p in part] + ([0] * (len(lens) - len(part)))
            #sorted_part = tuple(sorted(padded_part, reverse=True))
            #if sorted_part not in seen:
            #seen.add(sorted_part)
            for perm in more_itertools.distinct_permutations(padded_part):
                if all((a > b for a, b in zip(lens, perm))) and perm not in seen:
                    seen.add(perm)
                    yield tuple([candidates[idx][perm[idx]] for idx in range(len(candidates))])
        # res = [{q for q in itertools.permutations(p + ([0] * (len(lens)-len(p))))} for p in partition(n, n, []) if len(p) <= len(lens)]
        #res = [{q for q in itertools.permutations(p + ([0] * (len(lens) - len(p)))) if all((a > b for a, b in zip(lens, q)))} ]
        # res = [(p + ([0] * (n-len(p)))) for p in partition(n, n, [])]
        #print(list(partition(n, n, [])))
        #print(res)

# Adapted from https://github.com/linnik/roundrobin
class RRItemWeight:

    __slots__ = ('key', 'weight', 'current_weight', 'effective_weight')

    def __init__(self, key, weight):
        self.key = key
        self.weight = weight
        self.current_weight = 0
        self.effective_weight = weight

# Adapted from https://github.com/linnik/roundrobin
def smooth_roundrobin(dataset):
    dataset_length = len(dataset)
    dataset_extra_weights = [RRItemWeight(*x) for x in dataset]

    def get_next():
        if dataset_length == 0:
            return None
        if dataset_length == 1:
            return dataset[0][0]

        total_weight = 0
        result = None
        for extra in dataset_extra_weights:
            extra.current_weight += extra.effective_weight
            total_weight += extra.effective_weight
            if extra.effective_weight < extra.weight:
                extra.effective_weight += 1
            if not result or result.current_weight < extra.current_weight:
                result = extra
        if not result:  # this should be unreachable, but check anyway
            raise RuntimeError
        result.current_weight -= total_weight
        return result.key

    return get_next


def prettier_print(stats):
    if "True Positives" not in stats:
        return f"F1: {round(stats['F1'] if stats['F1'] is not None else 0.0, 4)}, P: {round(stats['Precision'] if stats['Precision'] is not None else 0.0, 4)}, R: {round(stats['Recall'] if stats['Recall'] is not None else 0.0, 4)}"
    else:
        return f"F1: {round(stats['F1'] if stats['F1'] is not None else 0.0, 4)}, TP: {stats['True Positives']}, FP: {stats['False Positives']}, FN: {stats['False Negatives']}, EM: {stats['Exact matches']}, P: {round(stats['Precision'] if stats['Precision'] is not None else 0.0, 4)}, R: {round(stats['Recall'] if stats['Recall'] is not None else 0.0, 4)}"
