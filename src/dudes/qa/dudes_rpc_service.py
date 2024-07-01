import os
import sys
import time
from functools import lru_cache
from threading import Thread

from guppy import hpy
from rpyc import ThreadedServer, ThreadPoolServer
from rpyc.utils.classic import obtain

import lemon

import rpyc  # type: ignore

from dudes import consts
from dudes.qa.sparql_selection.llm_query_selector import LLMQuerySelector, MultiLLMQuerySelector
from dudes.qa.tree_merging.entity_matching.trie_tagger import TrieTagger


class DUDESRPCService(rpyc.Service):

    def __init__(
            self,
            trie_tagger_path=os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels_trie_tagger_fr.cpl"),
            query_score_model_path=None
    ):
        self.trie_tagger = TrieTagger()
        if os.path.isfile(trie_tagger_path):
            print("Loading trie tagger from", trie_tagger_path)
            self.trie_tagger.load_from_file(trie_tagger_path)
        #self.query_score_models = None
        if query_score_model_path is None:
            query_score_model_path = [
                #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-29-16-536758.ckpt"),
                #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-30-21-434619.ckpt"),
                os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-30-49-282346.ckpt"),
                #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-30-59-969590.ckpt"),
                #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-31-30-134770.ckpt"),
                #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-31-54-743125.ckpt"),
                #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-32-25-476961.ckpt"),
                os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-32-30-917349.ckpt"),
                #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-33-02-776942.ckpt"),
                #os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "qald", "query_score_models", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_21-21-50-580922.ckpt"),
            ]
        elif isinstance(query_score_model_path, str):
            query_score_model_path = [query_score_model_path]
        #if all([os.path.isfile(qsp) for qsp in query_score_model_path]):
        print("Loading query score model from", query_score_model_path)
        self.query_selector = MultiLLMQuerySelector.from_paths(query_score_model_path)
            #self.query_score_models = [LLMQuerySelector(model_path=qsp) for qsp in query_score_model_path]
            #self.query_scorers = LLMQuerySelector(model_path=query_score_model_path) #QueryScoreT5.load_from_checkpoint(query_score_model_path)

    def exposed_tag(self, tag_str, threshold):
        return self._tag(tag_str, threshold)

    def exposed_compare_queries(self, question, query1, query2, dudes1, dudes2, numresults1, numresults2):
        return self._compare_queries(question, query1, query2, dudes1, dudes2, numresults1, numresults2)


    @lru_cache(maxsize=512)
    def _compare_queries(self, question, query1, query2, dudes1, dudes2, numresults1, numresults2):
        return self.query_selector.compare_queries(question, query1, query2, dudes1, dudes2, numresults1, numresults2)
        # res = []
        # for qs in self.query_score_models:
        #     qsres = qs.compare_queries(question, query1, query2, dudes1, dudes2, numresults1, numresults2)
        #     res.append(qsres)
        # return res

    @lru_cache(maxsize=128)
    def _tag(self, tag_str, threshold):
        return self.trie_tagger.tag(tag_str, threshold)

class TrieTaggerWrapper:
    def __init__(self, conn=None):
        if conn is not None:
            self.conn = conn
        else:
            self.conn = rpyc.connect(consts.rpc_host, consts.rpc_port, config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": 300})
        self.service = self.conn.root

    @lru_cache(maxsize=128)
    def tag(self, tag_str, threshold):
        res = obtain(self.service.tag(tag_str, threshold))
        return res

class LLMQuerySelectorWrapper:
    def __init__(self, conn):
        if conn is not None:
            self.conn = conn
        else:
            self.conn = rpyc.connect(consts.rpc_host, consts.rpc_port,
                                     config={"allow_public_attrs": True, "allow_pickle": True,
                                             "sync_request_timeout": 300})
        self.service = self.conn.root

    @lru_cache(maxsize=512)
    def compare_queries(self, question, query1, query2, dudes1, dudes2, numresults1, numresults2):
        res = obtain(self.service.compare_queries(question, query1, query2, str(dudes1), str(dudes2), numresults1, numresults2))
        return res


def rpc_thread():
    t = ThreadPoolServer(service=DUDESRPCService(),
                         nbThreads=consts.rpc_threads,
                         port=consts.rpc_port,
                         hostname=consts.rpc_host,
                         listener_timeout=None,
                         protocol_config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": 300})
    t.start()


def start_rpc_service(timeout=60):
    try:
        tagger = rpyc.connect(consts.rpc_host, consts.rpc_port)
    except ConnectionRefusedError:
        thread = Thread(target=rpc_thread)
        thread.start()
        start = time.time()
        while True:
            try:
                tagger = rpyc.connect(consts.rpc_host, consts.rpc_port)
                break
            except ConnectionRefusedError:
                if time.time() - start > timeout:
                    raise TimeoutError("Could not connect to RPC service!")
                else:
                    continue

        return thread
    print("Error: RPC service already running!")
    return None
