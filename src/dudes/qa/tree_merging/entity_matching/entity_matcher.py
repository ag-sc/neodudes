import os
import sys
from typing import Optional, List, Iterable, Tuple

import rpyc
from rdflib import Namespace
from rdflib.namespace import NamespaceManager
from treelib import Tree

from dudes import consts
from dudes.qa.tree_merging.entity_matching.data_sources import DataSource, LexiconSource, DBpediaSpotlightSource, \
    TrieTaggerSource, DictSource, RPCTrieTaggerSource
from dudes.qa.tree_merging.entity_matching.trie_tagger import TrieTagger
from lemon.lemon_parser import LEMONParser
from lemon.lexicon import Lexicon


class EntityMatcher:
    def __init__(self, data_sources: List[DataSource]):
        self.data_sources = data_sources

    @classmethod
    def default(
            cls,
            lexicon: Optional[Lexicon],
            rpc_conn: Optional[rpyc.Connection] = None,
            namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
            namespace_manager: Optional[NamespaceManager] = None,
    ):
        ds: List[DataSource] = []

        if lexicon is None:
            lexicon = LEMONParser.from_ttl_dir(namespaces=namespaces, namespace_manager=namespace_manager).lexicon

        ds.append(LexiconSource(lexicon))

        ds.append(DBpediaSpotlightSource())

        # if trie_tagger_path is not None:
        #     trie_tagger = TrieTagger()
        #     trie_tagger.load_from_file(trie_tagger_path)
        #     ds.append(TrieTaggerSource(tagger=trie_tagger, threshold=trie_threshold, candidate_limit=trie_candidate_limit))

        ds.append(RPCTrieTaggerSource(conn=rpc_conn))

        # if labels_db_path is not None:
        #     #labels = SqliteDict(labels_db_path)
        #     ds.append(DictSource.from_sqlite(labels_db_path))
        #
        # if anchors_db_path is not None:
        #     # anchors = SqliteDict(anchors_db_path,
        #     #                      encode=utils.sqlitedict_encode_zstd,
        #     #                      decode=utils.sqlitedict_decode_zstd)
        #     ds.append(DictSource.from_sqlite(anchors_db_path, use_zstd=True))

        return cls(ds)

    @staticmethod
    def all_nodes_recognized(tree: Tree):
        return all([len(node.data.token.candidate_uris) > 0 or len(node.data.lex_candidates) > 0 for node in tree.all_nodes()])

    def match(self, tree: Tree):
        for data_source in self.data_sources:
            changed = data_source.match_tree(tree)
            # CHANGED - apply all sources and generate all possible DUDES!
            # source exec conditions should be defined in a restrictive way such that
            # "not changed" means "all nodes have entries/URIs"
            # if self.all_nodes_recognized(tree):
            #     break
