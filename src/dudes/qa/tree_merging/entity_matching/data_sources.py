from __future__ import annotations

import logging
import statistics
from abc import ABC, abstractmethod
from collections import UserDict
from typing import Dict, Tuple, List

import more_itertools
import rpyc
from spacy.tokens import Token
from sqlitedict import SqliteDict
from treelib import Node

from dudes import consts, utils
from dudes.dudes_token import TokenWrapper, DUDESToken
from dudes.qa.dudes_rpc_service import TrieTaggerWrapper
from dudes.qa.tree_merging.entity_matching.trie_tagger import TrieLexicon
from dudes.utils import tree_map
from dudes.dudes_tree import Tree
from lemon.lemon import LexicalEntry, Marker
from lemon.lexicon import Lexicon


class DataSource(ABC):
    @abstractmethod
    def exec_condition(self, node: Node, tree: Tree) -> bool:
        pass

    @abstractmethod
    def match_node(self, node: Node, tree: Tree) -> bool:
        pass

    @abstractmethod
    def match_tree(self, tree: Tree) -> bool:
        pass


class DBpediaSpotlightSource(DataSource):
    # def exec_condition(self, node: Node, tree: Tree) -> bool:
    #     if len(node.data.token.candidate_uris) == 0 and len(node.data.token.ent_kb_id_) > 0:
    #         entries = node.data.lex_candidates
    #         written_reps = set().union(*[Lexicon.get_attr_by_name(l, "written_rep") for l in entries])
    #         cover_full_text = (utils.any_in_list(node.data.token.text_, written_reps) or utils.any_in_list(node.data.token.text_, written_reps, lower=True))
    #         if (
    #                 (len(entries) == 0 or not cover_full_text)
    #                 and any([
    #                     pos in ["PROPN"]  # "NOUN",
    #                     for pos in ([node.data.token.pos_] + [tok.pos_ for tok in node.data.token.merged_tokens])
    #                 ])
    #         ):
    #             return True
    #     return False
    def exec_condition(self, node: Node, tree: Tree) -> bool:
        return len(node.data.token.ent_kb_id_) > 0

    def match_node(self, node: Node, tree: Tree) -> bool:
        if self.exec_condition(node, tree):
            node.data.token.are_strict_candidate_uris = node.data.token.are_strict_candidate_uris or len(node.data.token.candidate_uris) == 0
            node.data.token.candidate_uris += [eid for eid in node.data.token.ent_kb_id_ if len(eid) > 0]
            return True
        return False

    def match_tree(self, tree: Tree) -> bool:
        return any(tree_map(self.match_node, tree))


class TrieTaggerSource(DataSource):

    def __init__(self, tagger, threshold=0.5, candidate_limit=10):
        self.trie_tagger = tagger
        self.threshold = threshold
        self.tags = None
        self.tags_strict = None
        self.tags_raw = None
        self.tags_raw_strict = None
        self.candidate_limit = candidate_limit

    # def exec_condition(self, node: Node, tree: Tree) -> bool:
    #     if len(node.data.token.candidate_uris) == 0:
    #         entries = node.data.lex_candidates
    #         written_reps = set().union(*[Lexicon.get_attr_by_name(l, "written_rep") for l in entries])
    #         cover_full_text = (utils.any_in_list(node.data.token.text_, written_reps) or utils.any_in_list(node.data.token.text_, written_reps, lower=True))
    #         if (
    #                 (len(entries) == 0 or not cover_full_text)
    #                 and any([
    #                     pos in ["NOUN", "PROPN"]  # "NOUN",
    #                     for pos in ([node.data.token.pos_] + [tok.pos_ for tok in node.data.token.merged_tokens])
    #                 ])
    #         ):
    #             return True
    #     return False

    def exec_condition(self, node: Node, tree: Tree) -> bool:
        return not utils.any_in_list(node.data.token.text_, consts.special_words, lower=True)

    def _tag_tree(self, tree: Tree):
        if self.tags is None and self.tags_raw is None:
            tokens = [TokenWrapper.from_token(t) for t in tree.get_node(tree.root).data.token.main_token.doc]
            input_str = TrieLexicon.reconstruct_str(tokens)
            #self.tags_strict, self.tags_raw_strict = self.trie_tagger.tag(input_str, threshold=0.99999)
            self.tags, self.tags_raw = self.trie_tagger.tag(input_str, threshold=self.threshold)

    @staticmethod
    def _strict_condition(td, node):
        return any([TrieLexicon.normalized_sim(td.found_str, t.lower()) > 0.99 for t in node.data.token.text_])

    def match_node(self, node: Node, tree: Tree):
        changed = False
        if self.exec_condition(node, tree):
            was_none = False  # differentiate single call from call for whole tree -> do not recalculate
            if self.tags is None and self.tags_raw is None:
                was_none = True
                self._tag_tree(tree)
            assert self.tags is not None and self.tags_raw is not None #and self.tags_strict is not None and self.tags_raw_strict is not None

            # Only considering first text_ to make sure this requirement is not too weak for few-word tokens
            # - first one should be most likely match, also intersection is too weak
            # if any([TrieLexicon.normalized_sim(td.intersection, t.lower()) > 0.99 for t in node.data.token.text_])]
            # tagger_data_strict = [td for td in self.tags[node.data.token.idx]
            #                       if self._strict_condition(td, node)]

            # if len(tagger_data_strict) > 0:
            #     tagger_data_strict.sort(key=lambda x: (x.sim, len(x.intersection)), reverse=True)
            #     node_tags_strict = list(dict.fromkeys(sum([list(t.uris) for t in tagger_data_strict], [])))
            #     node.data.token.tagger_kb_ids = node_tags_strict
            #     node.data.token.candidate_uris.update(node_tags_strict[:self.candidate_limit])
            #     #default value is True, so switching to False indicates non-strict entries somewhere

            # else:
            tagger_data = sorted(self.tags[node.data.token.idx], key=lambda x: (statistics.mean([TrieLexicon.normalized_sim(x.found_str, t.lower()) for t in node.data.token.text_]), len(x.intersection)), reverse=True)
            itnonstrict, itstrict = more_itertools.partition(lambda td: self._strict_condition(td, node), tagger_data)
            strict = list(itstrict)
            nonstrict = list(itnonstrict)
            node_tags = utils.make_distinct_ordered(sum([list(t.uris) for t in (strict + nonstrict)], []))


            node.data.token.tagger_kb_ids = node_tags
            node.data.token.candidate_uris += node_tags[:self.candidate_limit]
            #node.data.token.are_strict_candidate_uris = False
            node.data.token.are_strict_candidate_uris = (
                    len(strict) > 0 and
                    (node.data.token.are_strict_candidate_uris or len(node.data.token.candidate_uris) == 0)
            )
            changed = True

            if was_none:
                self.tags = None
                self.tags_raw = None
                #self.tags_strict = None
                #self.tags_raw_strict = None

        return changed

    def match_tree(self, tree: Tree):
        self._tag_tree(tree)
        res = tree_map(self.match_node, tree)
        self.tags = None
        self.tags_raw = None
        return any(res)

class RPCTrieTaggerSource(TrieTaggerSource):
    def __init__(self, conn=None, threshold=0.5, candidate_limit=10):
        self.tagger = TrieTaggerWrapper(conn=conn)
        super().__init__(self.tagger, threshold, candidate_limit)


class DictSource(DataSource):
    def __init__(self, uri_dict: UserDict):
        self.uri_dict = uri_dict

    @classmethod
    def from_sqlite(cls, path, use_zstd=False):
        if use_zstd:
            return cls(SqliteDict(path, encode=utils.sqlitedict_encode_zstd, decode=utils.sqlitedict_decode_zstd))
        else:
            return cls(SqliteDict(path))

    # def exec_condition(self, node: Node, tree: Tree) -> bool:
    #     if len(node.data.token.candidate_uris) == 0:
    #         if not utils.any_in_list(node.data.token.text_, consts.special_words, lower=True):
    #             entries = node.data.lex_candidates
    #             written_reps = set().union(*[Lexicon.get_attr_by_name(l, "written_rep") for l in entries])
    #             cover_full_text = (utils.any_in_list(node.data.token.text_, written_reps) or utils.any_in_list(node.data.token.text_, written_reps, lower=True))
    #             if (
    #                     (len(entries) == 0 or not cover_full_text)
    #                     and any([
    #                         pos in ["PROPN"]  # "NOUN",
    #                         for pos in ([node.data.token.pos_] + [tok.pos_ for tok in node.data.token.merged_tokens])
    #                     ])
    #             ):
    #                 return True
    #     return False
    def exec_condition(self, node: Node, tree: Tree) -> bool:
        return not utils.any_in_list(node.data.token.text_, consts.special_words, lower=True)

    def match_node(self, node: Node, tree: Tree):
        if self.exec_condition(node, tree):
            for t in node.data.token.text_:
                node.data.token.candidate_uris += self.uri_dict.get(t, set())
            node.data.token.are_strict_candidate_uris = False # exact match but typically data is noisy so we should be careful here
            return True
        return False

    def match_tree(self, tree: Tree):
        return any(tree_map(self.match_node, tree))


class LexiconSource(DataSource):
    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon

    # def exec_condition(self, node: Node, tree: Tree) -> bool:
    #     if len(node.data.lex_candidates) == 0 or not node.data.are_strict_candidates:
    #         return True
    #     return False

    def exec_condition(self, node: Node, tree: Tree) -> bool:
        return True

    def match_node(self, node: Node, tree: Tree):
        changed = False
        if self.exec_condition(node, tree):
            node.data.lex_candidates, node.data.are_strict_candidates = self.lexicon.find_entry(node.data.token) #if not node.data.token.is_stop else ([], False)

            if len(node.data.lex_candidates) == 0:
                logging.warning("No lexical entry found for " + node.data.token.text)

            comp_markers: List[str] = []

            pos_tags = [node.data.token.tag_] + [t.tag_ for t in node.data.token.merged_tokens]
            if utils.any_in_list(pos_tags, ["JJR", "RBR"]):
                comp_markers.append("than")

            # check whether there are markers which have been merged according to the lexical entry
            to_merge: Dict[Token, Tuple[Token, LexicalEntry, Marker]] = dict()
            lex_with_marker_match = []
            for mt in node.data.token.merged_tokens:
                marker_matches: List[tuple[Token, LexicalEntry, Marker | str]] = []
                for cand in node.data.lex_candidates:
                    markers = Lexicon.get_markers(cand)
                    for marker in markers:
                        try:
                            if marker.canonical_form.written_rep.lower() == mt.text.lower():
                                marker_matches.append((mt, cand, marker))
                                lex_with_marker_match.append(cand)
                        except AttributeError:
                            pass
                    if mt.text.lower() in comp_markers:
                        marker_matches.append((mt, cand, "than"))
                        lex_with_marker_match.append(cand)

                if len(marker_matches) > 0:
                    for merged_token, cand, marker in marker_matches:  # TODO: we currently overwrite all but one if multiple matches are present...
                        to_merge[merged_token] = merged_token, cand, marker

            # all entries have markers but none match -> no strict candidates
            if all([
                len(Lexicon.get_markers(cand)) > 0
                for cand in node.data.lex_candidates
            ]) and len(to_merge.values()) == 0:
                node.data.are_strict_candidates = False
                changed = True

            #lex_without_marker_match = node.data.lex_candidates
            lex_with_marker_match = utils.make_distinct(lex_with_marker_match)
            lex_without_marker = [cand for cand in node.data.lex_candidates if cand not in lex_with_marker_match and len(Lexicon.get_markers(cand)) == 0]
            lex_with_marker_without_match = [cand for cand in node.data.lex_candidates if cand not in lex_with_marker_match and len(Lexicon.get_markers(cand)) > 0]

            # filter candidate lexical entries -> if any marker match is found, keep only entries with marker matches
            # removed this behavior, just return prioritized list of entries
            if len(to_merge.values()) > 0:
                changed = True

            def token_entry_sim(e: LexicalEntry) -> Tuple[float, float]:
                t = node.data.token
                wrsim = max([statistics.mean([utils.levenshtein_sim_normalized(tr, wr)
                                              for wr in Lexicon.get_attr_by_name(e, "written_rep")])
                             for tr in t.text_])
                urisim = max([statistics.mean([utils.levenshtein_sim_normalized(tr, wr)
                                               for wr in Lexicon.get_uri_by_name(e, consts.uri_attrs)])
                              for tr in t.text_])
                return wrsim, urisim

            if len(lex_with_marker_match) + len(lex_without_marker) + len(lex_with_marker_without_match) > 0:
                lex_with_marker_match.sort(key=token_entry_sim, reverse=True)
                lex_without_marker.sort(key=token_entry_sim, reverse=True)
                lex_with_marker_without_match.sort(key=token_entry_sim, reverse=True)
                node.data.lex_candidates = lex_with_marker_match + lex_without_marker + lex_with_marker_without_match
            else:
                logging.error("All entry lists empty for " + node.data.token.text)

        return changed

    def match_tree(self, tree: Tree):
        return any(tree_map(self.match_node, tree))
