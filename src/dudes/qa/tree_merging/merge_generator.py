import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from queue import Queue
from typing import List, Set, Tuple, Dict

import rpyc
from treelib import Node, Tree
from treelib.exceptions import NodeIDAbsentError

from dudes import consts, utils
from dudes.dudes_token import TokenWrapper
from dudes.qa.dudes_rpc_service import TrieTaggerWrapper
from dudes.qa.tree_merging.entity_matching.trie_tagger import TrieLexicon
from dudes.utils import tree_map
from lemon.lemon import LexicalEntry, Marker
from lemon.lexicon import Lexicon


class NodeMergeGenerator(ABC):
    @abstractmethod
    def generate(self, node: Node, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        # set of sets of pairs (node to merge into, node to eliminate with merge) which can only be applied together
        pass

    @staticmethod
    def _skip_comp_merges(merges):
        # skip comparative phrases
        merges.difference_update(
            [frozenset([(r, c)]) for fs in merges for r, c in fs if
             utils.is_comp_phrase(r.data.token) or utils.is_comp_phrase(c.data.token)]
        )
        # skip comparative tokens
        merges.difference_update(
            [frozenset([(r, c)]) for fs in merges for r, c in fs if
             utils.is_comp_token(r.data.token) or utils.is_comp_token(c.data.token)]
        )


class TreeMergeGenerator(ABC):
    # def __init__(self, node_merge_generators: List[NodeMergeGenerator]):
    #     self.node_merge_generators = node_merge_generators

    @abstractmethod
    def generate(self, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        pass

    @staticmethod
    def _entity_map_to_merges(
            glob_entity_nodes: Dict[str, List[Node]],
            tree: Tree
    ) -> Set[frozenset[Tuple[Node, Node]]]:
        entity_merges: Set[frozenset[Tuple[Node, Node]]] = set()
        for ent, nodes in glob_entity_nodes.items():  # merge nodes of same entity
            try:
                curr_ent_merges: Set[Tuple[Node, Node]] = set()

                nodes = list(
                    sorted({n.identifier: n for n in nodes if tree.contains(n.identifier)}.values(), reverse=True,
                           key=lambda x: x.identifier))
                if len(nodes) == 0:
                    continue
                merge_entities = nodes[:-1]

                # last node is "hightest" node of entity (BFS), keep that one as "root"
                for n in merge_entities:
                    curr_ent_merges.add((nodes[-1], n))

                entity_merges.add(frozenset(curr_ent_merges))
            except NodeIDAbsentError as e:
                logging.warning("Merge combination failed for global", tree, glob_entity_nodes, e)
                continue
            except AttributeError as e:
                logging.warning("Merge combination failed for global", tree, glob_entity_nodes, e)
                continue

        return entity_merges


class DBpediaSpotlightMergeGenerator(TreeMergeGenerator):

    def generate(self, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        # entity_merges: Set[frozenset[Tuple[Node, Node]]] = set()
        glob_entity_nodes: Dict[str, List[Node]] = defaultdict(list)

        q: Queue[Node] = Queue()
        q.put(tree.get_node(tree.root))

        while not q.empty():
            node: Node
            node = q.get()

            if len(node.data.token.ent_kb_id_) > 0 and len(node.data.token.ent_kb_id_[0]) > 0:
                for eid in node.data.token.ent_kb_id_:
                    glob_entity_nodes[eid].append(node)

            for c in tree.children(node.identifier):
                q.put(c)

        return self._entity_map_to_merges(glob_entity_nodes, tree)


class TrieTaggerMergeGenerator(TreeMergeGenerator):

    def __init__(self, trie_tagger, threshold=0.5, candidate_limit=10):
        self.trie_tagger = trie_tagger
        self.threshold = threshold
        self.candidate_limit = candidate_limit
        self.tags = None
        self.tags_raw = None
        self.input_str = None

    def _refresh_tagging(self, tree: Tree):
        tokens = [TokenWrapper.from_token(t) for t in tree.get_node(tree.root).data.token.main_token.doc]
        input_str = TrieLexicon.reconstruct_str(tokens)
        if input_str != self.input_str:
            self.input_str = input_str
            self.tags, self.tags_raw = self.trie_tagger.tag(input_str, threshold=self.threshold)

    def generate(self, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        self._refresh_tagging(tree)

        assert self.tags is not None and self.tags_raw is not None

        glob_entity_nodes: Dict[str, List[Node]] = defaultdict(list)

        q: Queue[Node] = Queue()
        q.put(tree.get_node(tree.root))

        while not q.empty():
            node: Node
            node = q.get()

            if node.data.token.idx is None:
                pass
            tagger_data = sorted(self.tags[node.data.token.idx], key=lambda x: (x.sim, len(x.intersection)),
                                 reverse=True)
            node_tags = list(dict.fromkeys(sum([list(t.uris) for t in tagger_data], [])))

            if len(node_tags) > 0:
                for eid in node_tags[:self.candidate_limit]:
                    glob_entity_nodes[eid].append(node)

            for c in tree.children(node.identifier):
                q.put(c)

        return self._entity_map_to_merges(glob_entity_nodes, tree)

class RPCTrieTaggerMergeGenerator(TrieTaggerMergeGenerator):
    def __init__(self, conn=None, threshold=0.5, candidate_limit=10):
        self.tagger = TrieTaggerWrapper(conn=conn)
        super().__init__(self.tagger, threshold, candidate_limit)

class LexiconMergeGenerator(TreeMergeGenerator):
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def generate(self, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        q: Queue[Node] = Queue()
        q.put(tree.get_node(tree.root))

        lex_merges: Set[frozenset[Tuple[Node, Node]]] = set()

        while not q.empty():
            node: Node = q.get()
            lex_candidates, are_strict_candidates = self.lexicon.find_entry(node.data.token) #if not node.data.token.is_stop else ([], False)

            if len(lex_candidates) == 0:
                logging.warning("No lexical entry found for " + node.data.token.text)

            # merge markers into their main token
            to_merge: Dict[Node, Tuple[Node, LexicalEntry, Marker]] = dict()
            for c in tree.children(node.identifier):
                marker_matches: List[tuple[Node, LexicalEntry, Marker]] = []
                for cand in lex_candidates:
                    for marker in Lexicon.get_markers(cand):
                        try:
                            if marker.canonical_form.written_rep.lower() == c.data.token.text.lower():
                                marker_matches.append((c, cand, marker))
                        except AttributeError:
                            pass
                if len(marker_matches) > 0:
                    for child, cand, marker in marker_matches:  # TODO: we currently overwrite all but one if multiple matches are present...
                        to_merge[child] = child, cand, marker

            lex_merges.update([frozenset([(node, c)]) for c, cand, marker in to_merge.values()])

            for c in tree.children(node.identifier):
                q.put(c)

        return lex_merges

class MiscMergeGenerator(TreeMergeGenerator):
    def __init__(self, node_merge_generators: List[NodeMergeGenerator]):
        self.node_merge_generators = node_merge_generators

    @classmethod
    def default(cls):
        return cls([
            MergeVerbAux(),
            MergeModifiers(),
            MergeCompounds(),
            MergeAdpositions(),
            MergeComparatives(),
        ])

    def generate(self, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        misc_merges: Set[frozenset[Tuple[Node, Node]]] = set()

        for generator in self.node_merge_generators:
            gen_res = tree_map(generator.generate, tree)
            misc_merges.update(*[res for res in gen_res])

        return misc_merges


class MergeVerbAux(NodeMergeGenerator):
    def generate(self, node: Node, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        misc_merges: Set[frozenset[Tuple[Node, Node]]] = set()

        if node.data.token.pos_ == "VERB":
            misc_merges.update(
                [frozenset([(node, c)]) for c in tree.children(node.identifier) if c.data.token.pos_ in ["AUX"]])
        if node.data.token.pos_ == "AUX":
            misc_merges.update(
                [frozenset([(node, c)]) for c in tree.children(node.identifier) if c.data.token.pos_ in ["VERB"]])

        self._skip_comp_merges(misc_merges)

        return misc_merges


class MergeModifiers(NodeMergeGenerator):
    def generate(self, node: Node, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        misc_merges: Set[frozenset[Tuple[Node, Node]]] = set()

        if node.data.token.pos_ in ["NOUN", "PROPN"]:
            misc_merges.update([
                frozenset([(node, c)])
                for c in tree.children(node.identifier)
                if c.data.token.dep_ in ["amod", "advmod"] and c.data.token.text.lower() not in consts.special_words
            ])

        self._skip_comp_merges(misc_merges)

        return misc_merges

class MergeCompounds(NodeMergeGenerator):
    def generate(self, node: Node, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        misc_merges: Set[frozenset[Tuple[Node, Node]]] = set()

        #if node.data.token.pos_ in ["NOUN", "PROPN"]:
        # I guess POS check is unnecessary and if it is not NOUN but compound is there we still want to merge
        misc_merges.update([
            frozenset([(node, c)])
            for c in tree.children(node.identifier)
            if c.data.token.dep_ in ["compound"] and c.data.token.text.lower() not in consts.special_words
        ])

        self._skip_comp_merges(misc_merges)

        return misc_merges


class MergeAdpositions(NodeMergeGenerator):
    def generate(self, node: Node, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        misc_merges: Set[frozenset[Tuple[Node, Node]]] = set()

        # merge adpositions, but only if token is not part of entity -> does not happen due to continue
        # if not (node.data.token.ent_kb_id_ is not None and len(node.data.token.ent_kb_id_) > 0):
        misc_merges.update([
            frozenset([(node, c)])
            for c in tree.children(node.identifier)
            if ((c.data.token.dep_ in ["det"] or c.data.token.pos_ in ["ADP", "PART"])
                and c.data.token.text.lower() not in consts.special_words)
        ])

        self._skip_comp_merges(misc_merges)

        return misc_merges


class MergeComparatives(NodeMergeGenerator):
    def generate(self, node: Node, tree: Tree) -> Set[frozenset[Tuple[Node, Node]]]:
        comp_merges: Set[frozenset[Tuple[Node, Node]]] = set()

        # merge more/less/... than
        than_tokens = [c for c in tree.children(node.identifier) if c.data.token.main_token.text.lower() == "than"]
        gt_tokens = [c for c in tree.children(node.identifier) if
                     c.data.token.main_token.text.lower() in consts.comp_gt_keywords + consts.comp_gt_keywords_no_than]
        lt_tokens = [c for c in tree.children(node.identifier) if
                     c.data.token.main_token.text.lower() in consts.comp_lt_keywords + consts.comp_lt_keywords_no_than]

        # comp_tokens = set(gt_tokens + lt_tokens + than_tokens)

        assert len(than_tokens) <= 1 and len(gt_tokens) + len(lt_tokens) <= 1
        # TODO: Handle multiple comparisons by merging each closest indices than + more/less

        if len(than_tokens) == 1:
            if len(gt_tokens) == 1:
                comp_merges.add(frozenset([(than_tokens[0], gt_tokens[0])]))
            if len(lt_tokens) == 1:
                comp_merges.add(frozenset([(than_tokens[0], lt_tokens[0])]))

        return comp_merges
