import copy
import os
import sys
from typing import List, Tuple, Set, Optional, Any, Generator, Iterable, Dict

import rpyc
from rdflib import Namespace
from rdflib.namespace import NamespaceManager
from treelib import Tree, Node
from treelib.exceptions import NodeIDAbsentError

from dudes import utils, consts
from dudes.qa.tree_merging.entity_matching.entity_matcher import EntityMatcher
from dudes.qa.tree_merging.entity_matching.trie_tagger import TrieTagger
from dudes.qa.tree_merging.merge_generator import TreeMergeGenerator, DBpediaSpotlightMergeGenerator, \
    TrieTaggerMergeGenerator, LexiconMergeGenerator, MiscMergeGenerator, RPCTrieTaggerMergeGenerator
from dudes.qa.tree_merging.tree_scorer import TreeScorer, StrictFirstScorer, FewNodesStrictFirstScorer, \
    WeightedAvgStrictFirstScorer
from lemon.lemon_parser import LEMONParser
from lemon.lexicon import Lexicon


class TreeMerger:

    def __init__(self,
                 merge_generators: List[TreeMergeGenerator],
                 entity_matcher: EntityMatcher,
                 tree_scorer: TreeScorer):
        self.merge_generators = merge_generators
        self.entity_matcher = entity_matcher
        self.tree_scorer = tree_scorer

    @classmethod
    def default(cls,
                lexicon: Optional[Lexicon] = None,
                rpc_conn: Optional[rpyc.Connection] = None,
                namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                namespace_manager: Optional[NamespaceManager] = None,
                ):
        merge_generators: List[TreeMergeGenerator] = []
        merge_generators.append(DBpediaSpotlightMergeGenerator())
        # if trie_tagger_path is not None:
        #     trie_tagger = TrieTagger()
        #     trie_tagger.load_from_file(trie_tagger_path)
        #     merge_generators.append(TrieTaggerMergeGenerator(trie_tagger))

        merge_generators.append(RPCTrieTaggerMergeGenerator(conn=rpc_conn))


        if lexicon is None:
            lexicon = LEMONParser.from_ttl_dir(namespaces=namespaces, namespace_manager=namespace_manager).lexicon

        merge_generators.append(LexiconMergeGenerator(lexicon))
        merge_generators.append(MiscMergeGenerator.default())

        return cls(
            merge_generators=merge_generators,
            entity_matcher=EntityMatcher.default(lexicon=lexicon,
                                                 rpc_conn=rpc_conn,
                                                 namespaces=namespaces,
                                                 namespace_manager=namespace_manager),
            tree_scorer=WeightedAvgStrictFirstScorer(),
        )

    def merge(self, trees: Iterable[Tree]) -> Generator[Tree, None, None]:

        # def total_weight(tree: Tree) -> float:
        #     return sum([len(node.data.token.merged_tokens) + 1 for node in tree.nodes.values()])

        merged_trees: Dict[str, Tree] = dict()

        for tree in trees:
            merges: Set[frozenset[Tuple[Node, Node]]] = set()
            for mg in self.merge_generators:
                gen_germes = mg.generate(tree)
                merges = merges.union(gen_germes)

            seen: Set[frozenset[Tuple[Node, Node]]] = set()

            # init_weight = total_weight(tree)

            for merge_units in utils.powerset(merges):
                try:
                    fs_merges = frozenset(set().union(*merge_units))
                    if fs_merges not in seen:
                        seen.add(fs_merges)
                        curr_tree = copy.deepcopy(tree)
                        curr_merges = set(
                            [(into_node, elim_node) for merge_unit in merge_units for into_node, elim_node in
                             merge_unit])
                        curr_merges_sorted = list(
                            sorted(curr_merges, key=lambda x: (x[1].identifier, x[0].identifier), reverse=True))
                        for into_node, elim_node in curr_merges_sorted:
                            curr_tree.get_node(into_node.identifier).data.token.merge(curr_tree.get_node(elim_node.identifier).data.token)
                        for into_node, elim_node in curr_merges_sorted:
                            curr_tree.link_past_node(elim_node.identifier)

                        # if total_weight(curr_tree) < init_weight:
                        #     curr_tree2 = copy.deepcopy(tree)
                        #     curr_merges2 = set([(into_node, elim_node) for merge_unit in merge_units for into_node, elim_node in merge_unit])
                        #     curr_merges_sorted2 = list(
                        #         sorted(curr_merges2, key=lambda x: (x[1].identifier, x[0].identifier), reverse=True))
                        #     for into_node, elim_node in curr_merges_sorted2:
                        #         curr_tree2.get_node(into_node.identifier).data.token.merge(curr_tree2.get_node(elim_node.identifier).data.token)
                        #     for into_node, elim_node in curr_merges_sorted2:
                        #         curr_tree2.link_past_node(elim_node.identifier)

                        merged_trees[str(curr_tree)] = curr_tree
                except NodeIDAbsentError as e:
                    continue
                except AttributeError as e:
                    continue

        scored_trees: List[Tuple[Any, Tree]] = []

        for t in merged_trees.values():
            self.entity_matcher.match(t)
            score = self.tree_scorer.score(t)
            scored_trees.append((score, t))

        #scored_trees.sort(key=lambda x: (x[0], -len(x[1].nodes)), reverse=True)
        scored_trees.sort(key=lambda x: x[0], reverse=True)

        #maxval = scored_trees[0][0]

        # for score, tree in scored_trees:
        #     score = self.tree_scorer.score(tree)

        yield from scored_trees#[t for score, t in scored_trees]#[:self.candidate_limit]# if score == maxval]#
