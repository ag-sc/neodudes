from typing import List, Optional, Iterable, Tuple

from rdflib import Namespace
from rdflib.namespace import NamespaceManager
from treelib import Tree

from dudes import utils
from dudes.qa.dudes_creation.dudes_creation_strategy import DUDESCreationStrategy, FixedTermsStrategy, EntityURIStrategy, \
    LemonStrategy, RawStrategy


class DUDESCreator:
    def __init__(self, strategies: List[DUDESCreationStrategy]):
        self.strategies = strategies

    @classmethod
    def default(cls,
                namespaces: Optional[Iterable[Tuple[str, Namespace]]] = None,
                namespace_manager: Optional[NamespaceManager] = None):
        return DUDESCreator([
            FixedTermsStrategy(),
            LemonStrategy(namespaces=namespaces, namespace_manager=namespace_manager),
            EntityURIStrategy(namespaces=namespaces, namespace_manager=namespace_manager),
            RawStrategy(),
        ])

    def assign_atomic_dudes(self, tree: Tree) -> Tree:
        next_var_id: int = 0
        for node in tree.all_nodes():
            cand_prio_list = []
            for strategy in self.strategies:
                new_cands, next_var_id = strategy.node_to_dudes(node=node,
                                                                tree=tree,
                                                                next_var_id=next_var_id)
                if isinstance(new_cands, Iterable) and len(new_cands) > 0:
                    cand_prio_list.append(utils.unique_based_on_str(new_cands))
                # if len(node.data.dudes_candidates) > 0:
                #     break
            node.data.dudes_candidates = list(utils.weighted_roundrobin(cand_prio_list, list(reversed(range(1, len(cand_prio_list)+1)))))
        return tree
