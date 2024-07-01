from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from queue import Queue
from typing import Any, Optional, List, Dict, Tuple

import spacy
import stanza
from numerizer import numerize
from spacy.tokens import Token
from stanza import Document
from stanza.models.common.doc import Word
from treelib import Tree, Node

from dudes import consts
from dudes.dudes_token import DUDESToken
from dudes.dudes_tree import DUDESTreeCombData


class PreprocessingStep(ABC):
    @abstractmethod
    def preprocess(self, input: Any) -> Any:
        pass


class NumerizeStep(PreprocessingStep):
    def preprocess(self, input: Any) -> Any:
        assert isinstance(input, str)
        return numerize(input)


class SpacyStepTrf(PreprocessingStep):
    def __init__(self,
                 dbpedia_spotlight_endpoint: Optional[str] = consts.dbpedia_spotlight_endpoint,
                 nlp: Optional[spacy.language.Language] = None,
                 merge_noun_chunks: bool = False,
                 merge_entities: bool = False):
        if nlp is None:
            nlp = spacy.load("en_core_web_trf")

        if 'dbpedia_spotlight' in nlp.pipe_names:
            nlp.remove_pipe('dbpedia_spotlight')

        if dbpedia_spotlight_endpoint is not None and len(dbpedia_spotlight_endpoint) > 0:
            nlp.add_pipe('dbpedia_spotlight', config={'dbpedia_rest_endpoint': dbpedia_spotlight_endpoint})
        else:
            nlp.add_pipe('dbpedia_spotlight')

        if merge_noun_chunks:
            nlp.add_pipe('merge_noun_chunks')

        if merge_entities:
            nlp.add_pipe('merge_entities')

        self.nlp = nlp

    def preprocess(self, input: Any) -> Any:
        assert isinstance(input, str)
        doc = self.nlp(input)
        roots = [token for token in doc if token.dep_ == "ROOT"]
        # DUDESParser._print_dep_tree(list(doc.sents)[0].root)
        if len(roots) > 1:
            logging.warning("More than one root! Only considering first one!")

        return roots[0]

class SpacyStepLg(PreprocessingStep):
    def __init__(self,
                 dbpedia_spotlight_endpoint: Optional[str] = consts.dbpedia_spotlight_endpoint,
                 nlp: Optional[spacy.language.Language] = None,
                 merge_noun_chunks: bool = False,
                 merge_entities: bool = False):
        if nlp is None:
            nlp = spacy.load("en_core_web_lg")

        if 'dbpedia_spotlight' in nlp.pipe_names:
            nlp.remove_pipe('dbpedia_spotlight')

        if dbpedia_spotlight_endpoint is not None and len(dbpedia_spotlight_endpoint) > 0:
            nlp.add_pipe('dbpedia_spotlight', config={'dbpedia_rest_endpoint': dbpedia_spotlight_endpoint})
        else:
            nlp.add_pipe('dbpedia_spotlight')

        if merge_noun_chunks:
            nlp.add_pipe('merge_noun_chunks')

        if merge_entities:
            nlp.add_pipe('merge_entities')

        self.nlp = nlp

    def preprocess(self, input: Any) -> Any:
        assert isinstance(input, str)
        doc = self.nlp(input)
        roots = [token for token in doc if token.dep_ == "ROOT"]
        # DUDESParser._print_dep_tree(list(doc.sents)[0].root)
        if len(roots) > 1:
            logging.warning("More than one root! Only considering first one!")

        return roots[0]


class TreeStep(PreprocessingStep):

    @staticmethod
    def _dep_to_tree(
            dep_root: Token,
            ignore_pos: Optional[List[str]] = None
    ) -> Tree:
        """
        Basic dependency tree to Tree conversion

        :param dep_root: spaCy dependency parse root node to create Tree from.
        :type dep_root: Token
        :param ignore_pos: POS tags/nodes to ignore when creating the Tree. By default ["SYM", "EOL", "SPACE", "X", "PUNCT"].
        :type ignore_pos: Optional[List[str]]
        :return: Tree representation of given dependency parse tree, ignoring nodes with POS tag in ignore\_pos.
        :rtype: Tree
        """
        if ignore_pos is None:
            ignore_pos = ["SYM", "EOL", "SPACE", "X", "PUNCT"]
        tree = Tree()
        idx = 0

        q: Queue[tuple[Token, int | None, str | None]] = Queue()
        q.put((dep_root, None, None))

        while not q.empty():
            tok: Token
            parent: int | None
            tok, parent, marker = q.get()

            if tok.pos_ in ignore_pos:
                continue

            node: Node = tree.create_node(tag=tok.text, identifier=idx,
                                          data=DUDESTreeCombData(token=DUDESToken(tok),
                                                                 dudes_candidates=[],
                                                                 lex_candidates=[],
                                                                 marker=marker),
                                          parent=parent)

            if tok.pos_ == "ADP":
                marker = tok.text

            for c in tok.children:
                q.put((c, idx, marker))

            idx += 1

        return tree

    def preprocess(self, input: Any) -> Any:
        assert isinstance(input, Token)
        return [TreeStep._dep_to_tree(input)]

class StanzaStep(PreprocessingStep):

    def __init__(self):
        #stanza.download('en')
        self.nlp = stanza.Pipeline('en', use_gpu=False)  # , use_gpu=False

    @staticmethod
    def _dep_to_tree(
            doc: Document,
            ignore_pos: Optional[List[str]] = None
    ) -> Tree:

        assert len(doc.sentences) >= 1

        children: Dict[Word | str, List[Word]] = defaultdict(list)

        dep: str
        for p, dep, c in doc.sentences[0].dependencies:
            # children[parent.text].append((parent, dep, child))
            if dep.lower() == "root":
                children["ROOT"].append(c)
            else:
                children[p].append(c)


        assert len(children["ROOT"]) == 1

        if ignore_pos is None:
            ignore_pos = ["SYM", "EOL", "SPACE", "X", "PUNCT"]
        tree = Tree()
        idx = 0

        q: Queue[tuple[Word, int | None, str | None]] = Queue()
        q.put((children["ROOT"][0], None, None))

        while not q.empty():
            tok: Word
            parent: int | None
            tok, parent, marker = q.get()

            if tok.pos in ignore_pos:
                continue

            node: Node = tree.create_node(tag=tok.text, identifier=idx,
                                          data=DUDESTreeCombData(token=DUDESToken(tok),
                                                                 dudes_candidates=[],
                                                                 lex_candidates=[],
                                                                 marker=marker),
                                          parent=parent)

            if tok.pos == "ADP":
                marker = tok.text

            for c in children[tok]:
                q.put((c, idx, marker))

            idx += 1

        return tree
    def preprocess(self, input: Any) -> Any:
        assert isinstance(input, str)
        doc = self.nlp(input)
        return [StanzaStep._dep_to_tree(doc)]