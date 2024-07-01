from __future__ import annotations

from typing import List, Optional, Generator, Any

import spacy
from treelib import Tree

from dudes import consts
from dudes.qa.preprocessing.preprocessing_step import PreprocessingStep, NumerizeStep, SpacyStepTrf, TreeStep, \
    SpacyStepLg, StanzaStep


class QuestionPreprocessor:
    def __init__(self, steps: List[List[PreprocessingStep]]):
        """
        Initialize question preprocessor with list of preprocessing pipelines. Each list in steps should start with a
        step accepting a string (i.e., the question) as input and end with a step returning a list of Tree objects as
        output. Each step in a pipeline gets the output of the previous step as input. Each list of steps in the
        parameter steps will be executed separately and the resulting tree lists will be merged into a single list of
        trees.
        :param steps: List of preprocessing pipelines. Each pipeline is a list of preprocessing steps.
        :type steps: List[List[PreprocessingStep]]
        """
        self.steps = steps

    @classmethod
    def default(cls,
                dbpedia_spotlight_endpoint: str = consts.dbpedia_spotlight_endpoint,
                nlp: Optional[spacy.language.Language] = None):
        numerizer = NumerizeStep()
        spacytrf = SpacyStepTrf(dbpedia_spotlight_endpoint=dbpedia_spotlight_endpoint, nlp=nlp)
        spacytrfmerge = SpacyStepTrf(dbpedia_spotlight_endpoint=dbpedia_spotlight_endpoint, nlp=nlp,
                                     merge_noun_chunks=True, merge_entities=True)
        spacylg = SpacyStepLg(dbpedia_spotlight_endpoint=dbpedia_spotlight_endpoint, nlp=nlp)
        spacylgmerge = SpacyStepLg(dbpedia_spotlight_endpoint=dbpedia_spotlight_endpoint, nlp=nlp,
                                   merge_noun_chunks=True, merge_entities=True)
        stanza = StanzaStep()
        tree = TreeStep()
        return QuestionPreprocessor([
            [
                numerizer,
                spacytrf,
                tree
            ],
            [
                numerizer,
                spacylg,
                tree
            ],
            [
                numerizer,
                stanza
            ],
            [
                spacytrf,
                tree
            ],
            [
                spacylg,
                tree
            ],
            [
                stanza
            ],
            [
                numerizer,
                spacytrfmerge,
                tree
            ],
            [
                numerizer,
                spacylgmerge,
                tree
            ],
            [
                spacytrfmerge,
                tree
            ],
            [
                spacylgmerge,
                tree
            ],
        ])

    def preprocess(self, question: str) -> Generator[Tree, None, None]:
        #full_res: List[Tree] = []

        for pipeline in self.steps:
            res: Any = question
            for step in pipeline:
                res = step.preprocess(res)
            #assert isinstance(res, List) and all([isinstance(t, Tree) for t in res])
            #full_res.extend(res)
            yield from res

        #return full_res
