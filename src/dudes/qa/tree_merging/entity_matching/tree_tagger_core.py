import csv
import re
import string
from abc import ABC, abstractmethod
from typing import Optional, Dict, List

import spacy
from bidict import bidict
from Levenshtein import distance as levenshtein_distance
from marisa_trie import Trie, RecordTrie
from spacy import Language

from dudes import consts


def normalized_distance(string1, string2):
    return 1 - levenshtein_distance(string1, string2) / max(len(string1), len(string2))


class IDMap(ABC):
    def __init__(self):
        self.next_free_id = 0

    @property
    def next_id(self) -> int:
        res = self.next_free_id
        self.next_free_id += 1
        return res

    @abstractmethod
    def inverse(self, idx: int) -> str:
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def add_or_get_entry_id(self, entry: str) -> int:
        pass


class DictIDMap(IDMap):
    def __init__(self, id_map: Optional[Dict[str, int]] = None):
        super().__init__()
        self.id_map: Dict[str, int]
        if id_map is None:
            self.id_map = dict()
        else:
            self.id_map = id_map

    def inverse(self, idx: int):
        res = [k for k, v in self.id_map.items() if v == idx]
        if len(res) == 1:
            return res[0]
        else:
            raise KeyError(f"None or multiple entries for {idx}.")

    def __getitem__(self, item):
        self.id_map.__getitem__(item)

    def __setitem__(self, key, value):
        self.id_map.__setitem__(key, value)

    def add_or_get_entry_id(self, entry: str) -> int:
        if entry not in self.id_map.keys():
            newid = self.next_id
            self.id_map[entry] = newid
            return newid
        else:
            return self.id_map[entry]

    @classmethod
    def from_trie_map(cls, tm):
        if isinstance(tm, DictIDMap):
            return tm
        else:
            return cls(id_map={k: tm[k] for k in tm.trie.keys()})


class TrieIDMap(IDMap):
    def __init__(self, trie: Optional[Trie] = None): #, id_map: Optional[bidict[int, int]] = None
        super().__init__()
        # self.id_map: bidict
        # if id_map is None:
        #     self.id_map = bidict()
        # else:
        #     self.id_map = id_map

        self.id_map: Trie
        if trie is None:
            self.id_map = Trie([])
        else:
            self.id_map = trie

        #assert len(self.id_map) == len(self.trie.keys())

    def inverse(self, idx: int):
        return self.id_map.restore_key(idx)

    def __getitem__(self, item):
        return self.id_map[item]

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def add_or_get_entry_id(self, entry: str) -> int:
        if entry not in self.id_map.keys():
            raise NotImplementedError()

        return self.id_map[entry]

    @classmethod
    def from_dict_map(cls, dm):
        if isinstance(dm, TrieIDMap):
            return dm
        else:
            trie = Trie(dm.id_map.keys())
            id_map = bidict([(trie[k], v) for k, v in dm.id_map.items()])
            return cls(trie=trie), id_map


class MapperStore:
    def __init__(self, mapper: IDMap):
        self.mapper: IDMap = mapper


class TreeLexiconNode:

    def __init__(self, id_map: MapperStore):
        self.id_map: MapperStore = id_map
        self.children: Dict = dict()
        self.map: Dict = dict()

    @property
    def mapper(self):
        return self.id_map.mapper

    def print(self, indent):

        for child_idx in self.children:
            child = self.mapper.inverse(child_idx)
            print(indent + child)
            self.children[child].print(indent + "\t")
        for entry in self.map:
            for uri in self.map[entry]:
                print(indent + "\t" + self.mapper.inverse(entry) + " " + self.mapper.inverse(uri))

    def lookup(self, tokenized_list, i, matched_tokens):
        res = []
        matched_tokens_new = matched_tokens
        if i < len(tokenized_list):
            tl_idx = self.mapper[tokenized_list[i]]
            if tl_idx in self.children:
                child = self.children[tl_idx]
                matched_tokens_new.append(tokenized_list[i])
                sublist = child.lookup(tl_idx, i + 1, matched_tokens_new)
                for element in sublist:
                    res.append(element)

        for entry in self.map:
            for uri in self.map[entry]:
                # print("Adding "+tpl[1]+" for "+" ".join(matched_tokens))
                res.append((" ".join(matched_tokens), self.mapper.inverse(entry), self.mapper.inverse(uri)))

        return res

    def lookup_approx(self, tokenized_list, i, matched_tokens, threshold):
        res = []
        if i < len(tokenized_list):
            # print("Looking up in node: "+tokenized_list[i])
            for entry_idx in self.children:
                entry = self.mapper.inverse(entry_idx)
                normLD = normalized_distance(entry, tokenized_list[i])
                # print("Normalized Levensthein distance between "+entry+" and "+tokenized_list[i]+"= "+str(normLD))

                if normLD >= threshold:
                    # print(entry+" matches "+tokenized_list[i])
                    # print("There are "+str(len(self.children))+" children")
                    matched_tokens_new = matched_tokens.copy()
                    matched_tokens_new.append(tokenized_list[i])
                    child = self.children.get(entry_idx)
                    if child is not None:
                        sublist = child.lookup_approx(tokenized_list, i + 1, matched_tokens_new, threshold)
                        for element in sublist:
                            res.append(element)

        for entry in self.map:
            for uri in self.map[entry]:
                # print("Adding "+tpl[1]+" for "+" ".join(matched_tokens))
                res.append((" ".join(matched_tokens), self.mapper.inverse(entry), self.mapper.inverse(uri)))

        return res

    def insert(self, entry, tokenized_entry, uri, i):
        # print("Inserting: "+entry+"\t"+uri+"\t"+type)
        if i < len(tokenized_entry):
            if tokenized_entry[i] in self.children:
                child = self.children[tokenized_entry[i]]
                child.insert(entry, tokenized_entry, uri, i + 1)
            else:
                child = TreeLexiconNode(id_map=self.id_map)
                child.insert(entry, tokenized_entry, uri, i + 1)
                self.children[tokenized_entry[i]] = child

        if len(tokenized_entry) == i:
            if not entry in self.map:
                self.map[entry] = []
            self.map[entry].append(uri)

    def replace_ids(self, id_map):
        new_map = dict()
        for entry in self.map:
            new_map[id_map.inverse[entry]] = [id_map.inverse[uri] for uri in self.map[entry]]
        self.map = new_map

        new_child = dict()
        for entry in self.children:
            children = self.children[entry]
            children.replace_ids(id_map)
            new_child[id_map.inverse[entry]] = children

        self.children = new_child



class TreeLexicon:

    def __init__(self, id_map: MapperStore):
        self.id_map: MapperStore
        self.id_map = id_map
        #spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")#consts.spacy_model)
        self.root = TreeLexiconNode(id_map=self.id_map)


    def tokenize(self, entry):
        doc = self.nlp(entry, disable=['parser', 'tagger', 'ner'])
        return [token.text for token in doc]

    def insert(self, entry, uri):
        tokenized_entry = self.tokenize(entry)

        self.root.insert(self.id_map.mapper.add_or_get_entry_id(entry),
                         [self.id_map.mapper.add_or_get_entry_id(te) for te in tokenized_entry],
                         self.id_map.mapper.add_or_get_entry_id(uri),
                         0)

    def lookup(self, candidate):
        res = []
        tokenized_candidate = self.tokenize(candidate)
        for i in range(0, len(tokenized_candidate)):
            sublist = self.root.lookup(tokenized_candidate, i, [])
            for element in sublist:
                res.append(element)
        return res

    def lookup_approx(self, candidate, threshold):
        print("Looking up: " + candidate)
        res = []
        tokenized_candidate = self.tokenize(candidate)
        for i in range(0, len(tokenized_candidate)):
            sublist = self.root.lookup_approx(tokenized_candidate, i, [], threshold)
            for element in sublist:
                res.append(element)
        return res

    def replace_ids(self, id_map):
        self.root.replace_ids(id_map)

    def print(self):
        self.root.print("")


class TreeTagger:

    def __init__(self):
        self.lexicon = TreeLexicon(id_map=MapperStore(mapper=DictIDMap()))

    def load(self, file):
        counter = 0
        self.lexicon.id_map.mapper = DictIDMap.from_trie_map(self.lexicon.id_map.mapper)
        filterex = re.compile("^[a-zA-Z\d+\-\s]{2,}")
        with open(file, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                if re.search(filterex, row[0]):
                    self.lexicon.insert(row[0], row[1])
                    counter += 1
                    #if counter % 1000000 == 0:
                    #    break
        self.lexicon.id_map.mapper, id_translation = TrieIDMap.from_dict_map(self.lexicon.id_map.mapper)
        self.lexicon.replace_ids(id_translation)
        # self.lexicon.print()

    def tag(self, tag_str, threshold):
        print("Tagging:  " + tag_str)
        result = []
        for element in self.lexicon.lookup_approx(tag_str, threshold):
            entry = element[0]
            uri = element[1]
            lex_type = element[2]
            start = tag_str.find(entry)
            result.append((entry, uri, lex_type, start, start + len(entry)))
        return result
