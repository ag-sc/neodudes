import csv
import itertools
import logging
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional, Set
import re

from Levenshtein import distance as levenshtein_distance
from intervaltree import Interval, IntervalTree
import spacy
from marisa_trie import BytesTrie
import compress_pickle as cpl

from difflib import SequenceMatcher

import string

from dudes import utils


@dataclass(frozen=True)
class TaggerData:
    input_str: str
    uris: Tuple[str]
    _found_str: Optional[str] = None
    _intersection: Optional[str] = None
    sim: Optional[float] = None
    start: Optional[int] = None
    end: Optional[int] = None

    # def __init__(self,
    #              input_str: str,
    #              uris: List[str],
    #              found_str: Optional[str] = None,
    #              intersection: Optional[str] = None,
    #              sim: Optional[float] = None,
    #              start: Optional[int] = None,
    #              end: Optional[int] = None):
    #     self.input_str = input_str
    #     self._found_str = found_str if found_str is not None else input_str
    #     self._intersection = intersection if intersection is not None else input_str
    #     self.uris = uris
    #     self.sim = sim
    #     self.start = start
    #     self.end = end

    @property
    def found_str(self):
        return self._found_str if self._found_str is not None else self.input_str

    @property
    def intersection(self):
        return self._intersection if self._intersection is not None else self.input_str


class TrieLexicon:

    def __init__(self):
        # spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")  # consts.spacy_model)
        self.root = None
        self.rroot = None #datrie.Trie(string.printable)

    def tokenize(self, entry):
        doc = self.nlp(entry, disable=['parser', 'tagger', 'ner'])
        return [token for token in doc]

    def insert(self, entry: str, uri: str):
        if self.root is not None:
            new_trie = BytesTrie(itertools.chain(self.root.items(), [(entry, bytes(uri, 'utf-8'))]))
            self.root = new_trie
        else:
            new_trie = BytesTrie([(entry, bytes(uri, 'utf-8'))])
            self.root = new_trie

        if self.rroot is not None:
            new_rtrie = BytesTrie(itertools.chain(self.rroot.items(), [(entry[::-1], bytes(uri, 'utf-8'))]))
            self.rroot = new_rtrie
        else:
            new_rtrie = BytesTrie([(entry[::-1], bytes(uri, 'utf-8'))])
            self.rroot = new_rtrie

    def load(self, entries: Iterable[Tuple[str, str]]):
        entries = [(entry.lower(), uri) for entry, uri in entries]
        if self.root is not None:
            new_trie = BytesTrie(
                itertools.chain(self.root.items(), [(entry, bytes(uri, 'utf-8')) for entry, uri in entries]))
            self.root = new_trie
        else:
            new_trie = BytesTrie([(entry, bytes(uri, 'utf-8')) for entry, uri in entries])
            self.root = new_trie

        logging.info("Finished loading trie")

        if self.rroot is not None:
            new_rtrie = BytesTrie(
                itertools.chain(self.rroot.items(), [(entry[::-1], bytes(uri, 'utf-8')) for entry, uri in entries]))
            self.rroot = new_rtrie
        else:
            new_rtrie = BytesTrie([(entry[::-1], bytes(uri, 'utf-8')) for entry, uri in entries])
            self.rroot = new_rtrie

        logging.info("Finished loading reverse trie")
        # for entry, uri in entries:
        #    [entry[::-1]] = uri

    def load_from_file(self, path: str):
        with open(path, "rb") as f:
            self.root, self.rroot = cpl.load(f, compression="lzma")

    def save_to_file(self, path: str):
        with open(path, "wb") as f:
            cpl.dump((self.root, self.rroot), f, compression="lzma")

    def lookup(self, candidate):
        res = []
        tokenized_candidate = self.tokenize(candidate)
        for i in range(0, len(tokenized_candidate)):
            for j in range(i + 1, len(tokenized_candidate) + 1):
                rstr = self.reconstruct_str(tokenized_candidate[i:j])
                rrstr = rstr[::-1]
                if self.root is not None and rstr in self.root:
                    res.append(TaggerData(input_str=rstr, uris=tuple([buri.decode("utf-8") for buri in self.root[rstr]])))

                # Exact match does not need reverse lookup
                # if rrstr in self.rroot:
                #     res.append((rstr, [buri.decode("utf-8") for buri in self.root[rrstr]]))
        return res

    def lookup_approx(self, candidate, threshold: float = 0.0):
        res = []
        tokenized_candidate = self.tokenize(candidate)
        for i in range(0, len(tokenized_candidate)):
            for j in reversed(range(i + 1, len(tokenized_candidate) + 1)):
                if len([t for t in tokenized_candidate[i:j] if not t.is_punct and not t.is_stop]) > 0:
                    rstr = self.reconstruct_str(tokenized_candidate[i:j])
                    rrstr = rstr[::-1]
                    if self.root is not None:
                        # sublist = itertools.chain(*[self.root.items(el) for el in self.root.prefixes(rstr)])

                        res.extend([
                            TaggerData(
                                input_str=rstr,
                                _found_str=element,
                                _intersection=self.str_intersection(rstr, element),
                                uris=tuple([buri.decode("utf-8") for buri in self.root[element]]),
                                sim=self.normalized_sim(rstr, element)
                            )
                            for element in self.root.prefixes(rstr)
                            if self.normalized_sim(rstr, element) >= threshold
                        ])

                        # for element in self.root.prefixes(rstr):
                        #     sim = self.normalized_sim(rstr, element)
                        #     if sim >= threshold:
                        #         res.append((rstr,
                        #                     element,
                        #                     self.str_intersection(rstr, element),
                        #                     [buri.decode("utf-8") for buri in self.root[element]],
                        #                     sim))

                        res.extend([
                            TaggerData(
                                input_str=rstr,
                                _found_str=element,
                                _intersection=self.str_intersection(rstr, element),
                                uris=tuple([uri.decode("utf-8")]),
                                sim=self.normalized_sim(rstr, element)
                            )
                            for element, uri in self.root.items(rstr)
                            if self.normalized_sim(rstr, element) >= threshold
                        ])

                        # for element, uri in self.root.items(rstr):
                        #     sim = self.normalized_sim(rstr, element)
                        #     if sim >= threshold:
                        #         res.append((rstr,
                        #                     element,
                        #                     self.str_intersection(rstr, element),
                        #                     [uri.decode("utf-8")],
                        #                     sim))

                    if self.rroot is not None:
                        res.extend([
                            TaggerData(
                                input_str=rstr,
                                _found_str=element[::-1],
                                _intersection=self.str_intersection(rstr, element[::-1]),
                                uris=tuple([uri.decode("utf-8")]),
                                sim=self.normalized_sim(rstr, element[::-1])
                            )
                            for element, uri in self.rroot.items(rrstr)
                            if self.normalized_sim(rstr, element[::-1]) >= threshold
                        ])

                    # rsublist = self.rroot.items(rrstr)
                    # # print(rsublist)
                    # for element, uri in rsublist:
                    #     sim = self.normalized_sim(rstr, element[::-1])  # rrstr reversed is rstr, but element is still reversed -> more intuitive than rrstr and element comparison when debugging
                    #     if sim >= threshold:
                    #         res.append((rstr,
                    #                     element[::-1],
                    #                     self.str_intersection(rstr, element[::-1]),
                    #                     [uri.decode("utf-8")],
                    #                     sim))  # [buri.decode("utf-8") for buri in res]

        res = list(sorted(res, key=lambda x: (x.sim, len(x.intersection)), reverse=True))
        return res

    @staticmethod
    def reconstruct_str(tokens) -> str:
        return "".join([i.text + i.whitespace_ if i != tokens[-1] else i.text for i in tokens])
        # res = sorted(tokens, key=lambda x: x.idx)
        # idx_offset = res[0].idx
        # excl_end_idx = res[-1].idx + len(res[-1].text)
        #
        # rstr = " " * (excl_end_idx - idx_offset)
        # for t in res:
        #     rstr = rstr[:t.idx - idx_offset] + t.text + rstr[t.idx - idx_offset + len(t.text):]
        #
        # # rstr = re.sub(' +', '_', rstr.strip())
        # return rstr

    @staticmethod
    def str_intersection(str1, str2):#TODO: fix! 'the pillars of the earth' not working
        match = SequenceMatcher(None, str1, str2).find_longest_match()
        return str1[match.a:match.a + match.size]# ''.join(sorted(set(str1) & set(str2), key=str1.index))

    @staticmethod
    def normalized_sim(string1, string2):
        return utils.levenshtein_sim_normalized(string1, string2)
        #return 1 - levenshtein_distance(string1, string2) / max(len(string1), len(string2))


class TrieTagger:

    def __init__(self):
        self.lexicon = TrieLexicon()

    def load_from_file(self, path: str):
        self.lexicon.load_from_file(path)

    def save_to_file(self, path: str):
        self.lexicon.save_to_file(path)

    def items(self):
        for k, v in self.lexicon.root.items():
            yield k, v.decode("utf-8")

    def load(self, file):
        counter = 0
        filterex = re.compile("^[a-zA-Z\d+\-\s]{2,}")

        data = []

        with open(file, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                if re.search(filterex, row[0]):
                    data.append((row[0], row[1]))
                    counter += 1
                    # if counter % 1000000 == 0:
                    #    break

        logging.info("Data loaded")

        self.lexicon.load(data)
        # self.lexicon.print()

    def tag(self, tag_str, threshold):
        tag_str = tag_str.lower()
        logging.info("Tagging:  " + tag_str)
        result_filter = IntervalTree()
        result: List[Set] = [] * len(tag_str)
        for i in range(0, len(tag_str)):
            result.append(set())

        results_raw = []
        cands = self.lexicon.lookup_approx(tag_str, threshold)
        for element in cands:
            entry = element.input_str
            matched = element.found_str
            intersec = element.intersection
            uri = element.uris
            sim = element.sim
            start = tag_str.find(intersec)
            end = start + len(intersec)

            td = TaggerData(input_str=entry,
                            _found_str=matched,
                            _intersection=intersec,
                            uris=tuple(uri),
                            sim=sim,
                            start=start,
                            end=end)

            if start == -1:
                continue

            cres = result_filter[start:end]

            existing_uris = set().union(*[result[i] for i in range(start, end)])
            if len(set(uri).difference(existing_uris)) > 0:
                # if (not any([iv.contains_interval(Interval(start, end)) for iv in cres]) or
                #         any([(iv.begin == start and iv.end == end) for iv in cres])):
                #     # len(result.overlap(start, start + len(entry))) == 0
                result_filter[start:end] = True
                for i in range(start, end):
                    result[i].add(td)

            # cres = result[start:end]
            #
            # existing_uris = sum([r.data.uris for r in cres], [])
            # if len(set(uri).difference(existing_uris)) > 0:
            #     if (not any([iv.contains_interval(Interval(start, end)) for iv in cres]) or
            #             any([(iv.begin == start and iv.end == end) for iv in cres])):
            #         # len(result.overlap(start, start + len(entry))) == 0
            #         result[start:end] = td



            results_raw.append(td)
        return result, results_raw
