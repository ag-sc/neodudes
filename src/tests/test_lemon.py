import json
import os
import sys
from pprint import pprint

from dudes.dot import DotFile
from lemon.csv_to_lemon import csv_to_lemon
from lemon.lemon_parser import LEMONParser
from lemon.lemon_class_generator import LEMONDataClassGenerator, Pruned
from lemon.lemon import *
from lemon.lexicon import Lexicon


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj == Pruned.PRUNED:
            return "PRUNED"
        if isinstance(obj, set):
            if len(obj) == 1:
                return list(obj)[0]
            else:
                return list(obj)
        return json.JSONEncoder.default(self, obj)


def test_lemonparser():
    print("")
    klass = globals()["LexicalEntry"]
    print(klass())
    lp = LEMONParser.from_ttl_dir()
    print(lp.classtypes)
    entries = lp.entry_nodes
    for entry in entries:
        res = lp.parse_node(entry)
        print(res)
    # d = lp.to_dict(entries[127])
    # dvd = lp.cg.diff_val_dict()
    # flatdvd = lp.flatten_dict({"lemon:LexicalEntry": dvd})
    # print(json.dumps(flatdvd, sort_keys=True, indent=4, cls=SetEncoder))
    # print(lp.gen_classes())


def test_lemonparser2():
    print("")
    lp = LEMONParser.from_ttl_dir()
    dvd, lvmap = lp.cg._diff_val_dict()
    pprint(lvmap)
    # relLVs = [(k,v) for k, v in lvmap.items() if "lemon" in k]
    flatdvd = lp.cg._flatten_dict({"lemon:LexicalEntry": dvd})
    print(json.dumps(flatdvd, sort_keys=True, indent=4, cls=SetEncoder))
    pflatdvd = lp.cg._prune_flat_dict(flatdvd)
    #print(json.dumps(pflatdvd, sort_keys=True, indent=4, cls=SetEncoder))


def test_genlemonpy():
    print("")
    lp = LEMONParser.from_ttl_dir()
    cg = LEMONDataClassGenerator(lp.graph, lp.base_uri)
    with open("../lemon/lemon_new.py", "w") as file:
        classes, typedict = cg.gen_classes()
        file.write(classes)
        print(typedict)

def test_classgraph():
    print("")
    lp = LEMONParser.from_ttl_dir()
    cg = LEMONDataClassGenerator(lp.graph, lp.base_uri)
    dot = cg._gen_class_dot(skip_consts=True)
    dot.write_pdf("classgraph.pdf")
    dot.write_png("classgraph.png")
    DotFile.runXDot(dot.to_string(), run=False)
    dot = cg._gen_class_dot(skip_consts=False)
    dot.write_pdf("classgraph_with_consts.pdf")
    dot.write_png("classgraph_with_consts.png")
    DotFile.runXDot(dot.to_string())


def test_checklemon():
    # klass = globals()["LexicalEntry"]
    # print(klass())
    lp = LEMONParser.from_ttl_dir()
    # print(lp.classtypes)
    entries = lp.entry_nodes
    # forms: list[str | None] = []

    error_counter = 0
    errors = []

    for entry in entries:
        # try:
        # if str(entry) != 'http://localhost:8000/lexicon#borough-of':
        #    continue
        res: LexicalEntry = lp.parse_node(entry)
        # forms.append(res.canonical_form.written_rep)
        # except:
        #    print("Failed")
        try:
            if isinstance(res.canonical_form.written_rep, list) and len(res.canonical_form.written_rep) > 1:
                # print(res.uri)
                errors.append(res.uri[30:])
                error_counter += 1
                # if res.uri[30:] == 'Book-in':
                # pprint(res)
            if len(res.sense) > 1:
                pprint(res)

            #elif res.syn_behavior is not None and res.syn_behavior.prepositional_adjunct is not None and res.syn_behavior.prepositional_adjunct.marker is not None and len(
            #        res.syn_behavior.prepositional_adjunct.marker) > 1:
            #    errors.append(res.uri[30:])
            #    error_counter += 1
            #    pprint(res)
            # elif any([len(s.reference) > 0 for s in res.sense]):
            #    pprint(res)
        except:
            continue
    print(error_counter)
    print(errors)
    # print(forms)
    # res = lp.parse_node(entries[0], )
    # print(res.__str__())


def test_lexicon():
    lp = LEMONParser.from_ttl_dir()
    entry_nodes = lp.entry_nodes
    entries = lp.parse_nodes(entry_nodes)
    lex = Lexicon(entries)
    adjectives = [e for e in entries if e.part_of_speech == PartOfSpeech.ADJECTIVE and not e.uri.endswith("ed")]
    verb = [e for e in entries if e.part_of_speech == PartOfSpeech.VERB]
    noun = [e for e in entries if e.part_of_speech == PartOfSpeech.NOUN]
    multiple_senses = [e for e in entries if len(e.sense) > 1]
    #pprint(adjectives)
    #pprint(verb)
    #pprint(noun)
    print(len(multiple_senses))
    pprint(multiple_senses)
    pass

def test_csv():
    path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "entityLists", "industry.csv")
    res = list(csv_to_lemon(path))
    print(res)

