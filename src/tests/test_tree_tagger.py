import cProfile
import os
import sys
import compress_pickle as cpl

import dudes.qa.dudes_rpc_service
import lemon
from dudes import utils, consts
from dudes.qa.dudes_rpc_service import TrieTaggerWrapper
#from guppy import hpy

from dudes.qa.tree_merging.entity_matching.tree_tagger_core import TreeTagger
from dudes.qa.tree_merging.entity_matching.trie_tagger import TrieTagger


#

def test_tagging():
    tagger_thread = dudes.qa.dudes_rpc_service.start_rpc_service()
    print("Trie tagger started.")
    tagger = TrieTaggerWrapper(consts.rpc_host, consts.rpc_port)
    text = "What is Angela Merkel's birth name?"
    tags, tags_raw = tagger.tag(text, 0.5)
    tags = [sorted(tag, key=lambda x: x.sim, reverse=True) for tag in tags]
    print(tags)


def test_tag():
    tagger = TreeTagger()

    print("Loading lexicon")

    path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels.tsv")
    path_out = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels_tree_tagger.cpl")
    if os.path.isfile(path_out):
        with open(path_out, "rb") as f:
            tagger = cpl.load(f, compression="lzma")
    else:
        #profiler = cProfile.Profile()
        #profiler.enable()
        tagger.load(path)
        #profiler.disable()
        #profiler.dump_stats('treetagger.prof')

        with open(path_out, "wb") as f:
            cpl.dump(tagger, f, compression="lzma")

    print("Lexicon loaded")
    #h = hpy()
    #heap = h.heap()
    #print(heap.all)

    profiler = cProfile.Profile()
    profiler.enable()
    sentence = "Who wrote The Lord of the Rings is a very cool book about peace?"
    #sentence = input("Enter the sentence to tag! (Type exit to stop)")
    result, results_raw = tagger.tag(sentence,0.5)
    profiler.disable()
    profiler.dump_stats('tree-tagger.prof')
    print(result)
    print(results_raw)


    # while sentence != "exit":
    #     sentence = input("Enter the sentence to tag! (Type exit to stop)")
    #     result = tagger.tag(sentence,0.9)
    #     print(result)

def test_trie_add_data():
    tagger = TrieTagger()

    print("Loading lexicon")

    path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels_en_uris_fr.tsv")
    path_out = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels_trie_tagger.cpl")
    path_out_new = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels_trie_tagger_fr.cpl")
    if os.path.isfile(path_out):
        tagger.load_from_file(path_out)
        tagger.load(path)
        tagger.save_to_file(path_out_new)

def test_trie_tag():
    tagger = TrieTagger()

    print("Loading lexicon")

    path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels.tsv")
    path_out = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels_trie_tagger_fr.cpl")
    if os.path.isfile(path_out):
        tagger.load_from_file(path_out)
        #with open(path_out, "rb") as f:
        #    tagger = cpl.load(f, compression="lzma")
    else:
        #profiler = cProfile.Profile()
        #profiler.enable()
        tagger.load(path)
        #profiler.disable()
        #profiler.dump_stats('treetagger.prof')

        tagger.save_to_file(path_out)

        #with open(path_out, "wb") as f:
        #    cpl.dump(tagger, f, compression="lzma")

    print("Lexicon loaded")
    #h = hpy()
    #heap = h.heap()
    #print(heap.all)

    #profiler = cProfile.Profile()
    #profiler.enable()
    #sentence = "What is Prodigy?"
    #sentence = "The pillars of the Earth"
    sentence = "Les Piliers de la terre"
    #sentence = input("Enter the sentence to tag! (Type exit to stop)")
    result, result_raw = tagger.tag(sentence,0.8)
    #profiler.disable()
    #profiler.dump_stats('trie-tagger.prof')

    #print(sorted([i.data for i in result.items()], key=lambda x: x.sim, reverse=True))

    print("-------")
    print(result)
    print("#######")
    print(result_raw)


    # while sentence != "exit":
    #     sentence = input("Enter the sentence to tag! (Type exit to stop)")
    #     result = tagger.tag(sentence,0.9)
    #     print(result)

if __name__ == "__main__":
    test_trie_tag()