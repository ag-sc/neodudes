import cProfile
import csv
import itertools
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Set

import more_itertools
from sqlitedict import SqliteDict

import dudes.qa.dudes_rpc_service
from dudes import utils
from dudes.qa.qa_pipeline import QAPipeline
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from dudes.qa.tree_merging.entity_matching.trie_tagger import TrieTagger
from dudes.utils import compare_results, EvalStats

logging.basicConfig(level=logging.DEBUG, force=True)
from pprint import pprint

import spacy
import z3  # type: ignore

from SPARQLBurger.SPARQLQueryBuilder import *  # type: ignore

from dudes.consts import spacy_model
from dudes.dudes_token import DUDESToken
#from old.ontologies.comb_lemon_ontology import CombLEMONOntology
from dudes.qa.sparql.sparql_generator import BasicSPARQLGenerator
from lemon.lemon_parser import LEMONParser
from lemon.lexicon import Lexicon

import compress_pickle as cpl  # type: ignore


#from dudes.ontologies.lemon_ontology import LEMONOntology
#from old.dudes_parser import DUDESParser


# def test_dudes():
#     cond = z3.String("x") == z3.StringVal("Peter")
#     drs1 = DRS(variables={z3.String("x")}, conditions={cond})
#     dudes = DUDES(main_variable=z3.String("x"),
#                   main_label="l",
#                   drs={"l": drs1},
#                   selection_pairs=set(),
#                   sub_relations={SubRelation(left="l", right=None, rightType=SubRelation.Type.TOP)})
#     print(dudes)
#
#
# def test_dudes2():
#     f = z3.Function("cool", z3.StringSort(), z3.BoolSort())
#     x = z3.String("x")
#     cond = f(x) == True
#     drs1 = DRS(variables={x}, conditions={cond})
#     dudes = DUDES(main_variable=x,
#                   main_label="l",
#                   drs={"l": drs1},
#                   selection_pairs={SelectionPair(variable=x, label="l")},  # allowed_pos=frozenset(["NOUN", "PROPN"]),
#                   sub_relations=set())
#     print(dudes)
#
#
# def test_dudesparser():
#     print("")
#     # nlp = spacy.load("en_core_web_lg")
#     # nlp.add_pipe('dbpedia_spotlight')
#
#     # doc = nlp("The quick fox, which is working in the city, has jumped over the lazy dog Peter. Mr. Smith walks into a bar.")
#     # doc = nlp("The quick fox plays golf.")
#     # roots = [token for token in doc if token.dep_ == "ROOT"]
#     lp = LEMONParser.from_ttl_dir()
#     entries = lp.parse_nodes(lp.entry_nodes)
#     lex = Lexicon(entries)
#     oc = LEMONOntology(lexicon=lex)
#
#     dp = DUDESParser.from_str("The quick fox, which is working in the city, has jumped over the lazy dog Peter.",
#                               oc=BasicOntology())
#                               #oc=oc)  # BasicOntology()
#     dp.tree.show(line_type="ascii-em", data_property="desc")
#     dp.merge_dudes(debug=True)
#     # print(str(list(dp.tree.nodes.values())[0].data.dudes))
#     # dp.tree.show(line_type="ascii-em", data_property="dudes")
#
#
# def test_dudesparser2():
#     print("")
#
#     lp = LEMONParser.from_ttl_dir()
#     entries = lp.parse_nodes(lp.entry_nodes)
#     lex = Lexicon(entries)
#     oc = LEMONOntology(lexicon=lex)
#     # oc = BasicOntology()
#     dp = DUDESParser.from_str("The book Das Kapital by old Gary F. Marcus is written in German.", oc=oc)
#     dp.tree.show(line_type="ascii-em", data_property="desc")
#     dp.merge_dudes(debug=True)
#     res_dudes = list(dp.tree.nodes.values())[0].data.dudes
#     print(str(res_dudes))
#     print(list(res_dudes.drs.values())[0].pred_var_dict)
#     print(str(res_dudes.str_formula))
#
#
# def test_dudesparser3():
#     print("")
#
#     lp = LEMONParser.from_ttl_dir()
#     entries = lp.parse_nodes(lp.entry_nodes)
#     lex = Lexicon(entries)
#     oc = LEMONOntology(lexicon=lex)
#     # oc = BasicOntology()
#     dp = DUDESParser.from_str("The book Das Kapital is written in German.", oc=oc)
#     dp.tree.show(line_type="ascii-em", data_property="desc")
#     dp.merge_dudes(debug=True)
#     res_dudes = list(dp.tree.nodes.values())[0].data.dudes
#     print(str(res_dudes))
#     print(list(res_dudes.drs.values())[0].pred_var_dict)
#
#
# def test_dudesparser4():
#     print("")
#
#     lp = LEMONParser.from_ttl_dir()
#     entries = lp.parse_nodes(lp.entry_nodes)
#     lex = Lexicon(entries)
#     oc = LEMONOntology(lexicon=lex)
#     # oc = BasicOntology()
#     dp = DUDESParser.from_str("Das Kapital is a book by old Gary F. Marcus.", oc=oc)
#     dp.tree.show(line_type="ascii-em", data_property="desc")
#     dp.merge_dudes(debug=True)
#     res_dudes = list(dp.tree.nodes.values())[0].data.dudes
#     print(str(res_dudes))
#     print(list(res_dudes.drs.values())[0].pred_var_dict)
#
# def test_dudesparser5():
#     print("")
#
#     lp = LEMONParser.from_ttl_dir()
#     entries = lp.parse_nodes(lp.entry_nodes)
#     lex = Lexicon(entries)
#     oc = LEMONOntology(lexicon=lex)
#     # oc = BasicOntology()
#     dp = DUDESParser.from_str("Who lives in the retirement home is old.", oc=oc)
#     dp.tree.show(line_type="ascii-em", data_property="desc")
#     dp.merge_dudes(debug=True)
#     res_dudes = list(dp.tree.nodes.values())[0].data.dudes
#     print(str(res_dudes))
#     print(list(res_dudes.drs.values())[0].pred_var_dict)
#
# def test_dudes_sparql():
#     print("")
#
#     lp = LEMONParser.from_ttl_dir()
#     entries = lp.parse_nodes(lp.entry_nodes)
#     lex = Lexicon(entries)
#     oc = LEMONOntology(lexicon=lex)
#     # oc = BasicOntology()
#     question = "Give me all actors starring in movies directed by and starring William Shatner."
#     dp = DUDESParser.from_str(question, oc=oc)
#     dp.tree.show(line_type="ascii-em", data_property="desc")
#     dp.merge_dudes(debug=True)
#     res_dudes = list(dp.tree.nodes.values())[0].data.dudes
#     #model = res_dudes.get_model()
#     print(res_dudes.assigned_variables)
#     print(res_dudes.pred_var_dict)
#     sg = BasicSPARQLGenerator(nsmanager=oc.nsmanager)
#     print(sg.to_sparql(question, res_dudes))
#     res = sg.get_results_query(sg.to_sparql(question, res_dudes))
#     print(res)

    # sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # sparql.setQuery(sg.to_sparql(res_dudes))
    # sparql.setReturnFormat(JSON)
    # try:
    #     ret = sparql.queryAndConvert()
    #     print(ret)
    #
    #     for r in ret["results"]["bindings"]:
    #         print(r)
    # except Exception as e:
    #     print(e)

# def test_dudes_sparql_comb():
#     print("")
#     start = time.time()
#     profiler = cProfile.Profile()
#     profiler.enable()
#     lp = LEMONParser.from_ttl_dir()
#     entries = lp.parse_nodes(lp.entry_nodes)
#     lex = Lexicon(entries)
#     oc = CombLEMONOntology(lexicon=lex)
#     # oc = BasicOntology()
#     #question = "Which countries have more than two official languages?"
#     #question = "Which mountains are higher than the Nanga Parbat?"
#
#     #question = "Which software has been developed by organizations founded in California?"
#     #question = "Give me all companies in the advertising industry."
#     #question = "What was the first Queen album?"
#
#
#     #question = "Which companies have more than 1 million employees?"
#     #question = "Who is the son of Sonny and Cher?"
#     #question = "How many inhabitants does the largest city in Canada have?"
#     #question = "In which U.S. state is Area 51 located?"
#     #question = "What other books have been written by the author of The Fault in Our Stars?"
#     question = "Who wrote the Lord of the Rings?"
#     #question = "Which Chess players died in the same place they were born in?"
#     #question = "Which countries have more than ten caves?"
#     #question = "Show me all basketball players that are higher than 2 meters."
#     #question = Give me all actors starring in movies directed by and starring William Shatner."
#     #question = "How deep is Lake Placid?"
#     #question = "Give me all islands that belong to Japan."
#
#     #Others:
#
#     #question = "Give me all albums with Elton John."
#     #question = "What airlines are part of the SkyTeam alliance?"
#     #question = "In which state Penn State University is located?"
#     #question = "Who is the daughter of Ingrid Bergman married to?"
#     #question = "How many languages are spoken in Colombia?"
#     #question = "List all boardgames by GMT."
#     #question = "Which country has the most official languages?"
#     #question = "Give me all actors who were born in Berlin."
#     dp = DUDESParser.from_str(question, oc=oc)
#     for t in dp.trees:
#         t.show(line_type="ascii-em", data_property="desc")
#     dp.merge_dudes(debug=False)
#     res_dudes = sum([list(t.nodes.values())[0].data.dudes_candidates for t in dp.trees], [])
#     sg = BasicSPARQLGenerator(nsmanager=oc.nsmanager)
#
#     for rd in res_dudes:
#         if time.time() - start > 200:
#             break
#         try:
#             query = sg.to_sparql(question, rd)
#             print(query)
#             print(sg.to_sparql(question, rd, include_redundant=True))
#             res = sg.get_results_query(query)
#             print(res)
#         except Exception as e:
#             print("Error:", e)
#
#     profiler.disable()
#     profiler.dump_stats('test_dudes.prof')
#
#     # sparql = SPARQLWrapper("http://dbpedia.org/sparql")
#     # sparql.setQuery(sg.to_sparql(res_dudes))
#     # sparql.setReturnFormat(JSON)
#     # try:
#     #     ret = sparql.queryAndConvert()
#     #     print(ret)
#     #
#     #     for r in ret["results"]["bindings"]:
#     #         print(r)
#     # except Exception as e:
#     #     print(e)

def test_qa_pipeline():
    #["115", "154", "195", "199", "229"]
    #
    # {'id': '115', 'answertype': 'resource', 'question': 'In which U.S. state is Mount McKinley located?', 'sparql': 'SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Mount_McKinley> dbo:wikiPageRedirects ?x . ?x <http://dbpedia.org/ontology/locatedInArea> ?uri. ?uri rdf:type yago:WikicatStatesOfTheUnitedStates }'}
    # {'id': '154', 'answertype': 'resource', 'question': 'What is the smallest city by area in Germany?', 'sparql': 'PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX dbc: <http://dbpedia.org/resource/Category:> PREFIX dct: <http://purl.org/dc/terms/> SELECT ?city WHERE { ?m skos:broader dbc:Cities_in_Germany . ?city dct:subject ?m ; dbo:areaTotal ?area } ORDER BY ?area LIMIT 1'}
    # {'id': '195', 'answertype': 'resource', 'question': 'Who wrote the book Les Piliers de la terre?', 'sparql': 'PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX res: <http://dbpedia.org/resource/> SELECT DISTINCT ?uri WHERE { res:The_Pillars_of_the_Earth dbo:author ?uri }'}
    # {'id': '199', 'answertype': 'resource', 'question': 'Which ingredients do I need for carrot cake?', 'sparql': 'PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX res: <http://dbpedia.org/resource/> SELECT DISTINCT ?uri WHERE { res:Carrot_cake dbo:ingredient ?uri }'}
    # {'id': '229', 'answertype': 'string', 'question': 'Who is 8th president of US?', 'sparql': 'PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX dbp: <http://dbpedia.org/property/> PREFIX dbr: <http://dbpedia.org/resource/> SELECT ?name WHERE { dbr:8th_President_of_the_United_States dbo:wikiPageRedirects ?link . ?link dbp:name ?name }'}

    #TODO: Error for ID: 259 Question: Who composed the soundtrack for Cameron's Titanic? '>' not supported between instances of 'NoneType' and 'int'
    #TODO: Error for ID: 174 Question: Who is the novelist of the work a song of ice and fire? 'DuplexCondition' object has no attribute 'str_formula'

    cachepath = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "..", "..", "qald-cache-parallel.cpkl")
    cache: Dict = defaultdict(dict)

    if os.path.isfile(cachepath):
        with open(cachepath, "rb") as f:
            cache = cpl.load(f, compression="lzma")

    tagger_thread = dudes.qa.dudes_rpc_service.start_rpc_service()
    print("Trie tagger started.")

    qa = QAPipeline.default(
        #tree_merger_candidate_limit=3,
        #dudes_composer_candidate_limit=3,
        cache=cache
    )
    #question = "In which U.S. state is Mount McKinley located?"
    #question = "What is the smallest city by area in Germany?"
    #question = "Who wrote the book Les Piliers de la terre?"
    #question = "Which presidents were born in 1945?"
    #question = "Give me all people that were born in Vienna and died in Berlin."
    #question = "Who is the daughter of Ingrid Bergman married to?"
    #question = "Was Marc Chagall a jew?"
    #question = "When was the De Beers company founded?"
    #question = "Give me all spacecrafts that flew to Mars."
    question = "Which computer scientist won an oscar?"
    question = "Give me all cars that are produced in Germany."
    question = "Which European countries have a constitutional monarchy?"
    question = "Which country was Bill Gates born in?"
    question = "Give me all libraries established before 1400."
    question = "Do Prince Harry and Prince William have the same parents?"
    question = "Give me all actors starring in movies directed by and starring William Shatner."
    question = "Show me all basketball players that are higher than 2 meters."

    #question = "Who wrote the Lord of the Rings?"
    # profiler = cProfile.Profile()
    # profiler.enable()

    #res = qa.process(question)

    # profiler.disable()
    # profiler.dump_stats('les_piliers.prof')

    res = qa.process(question)
    for q in res:
        if q.count("<http://dbpedia.org/ontology/starring>") > 2:
            pass
        print(q)

    # res = qa.process_and_fetch(question)
    # res_list = [(len(r), r, q) for q, r in res.items()]
    # res_list.sort(key=lambda x: x[0])
    # for lenr, r, q in res_list:
    #     print(q)
    #     print("Result:", len(r))


def test_qald():
    qid = 42
    qid = None
    question_search = "Bill Gates"
    #question_search = "all libraries established before 1400"
    question_search = "What is the highest volcano in Africa?"
    #question_search = "Who is the novelist of the work a song of ice and fire?"
    question_search = "Which countries have more than ten volcanoes?"
    question_search = "What is Angela Merkel’s birth name?"
    question_search = "Show me all Czech movies."
    question_search = "Which languages are spoken in Estonia?"
    question_search = "Who were the parents of Queen Victoria?"
    question_search = "Who was Tom Hanks married to?"
    question_search = "Whom did Lance Bass marry?"
    question_search = "Who is the governor of Texas?"
    question_search = "Butch Otter is the governor of which U.S. state?"
    question_search = "What is the time zone of Salt Lake City?"
    question_search = "Which actors play in Big Bang Theory?"
    question_search = "What is the largest state in the United States?"
    question_search = "Which bridges are of the same type as the Manhattan Bridge?"
    question_search = "Which countries have places with more than two caves?"
    question_search = "Which companies have more than 1 million employees?"
    question_search = "Show me all basketball players that are higher than 2 meters."



    experiment = None

    path = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        "QALD9_train-dataset-raw.csv" if experiment is None else f"QALD9_train-dataset-raw-{experiment}.csv"
    )

    path_test = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        "QALD9_test-dataset-raw.csv" if experiment is None else f"QALD9_test-dataset-raw-{experiment}.csv"
    )

    rows = []

    with open(path) as csv_file:
        csv_dict = csv.DictReader(csv_file, delimiter=',')
        rows = list(csv_dict)

    with open(path_test) as csv_file:
        csv_dict = csv.DictReader(csv_file, delimiter=',')
        rows = rows + list(csv_dict)

    if qid is not None:
        rows = [r for r in rows if int(r["id"]) == qid]
    if question_search is not None:
        rows = [r for r in rows if question_search.lower() in r["question"].lower()]

    if len(rows) > 1:
        print("Multiple questions found:")
        for r in rows:
            print(r)
        return
    elif len(rows) == 0:
        print("No questions found.")
        return

    row = rows[0]
    print("Selected:", row)

    cachepath = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "..", "..", "qald-cache-parallel.cpkl")
    cache: Dict = defaultdict(dict)

    if os.path.isfile(cachepath):
        with open(cachepath, "rb") as f:
            cache = cpl.load(f, compression="lzma")

    tagger_thread = dudes.qa.dudes_rpc_service.start_rpc_service()
    print("Trie tagger started.")

    qa = QAPipeline.default(
        # tree_merger_candidate_limit=3,
        # dudes_composer_candidate_limit=3,
        cache=cache
    )

    sg = qa.sparql_generator

    gold_res = sg.get_results_query(row["sparql"])

    gold = utils.sanitize_sparql_result(gold_res)

    best_query = None
    best_stats = EvalStats(fn=len(gold))

    res = qa.process(row["question"])

    path_out = f"QALD9_queries_{row['id']}.csv"
    fieldnames = ['ID', "Query Number", 'Question', 'SPARQL', "True Positive", "False Positive", "False Negative",
                  "Precision", "Recall", "F1", "Exact matches"]

    with open(path_out, "w", newline='') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            "ID": row["id"],
            "Query Number": "Gold",
            "Question": row["question"],
            "SPARQL": utils.remove_prefix(row["sparql"]),
            "True Positive": len(gold),
            "False Positive": 0,
            "False Negative": 0,
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0,
            "Exact matches": 1
        })
        out_file.flush()

        for idx, query in enumerate(res):
            print("Query:", idx, flush=True)
            try:
                sys_res = sg.get_results_query(query)
            except Exception as e:
                print("Error:", e)
                continue

            curr_stats = compare_results(gold_res=gold_res, sys_res=sys_res)

            writer.writerow({
                "ID": row["id"],
                "Query Number": idx,
                "Question": row["question"],
                "SPARQL": utils.remove_prefix(query),
                "True Positive": curr_stats.tp,
                "False Positive": curr_stats.fp,
                "False Negative": curr_stats.fn,
                "Precision": curr_stats.precision,
                "Recall": curr_stats.recall,
                "F1": curr_stats.f1,
                "Exact matches": curr_stats.emc
            })
            out_file.flush()

            #print(row)
            print(utils.remove_prefix(query))
            print(curr_stats)
            if curr_stats > best_stats:
                best_query = query
                best_stats = curr_stats
            print("Current best:", best_stats, "\n", utils.remove_prefix(best_query))
            if curr_stats.emc == 1:
                print("Perfect match found.")
                break

def test_exact_match_csv():
    broken = [42, 213, 169, 143, 124, 105]
    emcs = broken #[187, 168, 73, 143, 150, 38, 101]
    matching = []

    # emcs = [1, 6, 8, 9, 21, 22, 23, 27, 31, 37, 38, 39, 40, 45, 60, 62, 64, 73, 79, 81, 87, 88, 92, 96, 99, 101, 103,
    #         104, 107, 111, 117, 119, 120, 122, 123, 126, 128, 129, 131, 132, 135, 141, 143, 144, 145, 150, 154, 155,
    #         156, 158, 160, 162, 164, 168, 176, 182, 183, 187, 188, 192, 198, 199, 213]
    # #matching = [199, 32, 31, 68, 22, 64, 99, 176, 62, 160, 37, 6, 132, 136, 117, 155, 111, 123, 183, 143, 128, 135, 213, 39, 1, 122, 145, 129, 21, 162, 198, 154, 96, 79, 104, 164, 156, 103, 131, 27, 45, 107, 88, 182, 60, 9, 8, 192, 92, 119, 87]
    # matching = [199, 32, 22, 23, 64, 160, 37, 188, 136, 187, 132, 62, 99, 123, 111, 117, 155, 128, 6, 143, 135, 39, 213,
    #             1, 122, 21, 145, 129, 198, 176, 79, 164, 156, 131, 27, 40, 103, 107, 88, 45, 182, 192, 104, 8, 119, 87,
    #             92, 101, 148, 162, 174, 183, 96, 120, 154, 73, 141, 31, 126, 199, 160, 132, 123, 22, 117, 111, 168, 128,
    #             188, 23, 187, 64, 37, 183, 73, 155, 135, 39, 21, 145, 1, 162, 122, 62, 198, 27, 164, 96, 154, 103, 156,
    #             45, 129, 79, 131, 182, 88, 6, 8, 87, 40, 119, 126, 107, 99, 192, 176, 101, 92, 104, 120, 81]
    emcs = [e for e in emcs if e not in matching]

    path_test = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        "QALD9_test-dataset-raw.csv"
    )
    path_test_out = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        "QALD9_test-dataset-raw-broken.csv"
    )

    rows = []

    with open(path_test) as csv_file:
        csv_dict = csv.DictReader(csv_file, delimiter=',')
        rows = list(csv_dict)

    emc_rows = [r for r in rows if int(r["id"]) in emcs]
    with open(path_test_out, "w", newline='') as stats_file:
        stats_writer = csv.DictWriter(stats_file, fieldnames=["id", "answertype", "question", "sparql"])
        stats_writer.writeheader()
        for stats in emc_rows:
            stats_writer.writerow(stats)
        stats_file.flush()

def partition(n, m, prefix):
    if n == 0:
        yield prefix

    for i in range(1, min(m, n)+1):
        yield from partition(n-i, i, prefix + [i])


def test_weighted_rr():
    print(list(utils.weighted_roundrobin([[1,2,3], [4,5,6], [7,8,9,10,11]], [3,2,1])))

def test_diagonals():
    a = list(utils.diagonal_generator([[1, 2, 3], [4], [5, 6, 7, 8, 9]]))
    print(a)
    #b = set(itertools.product([1, 2, 3], [4], [5, 6, 7, 8, 9]))
    #c = list(more_itertools.gray_product([1, 2, 3], [4], [5, 6, 7, 8, 9]))
    #print(c)
    #assert a == b
    # n = 2
    # lens = [1, 2, 3, 2]
    # for n in range(8):
    #     #res = [{q for q in itertools.permutations(p + ([0] * (len(lens)-len(p))))} for p in partition(n, n, []) if len(p) <= len(lens)]
    #     res = [{q for q in itertools.permutations(p + ([0] * (len(lens)-len(p)))) if all((a > b for a, b in zip(lens, q)))} for p in partition(n, n, []) if len(p) <= len(lens)]
    #     #res = [(p + ([0] * (n-len(p)))) for p in partition(n, n, [])]
    #     print(list(partition(n, n, [])))
    #     print(res)

def test_entity_preds():
    trie_tagger_path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels_trie_tagger_fr.cpl")
    trie_tagger = TrieTagger()
    trie_tagger.load_from_file(trie_tagger_path)
    se = SPARQLEndpoint()

    #entity_predicates: Dict[str, Set[str]] = defaultdict(set)
    entity_predicates = SqliteDict("entity_predicates.sqlite", autocommit=True, journal_mode="WAL")#, autocommit=False)
    entity_predicates_subj = SqliteDict("entity_predicates_subj.sqlite", autocommit=True, journal_mode="WAL")#, autocommit=False)
    entity_predicates_obj = SqliteDict("entity_predicates_obj.sqlite", autocommit=True, journal_mode="WAL")#, autocommit=False)

    inserted = 0

    query = "SELECT DISTINCT ?p WHERE {{ {{ <{}> ?p ?o }} UNION {{ ?s ?p <{}> }} }}"
    query_subj = "SELECT DISTINCT ?p WHERE {{ <{}> ?p ?o }}"
    query_obj = "SELECT DISTINCT ?p WHERE {{ ?s ?p <{}> }}"
    for label, uri in trie_tagger.items():
        #print(label, uri)
        while True:
            try:
                preds_subj = utils.sanitize_sparql_result(se.get_results_query(query_subj.format(uri)))
                preds_obj = utils.sanitize_sparql_result(se.get_results_query(query_obj.format(uri)))
                entity_predicates[uri] = preds_subj.union(preds_obj)
                entity_predicates_subj[uri] = preds_subj
                entity_predicates_obj[uri] = preds_obj
                break
            except Exception as e:
                print("Error:", e)
                time.sleep(1)

        inserted += 1
        if inserted % 10000 == 0:
            entity_predicates.commit()
            entity_predicates_subj.commit()
            entity_predicates_obj.commit()
            print("Inserted", inserted, flush=True)

    entity_predicates.commit()
    entity_predicates_subj.commit()
    entity_predicates_obj.commit()

    entity_predicates.close()
    entity_predicates_subj.close()
    entity_predicates_obj.close()

def test_entity_num():
    trie_tagger_path = os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "labels_trie_tagger_fr.cpl")
    trie_tagger = TrieTagger()
    trie_tagger.load_from_file(trie_tagger_path)
    inserted = 0

    for label, uri in trie_tagger.items():
        inserted += 1

    print("Inserted", inserted)

# def test_sparql():
#     # Create an object of class SPARQLSelectQuery and set the limit for the results to 100
#     select_query = SPARQLSelectQuery(distinct=True, limit=100)
#
#     # Add a prefix
#     select_query.add_prefix(
#         prefix=Prefix(prefix="ex", namespace="http://www.example.com#")
#     )
#
#     # Add the variables we want to select
#     select_query.add_variables(variables=["?person", "?age"])
#
#     # Create a graph pattern to use for the WHERE part and add some triples
#     where_pattern = SPARQLGraphPattern()
#     where_pattern.add_triples(
#         triples=[
#             Triple(subject="?person", predicate="rdf:type", object="dbr:Gary_Marcus"),
#             Triple(subject="?person", predicate="ex:hasAge", object="?age"),
#             Triple(subject="?person", predicate="ex:address", object="?address"),
#         ]
#     )
#
#     # Set this graph pattern to the WHERE part
#     select_query.set_where_pattern(graph_pattern=where_pattern)
#
#     # Group the results by age
#     select_query.add_group_by(
#         group=GroupBy(
#             variables=["?age"]
#         )
#     )
#
#     # Print the query we have defined
#     print(select_query.get_text())

# def test_qald9():
#     print("")
#
#     lp = LEMONParser.from_ttl_dir()
#     entries = lp.parse_nodes(lp.entry_nodes)
#     lex = Lexicon(entries)
#     oc = LEMONOntology(lexicon=lex)
#     # oc = BasicOntology()
#     path = os.path.join(
#         os.path.dirname(sys.modules["lemon"].__file__),
#         "resources",
#         "QALD9_train-dataset-raw.csv"
#     )
#     with open(path) as csv_file:
#         csv_dict = csv.DictReader(csv_file, delimiter=',')
#         for row in csv_dict:
#             #print(row)
#             dp = DUDESParser.from_str(row["question"], oc=oc)
#             dp.merge_dudes(debug=False)
#             res_dudes = list(dp.tree.nodes.values())[0].data.dudes
#             print("ID:", row["id"], "Question:", row["question"], "Formula:", str(res_dudes.str_formula), flush=True)
#
#     #dp = DUDESParser.from_str("Who lives in the retirement home is old.", oc=oc)



def test_lexiconsearch():
    nlp = spacy.load(spacy_model)
    nlp.add_pipe('dbpedia_spotlight')

    doc = nlp("The book Das Kapital was written by Adam Smith.")

    book = list(doc)[1]
    dtok = DUDESToken(book)
    print('Entities', [(ent.text, ent.label_, ent.kb_id_) for ent in doc.ents])

    lp = LEMONParser.from_ttl_dir()
    entries = lp.parse_nodes(lp.entry_nodes)
    lex = Lexicon(entries)
    entries = lex.find_entry(dtok)[0]
    for e in entries:
        pprint(e)


if __name__ == "__main__":
    test_qald()
    #test_qa_pipeline()
