import csv
import itertools
import json
import os
import pickle
import random
import socket
import statistics
import sys
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.globals import set_verbose, set_debug

from llm.gold_lexicon_determiner import get_entries_for_question, entry_to_str

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dudes import utils, consts
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from dudes.utils import powerset, EvalStats, remove_prefix
from lemon.lemon_parser import LEMONParser

prompts = [
    "You are a system which creates SPARQL queries for DBPEDIA from 2016-10 from natural language user questions. You answer just with SPARQL queries and nothing else.",
    # Prompt for generating the prompts below (and regenerating a few times:
    # You are a world-class prompt engineer. Refine this prompt:
    # You are a system which creates SPARQL queries for DBPEDIA from 2016-10 from natural language user questions. You answer just with SPARQL queries and nothing else.
    "Generate SPARQL queries from user questions for DBpedia from October 2016. Answer solely with SPARQL queries.",
    "Develop a system capable of generating SPARQL queries for DBPedia based on user questions in natural language, with a knowledge base updated until October 2016. The system should exclusively respond with SPARQL queries and no additional information.",
    "Craft SPARQL queries from October 2016 based on user questions in natural language, exclusively dedicated to extracting information from DBpedia. Your responses should consist solely of SPARQL queries.",
    "Create SPARQL queries to generate responses to user questions by interpreting natural language queries, specifically targeting DBpedia, beginning from October 2016.",
]

stat_keys = ["True Positive", "False Positive", "False Negative", "Precision", "Recall", "F1", "Exact matches"]

def finetune_modify(
        train_path=None,
        valid_path=None,
        test_path=None,
        use_lexicon=False):

    validation_ids = [355, 287, 225, 105, 36, 251, 4, 183, 378, 155, 216, 310, 151, 194, 315, 360, 100, 3, 69, 224, 80, 25, 303, 394, 275, 346, 385, 37, 283, 158, 323, 112, 390, 369, 56, 265, 400, 267, 300, 77, 177]

    if train_path is None:
        train_path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "qald",
            "gpt",
            "finetune",
            f"train{'-lex' if use_lexicon else '-nolex'}.jsonl"
        )

    if valid_path is None:
        valid_path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "qald",
            "gpt",
            "finetune",
            f"valid{'-lex' if use_lexicon else '-nolex'}.jsonl"
        )

    if test_path is None:
        test_path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "qald",
            "gpt",
            "finetune",
            f"test{'-lex' if use_lexicon else '-nolex'}.jsonl"
        )

    with open(train_path, "r") as f:
        train_data = [json.loads(line) for line in f]

    valid_data = [row for i, row in enumerate(train_data) if i in validation_ids]
    train_data = [row for i, row in enumerate(train_data) if i not in validation_ids]

    with open(valid_path, "w") as f:
        for row in valid_data:
            f.write(f"{json.dumps(row)}\n")

    with open(train_path, "w") as f:
        for row in train_data:
            f.write(f"{json.dumps(row)}\n")

    with open(test_path, "r") as f:
        test_data = [json.loads(line) for line in f]

    with open(test_path, "w") as f:
        for row in test_data:
            f.write(f"{json.dumps(row)}\n")


def gen_finetune_file(
        prompt,
        train_path=None,
        test_path=None,
        #model="gpt-3.5-turbo-0125",
        use_lexicon=False):
    if train_path is None:
        train_path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "qald",
            "gpt",
            "finetune",
            f"train{'-lex' if use_lexicon else '-nolex'}.jsonl"
        )

    if test_path is None:
        test_path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "qald",
            "gpt",
            "finetune",
            f"test{'-lex' if use_lexicon else '-nolex'}.jsonl"
        )

    path_train = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        "QALD9_train-dataset-raw.csv"
    )

    path_test = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        "QALD9_test-dataset-raw.csv"
    )

    train_data = []
    test_data = []

    with open(path_train) as csv_file:
        train_data = list(csv.DictReader(csv_file, delimiter=','))

    with open(path_test) as csv_file:
        test_data = list(csv.DictReader(csv_file, delimiter=','))

    set_verbose(True)
    set_debug(True)

    lexicon = LEMONParser.from_ttl_dir().lexicon if use_lexicon else None

    print("## Prompt:", prompt)


    messages_train = []
    messages_test = []

    with open(train_path, "w") as f:
        for row in train_data:
            print("Train question:", row["id"],flush=True)
            if use_lexicon:
                msg = {"messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": row["question"]},
                    {"role": "user", "content": "\n".join([entry_to_str(ent) for ent in get_entries_for_question(row["question"], row["sparql"], lexicon)])},
                    {"role": "ai", "content": row["sparql"]}
                ]}
            else:
                msg = {"messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": row["question"]},
                    {"role": "ai", "content": row["sparql"]}
                ]}
            f.write(f"{json.dumps(msg)}\n")

    with open(test_path, "w") as f:
        for row in test_data:
            print("Test question:", row["id"], flush=True)
            if use_lexicon:
                msg = {"messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": row["question"]},
                    {"role": "user", "content": "\n".join([entry_to_str(ent) for ent in get_entries_for_question(row["question"], row["sparql"], lexicon)])},
                    {"role": "ai", "content": row["sparql"]}
                ]}
            else:
                msg = {"messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": row["question"]},
                    {"role": "ai", "content": row["sparql"]}
                ]}
            f.write(f"{json.dumps(msg)}\n")
            f.flush()

def query_gpt_dataset(experiment, model, dataset_path):
    with open(dataset_path, "r") as f:
        bench_data = [json.loads(line) for line in f]

    # {"messages": [
    #     {"role": "system",
    #      "content": "You are a system which creates SPARQL queries for DBPEDIA from 2016-10 from natural language user questions. You answer just with SPARQL queries and nothing else."},
    #     {"role": "user",
    #     "content": "What is the time zone of Salt Lake City?\nLexical Entry for time zone:\nPart of speech: PartOfSpeech.NOUN\nCanonical form: time zone\nOther forms:\ntime zone Number.SINGULAR\ntime zones Number.PLURAL\nSense:\nCondition:\nProperty domain: dbo:City\nProperty range: xsd:string\nReference: dbo:timeZone\nSubject of property: arg0 Marker: of\nObject of property: arg1\nSyntactic behavior: \nCopulative argument: arg1\nPrepositional adjunct: arg0  Marker: of\n\n\n\nLexical Entry for time zone:\nPart of speech: PartOfSpeech.NOUN\nCanonical form: time zone\nOther forms:\ntime zone Number.SINGULAR\nXX Number.PLURAL\nSense:\nCondition:\nProperty domain: dbo:Place\nProperty range: dbo:Country\nReference: dbo:timeZone\nSubject of property: arg0 Marker: in\nObject of property: arg1\nSyntactic behavior: \nCopulative argument: arg1\nPrepositional adjunct: arg0  Marker: in\n\n\n\nLexical Entry for time zone:\nPart of speech: PartOfSpeech.NOUN\nCanonical form: time zone\nOther forms:\ntime zone Number.SINGULAR\ntime zones Number.PLURAL\nSense:\nCondition:\nProperty domain: dbo:City\nProperty range: xsd:string\nReference: dbo:timeZone\nSubject of property: arg0 Marker: of\nObject of property: arg1\nCondition:\nProperty domain: dbo:City\nProperty range: xsd:string\nReference: dbo:timeZone\nSubject of property: arg0 Marker: of\nObject of property: arg1\nCondition:\nProperty domain: dbo:City\nProperty range: xsd:string\nReference: dbo:timeZone\nSubject of property: arg0 Marker: of\nObject of property: arg1\nSyntactic behavior: \nCopulative argument: arg1\nPrepositional adjunct: arg0  Marker: of\n\n\n\nLexical Entry for time zone:\nPart of speech: PartOfSpeech.NOUN\nCanonical form: time zone\nOther forms:\ntime zone Number.SINGULAR\nXX Number.PLURAL\nSense:\nCondition:\nProperty domain: dbo:Place\nProperty range: dbo:Country\nReference: dbo:timeZone\nSubject of property: arg0 Marker: in\nObject of property: arg1\nSyntactic behavior: \nCopulative argument: arg1\nPrepositional adjunct: arg0  Marker: in\n\n\n\nLexical Entry for time zone:\nPart of speech: PartOfSpeech.NOUN\nCanonical form: time zone\nOther forms:\ntime zone Number.SINGULAR\ntime zones Number.PLURAL\nSense:\nCondition:\nProperty domain: dbo:Place\nProperty range: dbo:Country\nReference: dbo:timeZone\nSubject of property: arg0 Marker: of\nObject of property: arg1\nSyntactic behavior: \nCopulative argument: arg1\nPrepositional adjunct: arg0  Marker: of\n\n\n\nLexical Entry for time zone:\nPart of speech: PartOfSpeech.NOUN\nCanonical form: time zone\nOther forms:\ntime zone Number.SINGULAR\ntime zones Number.PLURAL\nSense:\nCondition:\nProperty domain: dbo:Place\nProperty range: dbo:Country\nReference: dbo:timeZone\nSubject of property: arg0 Marker: in\nObject of property: arg1\nSyntactic behavior: \nCopulative argument: arg1\nPrepositional adjunct: arg0  Marker: in\n\n\n"},
    #     {"role": "assistant",
    #     "content": "PREFIX res: <http://dbpedia.org/resource/> PREFIX dbp: <http://dbpedia.org/property/> SELECT DISTINCT ?uri WHERE { res:Salt_Lake_City <http://dbpedia.org/ontology/timeZone> ?uri }"}]}

    # path_out = dataset_path + ".csv"
    path_out = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        "gpt",
        "QALD9_{}_{}_{}.csv".format(model, experiment, Path(dataset_path).stem)
    )

    # Load LLM
    llm = ChatOpenAI(
        model=model,
        openai_organization=os.environ.get("OPENAI_ORG"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.0,
        request_timeout=60,
        max_retries=3,
        verbose=True,
    )

    set_verbose(True)
    set_debug(True)

    mode = "w"
    if os.path.isfile(path_out):
        mode = "a"

    with open(path_out, mode, newline='') as out_file:
        fieldnames = ['id', 'question', 'sparql', 'prompt']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)

        if mode == "w":
            writer.writeheader()

        messages = [("system", "{prompt}"), ("user", "{question}")]

        prompt = ChatPromptTemplate.from_messages(messages)

        fewshot = prompt | llm

        for idx in range(len(bench_data)):
            nonshot = bench_data[idx]
            print(f"Initialization {idx}/{len(bench_data)}:")
            print(nonshot)
            print("##########")

            res = fewshot.invoke({
                "prompt": nonshot["messages"][0]["content"],
                "question": nonshot["messages"][1]["content"],
            })
            writer.writerow({
                "id": None,
                "question": nonshot["messages"][1]["content"],
                "sparql": res.content,
                "prompt": nonshot["messages"][0]["content"]
            })
            out_file.flush()
            print("Question:", nonshot["messages"][1]["content"])
            print("Gold:", nonshot["messages"][2]["content"])
            print("Pred:", res.content)
def query_gpt(experiment, nshot, model, repetitions, use_lexicon=False):
    # Load data
    path_train = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        f"QALD9_train-dataset-raw-{experiment}.csv" if experiment not in ["train", "test"] else f"QALD9_train-dataset-raw.csv"
    )

    path_test = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        f"QALD9_test-dataset-raw-{experiment}.csv" if experiment not in ["train", "test"] else f"QALD9_test-dataset-raw.csv"
    )

    bench_data = []
    test_data = []

    with open(path_train) as csv_file:
        train_data = list(csv.DictReader(csv_file, delimiter=','))

    with open(path_test) as csv_file:
        test_data = list(csv.DictReader(csv_file, delimiter=','))

    if experiment == "train":
        bench_data = train_data
    elif experiment == "test":
        bench_data = test_data
    else:#and ... used train
        bench_data = train_data

    path_out = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        "gpt",
        "QALD9_{}_{}-shot_{}{}.csv".format(model, nshot, experiment, ("_lexicon" if use_lexicon else ""))
    )


    # Load LLM
    llm = ChatOpenAI(
        model=model,
        openai_organization=os.environ.get("OPENAI_ORG"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.0,
        request_timeout=60,
        max_retries=3,
        verbose=True,
    )

    set_verbose(True)
    set_debug(True)



    # path_out_pkl = os.path.join(
    #     os.path.dirname(sys.modules["lemon"].__file__),
    #     "resources",
    #     "QALD9_{}_{}-shot_{}.pkl".format(model, nshot, experiment)
    # )

    mode = "w"
    if os.path.isfile(path_out):
        mode = "a"

    lexicon = LEMONParser.from_ttl_dir().lexicon if use_lexicon else None

    with open(path_out, mode, newline='') as out_file:
        fieldnames = ['id', 'question', 'sparql', 'prompt']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)

        if mode == "w":
            writer.writeheader()

        for pr in prompts:
            print("## Prompt:", pr)
            messages = []

            for idx in range(nshot):
                messages.append(("user", "{question" + str(idx) + "}"))
                # if lexicon:
                #     messages.append(("user", "{lexicon" + str(idx) + "}"))
                messages.append(("assistant", "{sparql" + str(idx) + "}"))

            messages.append(("user", "{question}"))
            # if lexicon:
            #     messages.append(("user", "{lexicon}"))
            messages = [("system", pr)] + messages

            prompt = ChatPromptTemplate.from_messages(messages)

            fewshot = prompt | llm
            # LLMChain(
            #     prompt=prompt,
            #     llm=llm,
            #     verbose=True,
            # )
            #

            macro_stats_list = []
            micro_stats_list = []

            for idx in range(len(bench_data)):
                remaining_data = [x for i, x in enumerate(bench_data) if i != idx]
                for rpt in range(repetitions):
                    shots = random.sample(remaining_data, nshot)
                    nonshot = bench_data[idx]
                    print(f"Initialization {idx}/{len(bench_data)} repetition {rpt}/{repetitions}:")
                    print("Shots:")
                    pprint([shot["question"] for shot in shots])
                    print("Question:", nonshot["question"])
                    print("##########")


                    for pshots in itertools.permutations(shots, r=nshot):
                        if lexicon:
                            res = fewshot.invoke({
                                "question": nonshot["question"] + "\n".join([entry_to_str(ent) for ent in get_entries_for_question(nonshot["question"], nonshot["sparql"], lexicon)]),
                                **{f"question{i}": shot["question"] + "\n".join([entry_to_str(ent) for ent in get_entries_for_question(shot["question"], shot["sparql"], lexicon)]) for i, shot in enumerate(pshots)},
                                **{f"sparql{i}": shot["sparql"] for i, shot in enumerate(pshots)}
                            })
                        else:
                            res = fewshot.invoke({
                                "question": nonshot["question"],
                                **{f"question{i}": shot["question"] for i, shot in enumerate(pshots)},
                                **{f"sparql{i}": shot["sparql"] for i, shot in enumerate(pshots)}
                            })
                        writer.writerow({"id": nonshot["id"], "question": nonshot["question"], "sparql": res.content, "prompt": pr})
                        out_file.flush()
                        print("Shots:")
                        pprint([shot["question"] for shot in pshots])
                        print("Question:", nonshot["question"])
                        print("Gold:", nonshot["sparql"])
                        print("Pred:", res.content)

def reeval_stats(experiment, nshot, model, use_lexicon, verbose=False, single_prompt_only=None, round_digits=2):
    try:
        path_in = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "qald",
            "gpt",
            "QALD9_{}_{}-shot_{}{}.csv".format(model, nshot, experiment, ("_lexicon" if use_lexicon else ""))
        )

        gpt_queries: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        with open(path_in) as csv_file:
            gpt_queries_rows = csv.DictReader(csv_file, delimiter=',')
            for row in gpt_queries_rows:
                if single_prompt_only is None or row["prompt"] == single_prompt_only:
                    gpt_queries[row["id"]][row["prompt"]].append(row["sparql"])


        path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "qald",
            "QALD9_train-dataset-raw.csv"
        )

        if experiment == "test":
            path = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                "qald",
                "QALD9_test-dataset-raw.csv"
            )

        gold_queries = dict()
        gold_questions = dict()
        with open(path) as csv_file:
            gold_queries_rows = csv.DictReader(csv_file, delimiter=',')
            for row in gold_queries_rows:
                gold_queries[row["id"]] = row["sparql"]
                gold_questions[row["id"]] = row["question"]

        sparql_endpoint = SPARQLEndpoint(endpoint=consts.sparql_endpoint)

        agg_stats: Dict[str, Dict[str, EvalStats]] = defaultdict(dict)

        stats_path = datetime.today().strftime('%Y-%m-%d') + f"-{model}-{nshot}-{experiment}-eval-stats-{socket.gethostname()}.csv"
        fieldnames = ['id', 'question', "prompt", "Gold SPARQL", "Generated SPARQL", "True Positive",
                      "False Positive", "False Negative", "Precision", "Recall", "F1", "Exact matches"]


        stats_writer: csv.DictWriter
        agg_rows = []

        with open(stats_path, "w", newline='') as stats_file:
            stats_writer = csv.DictWriter(stats_file, fieldnames=fieldnames)
            stats_writer.writeheader()
            id_stats = defaultdict(dict)
            for id, gold_query in gold_queries.items():
                #print("ID:", id)
                id_stats_single_for_micro = []
                #id_stats_mean_for_macro = []
                for prompt, prompt_gpt_queries in gpt_queries[id].items():
                    #print("Prompt:", prompt)
                    prompt_stats = []
                    for gpt_query in prompt_gpt_queries:
                        gpt_query = gpt_query.removeprefix("```sparql").removesuffix("```")
                        #print("Gold:", gold_query)
                        #print("Pred:", gpt_query)

                        estats, stats = utils.eval_queries(gold=gold_query, pred=gpt_query,
                                                           sparql_endpoint=sparql_endpoint, debug=False)
                        prompt_stats.append(estats)

                        #pprint(stats)
                    id_stats_single_for_micro.extend(prompt_stats)
                    nstats = EvalStats(
                        tp=statistics.mean([st.tp for st in prompt_stats]),
                        fp=statistics.mean([st.fp for st in prompt_stats]),
                        fn=statistics.mean([st.fn for st in prompt_stats]),
                        emc=statistics.mean([st.emc for st in prompt_stats])
                    )

                    stats_writer.writerow({
                        "id": id,
                        "question": gold_questions[id],
                        "prompt": prompt,
                        "Gold SPARQL": remove_prefix(gold_query),
                        "Generated SPARQL": remove_prefix(gpt_query),
                        "True Positive": nstats.tp,
                        "False Positive": nstats.fp,
                        "False Negative": nstats.fn,
                        "Precision": nstats.prec if nstats.prec is not None else 0.0,
                        "Recall": nstats.rec if nstats.rec is not None else 0.0,
                        "F1": nstats.f1 if nstats.f1 is not None else 0.0,
                        "Exact matches": nstats.emc,
                    })
                nstats = EvalStats(
                    tp=statistics.mean([st.tp for st in id_stats_single_for_micro]),
                    fp=statistics.mean([st.fp for st in id_stats_single_for_micro]),
                    fn=statistics.mean([st.fn for st in id_stats_single_for_micro]),
                    emc=statistics.mean([st.emc for st in id_stats_single_for_micro])
                )
                #id_stats_mean_for_macro.append(nstats)

                stats_writer.writerow({
                    "id": id,
                    "question": gold_questions[id],
                    "prompt": "Average of all prompts",
                    "Gold SPARQL": remove_prefix(gold_query),
                    "Generated SPARQL": "Average of all prompts",
                    "True Positive": nstats.tp,
                    "False Positive": nstats.fp,
                    "False Negative": nstats.fn,
                    "Precision": nstats.prec if nstats.prec is not None else 0.0,
                    "Recall": nstats.rec if nstats.rec is not None else 0.0,
                    "F1": nstats.f1 if nstats.f1 is not None else 0.0,
                    "Exact matches": nstats.emc,
                })
                stats_file.flush()
                id_stats[id]["id_stats_single_for_micro"] = id_stats_single_for_micro
                id_stats[id]["id_stats_mean_for_macro"] = nstats

            print("Micro:")
            micro = sum(sum([st["id_stats_single_for_micro"] for st in id_stats.values()], []), EvalStats())
            pprint(micro)
            print("Macro:")
            macro = {
                "Precision": statistics.mean([st["id_stats_mean_for_macro"].prec if st["id_stats_mean_for_macro"].prec is not None else 0.0 for st in id_stats.values()]),
                "Recall": statistics.mean([st["id_stats_mean_for_macro"].rec if st["id_stats_mean_for_macro"].rec is not None else 0.0 for st in id_stats.values()]),
                "F1": statistics.mean([st["id_stats_mean_for_macro"].f1 if st["id_stats_mean_for_macro"].f1 is not None else 0.0 for st in id_stats.values()]),
            }
            pprint(macro)
            print("", flush=True)

            row_model = model
            if "GPT-3.5-Turbo".lower() in model.lower():
                row_model = "GPT-3.5-Turbo"
            elif "GPT-3.5".lower() in model.lower():
                row_model = "GPT-3.5"
            elif "GPT-4".lower() in model.lower():
                row_model = "GPT-4"

            agg_rows.append({
                "Model": row_model,
                "FT": "\\xmark",
                "Prompt": prompts.index(single_prompt_only) + 1 if single_prompt_only else "All",
                "Lexicon": "\\cmark" if use_lexicon else "\\xmark",
                "Micro $F_1$": "$" + (f"%.{round_digits}f" % round(micro.f1, round_digits)) + "$",
                "Micro P": "$" + (f"%.{round_digits}f" % round(micro.precision, round_digits)) + "$",
                "Micro R": "$" + (f"%.{round_digits}f" % round(micro.recall, round_digits)) + "$",
                "Macro $F_1$": "$" + (f"%.{round_digits}f" % round(macro["F1"], round_digits)) + "$",
                "Macro P": "$" + (f"%.{round_digits}f" % round(macro["Precision"], round_digits)) + "$",
                "Macro R": "$" + (f"%.{round_digits}f" % round(macro["Recall"], round_digits)) + "$",
            })
    except Exception as e:
        print(e)
        return []

    return agg_rows

def reeval_stats_dataset(dataset_path, round_digits=2):
    try:
        path_in = dataset_path

        if "-nolex-" in dataset_path:
            use_lexicon = False
        else:
            use_lexicon = True

        gpt_queries: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        with open(path_in) as csv_file:
            gpt_queries_rows = csv.DictReader(csv_file, delimiter=',')
            for row in gpt_queries_rows:
                gpt_queries[row["question"].split("\n")[0]][row["prompt"]].append(row["sparql"])

        path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "qald",
            "QALD9_test-dataset-raw.csv"
        )

        gold_queries = dict()
        gold_questions = dict()
        with open(path) as csv_file:
            gold_queries_rows = csv.DictReader(csv_file, delimiter=',')
            for row in gold_queries_rows:
                gold_queries[row["question"]] = row["sparql"]
                gold_questions[row["question"]] = row["id"]

        sparql_endpoint = SPARQLEndpoint(endpoint=consts.sparql_endpoint)

        agg_stats: Dict[str, Dict[str, EvalStats]] = defaultdict(dict)

        stats_path = datetime.today().strftime('%Y-%m-%d') + f"-{Path(dataset_path).stem}-eval-stats-{socket.gethostname()}.csv"
        fieldnames = ['id', 'question', "prompt", "Gold SPARQL", "Generated SPARQL", "True Positive",
                      "False Positive", "False Negative", "Precision", "Recall", "F1", "Exact matches"]


        stats_writer: csv.DictWriter

        agg_rows = []

        all_present_prompts = set()

        with open(stats_path, "w", newline='') as stats_file:
            stats_writer = csv.DictWriter(stats_file, fieldnames=fieldnames)
            stats_writer.writeheader()
            id_stats = defaultdict(dict)
            for quest, gold_query in gold_queries.items():
                #print("ID:", id)
                id_stats_single_for_micro = []
                #id_stats_mean_for_macro = []
                for prompt, prompt_gpt_queries in gpt_queries[quest].items():
                    all_present_prompts.add(prompt)
                    #print("Prompt:", prompt)
                    prompt_stats = []
                    for gpt_query in prompt_gpt_queries:
                        gpt_query = gpt_query.removeprefix("```sparql").removesuffix("```")
                        #print("Gold:", gold_query)
                        #print("Pred:", gpt_query)

                        estats, stats = utils.eval_queries(gold=gold_query, pred=gpt_query,
                                                           sparql_endpoint=sparql_endpoint, debug=False)
                        prompt_stats.append(estats)

                        #pprint(stats)
                    id_stats_single_for_micro.extend(prompt_stats)
                    nstats = EvalStats(
                        tp=statistics.mean([st.tp for st in prompt_stats]),
                        fp=statistics.mean([st.fp for st in prompt_stats]),
                        fn=statistics.mean([st.fn for st in prompt_stats]),
                        emc=statistics.mean([st.emc for st in prompt_stats])
                    )

                    stats_writer.writerow({
                        "id": gold_questions[quest],
                        "question": quest,
                        "prompt": prompt,
                        "Gold SPARQL": remove_prefix(gold_query),
                        "Generated SPARQL": remove_prefix(gpt_query),
                        "True Positive": nstats.tp,
                        "False Positive": nstats.fp,
                        "False Negative": nstats.fn,
                        "Precision": nstats.prec if nstats.prec is not None else 0.0,
                        "Recall": nstats.rec if nstats.rec is not None else 0.0,
                        "F1": nstats.f1 if nstats.f1 is not None else 0.0,
                        "Exact matches": nstats.emc,
                    })
                nstats = EvalStats(
                    tp=statistics.mean([st.tp for st in id_stats_single_for_micro]),
                    fp=statistics.mean([st.fp for st in id_stats_single_for_micro]),
                    fn=statistics.mean([st.fn for st in id_stats_single_for_micro]),
                    emc=statistics.mean([st.emc for st in id_stats_single_for_micro])
                )
                #id_stats_mean_for_macro.append(nstats)

                stats_writer.writerow({
                    "id": gold_questions[quest],
                    "question": quest,
                    "prompt": "Average of all prompts",
                    "Gold SPARQL": remove_prefix(gold_query),
                    "Generated SPARQL": "Average of all prompts",
                    "True Positive": nstats.tp,
                    "False Positive": nstats.fp,
                    "False Negative": nstats.fn,
                    "Precision": nstats.prec if nstats.prec is not None else 0.0,
                    "Recall": nstats.rec if nstats.rec is not None else 0.0,
                    "F1": nstats.f1 if nstats.f1 is not None else 0.0,
                    "Exact matches": nstats.emc,
                })
                stats_file.flush()
                id_stats[quest]["id_stats_single_for_micro"] = id_stats_single_for_micro
                id_stats[quest]["id_stats_mean_for_macro"] = nstats

            print("Micro:")
            micro = sum(sum([st["id_stats_single_for_micro"] for st in id_stats.values()], []), EvalStats())
            pprint(micro)
            print("Macro:")
            macro = {
                "Precision": statistics.mean([st["id_stats_mean_for_macro"].prec if st["id_stats_mean_for_macro"].prec is not None else 0.0 for st in id_stats.values()]),
                "Recall": statistics.mean([st["id_stats_mean_for_macro"].rec if st["id_stats_mean_for_macro"].rec is not None else 0.0 for st in id_stats.values()]),
                "F1": statistics.mean([st["id_stats_mean_for_macro"].f1 if st["id_stats_mean_for_macro"].f1 is not None else 0.0 for st in id_stats.values()]),
            }
            pprint(macro)
            assert len(all_present_prompts) == 1
            agg_rows.append({
                "Model": "GPT-3.5-Turbo-0125",
                "FT": "\\cmark",
                "Prompt": prompts.index(list(all_present_prompts)[0])+1,
                "Lexicon": "\\cmark" if use_lexicon else "\\xmark",
                "Micro $F_1$": "$" + (f"%.{round_digits}f" % round(micro.f1, round_digits)) + "$",
                "Micro P": "$" + (f"%.{round_digits}f" % round(micro.precision, round_digits)) + "$",
                "Micro R": "$" + (f"%.{round_digits}f" % round(micro.recall, round_digits)) + "$",
                "Macro $F_1$": "$" + (f"%.{round_digits}f" % round(macro["F1"], round_digits)) + "$",
                "Macro P": "$" + (f"%.{round_digits}f" % round(macro["Precision"], round_digits)) + "$",
                "Macro R": "$" + (f"%.{round_digits}f" % round(macro["Recall"], round_digits)) + "$",
            })

            print("", flush=True)
    except Exception as e:
        print(e)
        return []

    return agg_rows

if __name__ == "__main__":
    # finetune_modify()
    # # gen_finetune_file(prompts[0])
    # exit(0)
    argparser = ArgumentParser()
    argparser.add_argument('--eval', action='store_true')
    argparser.add_argument('--verbose', action='store_true')
    argparser.add_argument('--model', type=str, default="gpt-3.5-turbo")
    argparser.add_argument('--exp', type=str, default="and")
    argparser.add_argument('--dataset', type=str, default=None)
    argparser.add_argument('--nshot', type=int, default=3)
    argparser.add_argument('--repetitions', type=int, default=5)
    argparser.add_argument('--lexicon', action='store_true')
    arguments = argparser.parse_args()
    experiment = arguments.exp  #"and"
    nshot = arguments.nshot # 3
    model = arguments.model # "gpt-3.5-turbo"
    repetitions = arguments.repetitions # 5
    print(arguments)
    if arguments.eval:
        round_digits = 3
        agg_rows = []
        fns = ["Model", "FT", "Prompt", "Lexicon", "Micro $F_1$", "Micro P", "Micro R", "Macro $F_1$", "Macro P", "Macro R"]
        all_exps = ["test"]#["train", "test"]  # ["and", "order"]
        all_nshots = [0]
        all_models = ["gpt-3.5-turbo", "gpt-4"]
        all_lex = [True, False]
        all_prompts = prompts  # [None] + prompts
        # all_reps = [5, 1]

        for model in all_models:
            for exp in all_exps:
                for nshot in all_nshots:
                    for lex in all_lex:
                        for pr in all_prompts:
                            # for reps in all_reps:
                            print(
                                f"\n\n###### Evaluating experiment '{exp}' with {nshot}-shot, model {model}, {len(prompts)} different prompts, lexicon {lex}, prompt: {pr}:")
                            # aggregate_stats(exp, nshot, model, verbose=arguments.verbose)
                            agg_rows += reeval_stats(exp, nshot, model, use_lexicon=lex, verbose=arguments.verbose, single_prompt_only=pr)
        #if arguments.dataset is not None:
        dataset_files = [
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:lex:9Yi7BEjS_lex-prompt0_test-lex-prompt0.csv",
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:nolex:9YhsnnN4_nolex-prompt0_test-nolex-prompt0.csv",
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:lex-prompt1:9YsVxisj_lex-prompt1_test-lex-prompt1.csv",
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:nolex-prompt1:9YtS0R26_nolex-prompt1_test-nolex-prompt1.csv",
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:lex-prompt2:9Ytm2zMK_lex-prompt2_test-lex-prompt2.csv",
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:nolex-prompt2:9YtZZliI_nolex-prompt2_test-nolex-prompt2.csv",
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:lex-prompt3:9YuaDxYG_lex-prompt3_test-lex-prompt3.csv",
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:nolex-prompt3:9YuvbE3m_nolex-prompt3_test-nolex-prompt3.csv",
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:lex-prompt4:9YwTjjXw_lex-prompt4_test-lex-prompt4.csv",
            "QALD9_ft:gpt-3.5-turbo-0125:ag-sc-citec:nolex-prompt4:9Yv6zvBe_nolex-prompt4_test-nolex-prompt4.csv",
        ]
        all_datasets = [
            os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                "qald",
                "gpt",
                csvfn
            )
            for csvfn in dataset_files
        ]

        for ds in all_datasets:
            print(f"\n\n###### Evaluating dataset '{ds}':")
            agg_rows += reeval_stats_dataset(ds)

        agg_df = pd.DataFrame(agg_rows, columns=fns)
        print(agg_df.to_latex(index=False, escape=False))
        agg_df.to_csv(f"gpt-full-eval-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)
        #else:

    else:
        if arguments.dataset is not None:
            query_gpt_dataset(experiment, model, arguments.dataset)
        else:
            query_gpt(experiment, nshot, model, repetitions, use_lexicon=arguments.lexicon)

