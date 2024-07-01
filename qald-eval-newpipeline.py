import cProfile
import csv
import multiprocessing
import os
import queue
import socket
import sys
import time
import traceback
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from queue import Queue
from typing import Optional, Dict, Tuple, List, Generator, Any, Union

import logging

import rpyc

from dudes.qa.dudes_rpc_service import LLMQuerySelectorWrapper

import dudes.qa.dudes_rpc_service
from dudes import consts, utils
from dudes.qa.qa_pipeline import QAPipeline
# from dudes.ontologies.lemon_ontology import LEMONOntology
import compress_pickle as cpl  # type: ignore
import stanza

from dudes.qa.sparql_selection.query_evaluation_strategies import BestScoreEval, LLMMostWinsEval
from dudes.qa.sparql_selection.query_evaluator import QueryEvaluator
from dudes.utils import compare_results, remove_prefix, EvalStats, prettier_print

multiprocessing.set_start_method("fork", force=True)  #"spawn"

cachepath = "qald-cache-parallel.cpkl"
LOG = logging.getLogger(__name__)


def get_query_dudes(row, qa: QAPipeline):
    result_generator = qa.process_qald(question=row["question"])

    print("ID:", row["id"], "Question:", row["question"],  #"Formula:", str([rd.str_formula for rd in res_dudes]),
          "\nGold SPARQL:", remove_prefix(row["sparql"]),
          #"\nSPARQL:", "\n" + str([remove_prefix(query) for query in res_queries]),
          flush=True)

    return result_generator  #res_queries, res_dudes, res_queries_full


def gen_error_result(message, err_fn=0, err_combinations=None):
    return message, None, err_combinations


def stats_row_creator(row, strategy, nstats, query, dudes, query_full, combinations, runtime, no_prefix=True):
    return {
        "id": row["id"],
        "question": row["question"],
        "Strategy": strategy,
        "Gold SPARQL": remove_prefix(row["sparql"]) if no_prefix else row["sparql"],
        "Generated SPARQL": remove_prefix(query) if no_prefix else query,
        "Generated SPARQL Full": remove_prefix(query_full) if no_prefix else query_full,
        "True Positive": nstats.tp if nstats.tp is not None else 0,
        "False Positive": nstats.fp if nstats.fp is not None else 0,
        "False Negative": nstats.fn if nstats.fn is not None else 0,
        "Precision": nstats.prec if nstats.prec is not None else 0.0,
        "Recall": nstats.rec if nstats.rec is not None else 0.0,
        "F1": nstats.f1 if nstats.f1 is not None else 0.0,
        "Exact matches": nstats.emc if nstats.emc is not None else 0,
        "Runtime": runtime,#already_elapsed + (time.time() - start)
        "Combinations": combinations, #f"{best_changed}/{valid_combs}/{all_combs}",
        "DUDES": str(dudes)
    }

not_ready_message = "No valid result yet."
def not_ready_stats_row_creator(row, gold):
    return stats_row_creator(row, "", EvalStats(fn=len(gold)), not_ready_message, None, not_ready_message, f"0/0/0", 0.0)

def best_results_generator(
        row: Dict,
        gold_res: Any,
        qa: QAPipeline,
        skip_empty_queries: bool,
        all_query_queue: Any,
        rpc_conn: rpyc.Connection
) -> Generator[tuple[Union[str, QueryEvaluator], Optional[int], Optional[int]], None, None]:
    #start = time.time()
    comb_counter = 0
    valid_comb_counter = 0

    if gold_res is None:
        print(f"Error: Gold query returned bad results! Skipping {row['id']}")
        yield gen_error_result(message=f"Error: Gold query returned no results! Skipping {row['id']}")
        return

    gold = utils.sanitize_sparql_result(gold_res)

    if len(gold) == 0:
        print(f"Error: Gold query returned no results! Skipping {row['id']}")
        yield gen_error_result(message=f"Error: Gold query returned no results! Skipping {row['id']}")
        return

    sg = qa.sparql_generator

    try:
        # print(row)
        result_generator = get_query_dudes(row, qa)

        query_idx = 0

        qeval = QueryEvaluator.default(gold, row["question"], rpc_conn)

        yield qeval, valid_comb_counter, comb_counter

        for query, dudes, full_query in result_generator:
            comb_counter += 1

            if dudes is None:
                print("Error: No DUDES generated for ID:", row["id"], "Question:", row["question"], query)
                continue

            if skip_empty_queries and "WHERE {\n}" in query:
                continue

            sys_res = None
            query_idx += 1
            if query_idx % 100 == 0:
                print(f"Fetching query {query_idx} of question '{row['question']}'", flush=True)
            try:
                sys_res = sg.get_results_query(query)
            except Exception as e:
                print("Error: Bad query:", e, query)
                continue

            curr_stats = compare_results(gold_res=gold_res, sys_res=sys_res)
            valid_comb_counter += 1

            while True:
                try:
                    all_query_queue.put(
                        item=stats_row_creator(
                            row=row, strategy="", nstats=curr_stats, query=query, dudes=dudes, query_full=full_query,
                            combinations=f"{comb_counter}/{valid_comb_counter}/{query_idx}", runtime=0.0,
                            no_prefix=False
                        ),
                        timeout=10
                    )
                    break
                except queue.Full:
                    print("all_query result queue timed out!", flush=True)
                    continue
                except Exception as e:
                    print("Unexpected error putting all_query result! ", e, flush=True)
                    break


            #if best_score_eval(curr_stats, best_dudes, best_stats):
            qeval.eval(curr_stats, query, dudes, full_query)

            if curr_stats.emc == 1:
                print("Exact match at ID", row["id"], query)
                #yield best_stats, best_query, best_dudes, best_full_query, best_changed, valid_comb_counter, comb_counter
                #return
            elif curr_stats.tp > 0:
                print("Partial match at ID", row["id"], query)

            #if best_dudes is not None:
            yield qeval, valid_comb_counter, comb_counter

        if comb_counter == 0:
            print("Warning: No queries_gen generated for", row["id"])
            yield gen_error_result(
                message="Error: No queries generated for" + row["id"],
                err_combinations=comb_counter,
                err_fn=len(gold)
            )
            return

    except RuntimeError as e:
        print("Error for ID:", row["id"], "Question:", row["question"], e)
        print(traceback.format_exc(), flush=True)
        yield gen_error_result(message="Error: " + str(e), err_fn=len(gold))
        return
    except KeyboardInterrupt as e:
        print("Error for ID:", row["id"], "Question:", row["question"], e)
        print(traceback.format_exc(), flush=True)
        yield gen_error_result(message="Error: " + str(e), err_fn=len(gold))
        return
    except Exception as e:
        # raise e
        print("Error Unexpected for ID:", row["id"], "Question:", row["question"], e)
        print(traceback.format_exc(), flush=True)
        yield gen_error_result(message="Error: " + str(e), err_fn=len(gold))
        return


def dudes_process(params) -> Dict:
    total_start = time.time()

    row_queue, result_queue, all_query_queue, total_timeout, round_timeout, skip_empty_queries, tid, use_profiler = params
    print(f"Error: Starting process {tid}!", flush=True)

    profiler = cProfile.Profile()
    # h = hpy()
    if use_profiler:
        print("Error: Profiling enabled!", flush=True)
        profiler.enable()

    cache: Dict = defaultdict(dict)

    if os.path.isfile(cachepath):
        with open(cachepath, "rb") as f:
            cache = cpl.load(f, compression="lzma")
    # endpoint = "http://dbpedia.org/sparql"
    # endpoint = "http://localhost:8890/sparql"
    # endpoint = http://client.linkeddatafragments.org/#datasources=http%3A%2F%2Ffragments.dbpedia.org%2F2016-04%2Fen

    # dbpedia_spotlight_endpoint = 'http://localhost:2222/rest'

    rpc_conn = rpyc.connect(consts.rpc_host,
                            consts.rpc_port,
                            config={
                                "allow_public_attrs": True,
                                "allow_pickle": True,
                                "sync_request_timeout": 300
                            })

    qa = QAPipeline.default(
        dbpedia_spotlight_endpoint=consts.dbpedia_spotlight_endpoint,
        cache=cache,
        # rpc_conn=rpc_conn,
        # trie_tagger_host=None,
        # trie_tagger_port=None,
        #dudes_composer_candidate_limit=3,
    )
    sg = qa.sparql_generator

    result_stats: Dict = dict()
    micro_stats_list: Dict = dict()
    next_round_queue: Queue[Tuple[Any, Any, float, Any]] = Queue()

    def process_single_question(row, result_generator, already_elapsed, gold_res):
        start = time.time()

        if already_elapsed < 0.0001:
            while True:
                try:
                    gold = utils.sanitize_sparql_result(gold_res)
                    not_ready_results = not_ready_stats_row_creator(row, gold)

                    result_queue.put(
                        item=(
                        row, EvalStats(fn=len(gold)), not_ready_message, str(None), not_ready_message, not_ready_results),
                        timeout=10
                    )
                    break
                except queue.Full:
                    print("Initial result queue timed out!", flush=True)
                    continue
                except Exception as e:
                    print("Unexpected error putting initial result! ", e, flush=True)
                    break

        for qeval, valid_combs, all_combs in result_generator:

            if isinstance(qeval, str):
                print("Error: In this step no DUDES generated for ID:", row["id"], "Question:", row["question"], qeval, flush=True)
                elapsed = time.time() - start
                if elapsed > round_timeout:
                    print(f"Process {tid} Round timeout!", row["id"], elapsed, flush=True)
                    next_round_queue.put((row, result_generator, already_elapsed + elapsed, gold_res))
                    break
                continue

            assert isinstance(qeval, QueryEvaluator)
            for strat in qeval.best_query.keys():
                nstats = qeval.best_stats[strat]
                query = qeval.best_query[strat]
                dudes = qeval.best_dudes[strat]
                query_full = qeval.best_full_query[strat]
                best_changed = qeval.best_changed[strat]
                micro_stats_list[row["id"]] = nstats
                print(f"Thread {tid} eval:", strat, row["id"], nstats,  # "Macro:", utils.macro_stats(micro_stats_list),
                      "Processed rows:", len(result_stats), flush=True)

                result_stats[row["id"]] = stats_row_creator(row, strat, nstats, query, dudes, query_full,
                                                            combinations=f"{best_changed}/{valid_combs}/{all_combs}",
                                                            runtime=already_elapsed + (time.time() - start))

                print(result_stats[row["id"]], flush=True)

                while True:
                    try:
                        result_queue.put(
                            item=(row, nstats, query, str(dudes), query_full, result_stats[row["id"]]),
                            timeout=10
                        )
                        break
                    except queue.Full:
                        print("Result queue timed out!", flush=True)
                        continue
                    except Exception as e:
                        print("Unexpected error putting result! ", e, flush=True)
                        break
            elapsed = time.time() - start
            if elapsed > round_timeout:
                print(f"Process {tid} Round timeout!", row["id"], elapsed, flush=True)
                next_round_queue.put((row, result_generator, already_elapsed + elapsed, gold_res))
                break

        print(f"Process {tid} Finished ID:", row["id"], "Question:", row["question"], flush=True)

    while not row_queue.empty():
        try:
            if time.time() - total_start > total_timeout:
                print(f"Process {tid} Total timeout!", flush=True)
                break

            row = row_queue.get(timeout=10)  # block at most 10 seconds
            print(f"Process {tid} Processing ID:", row["id"], "Question:", row["question"], flush=True)

            gold_res = None
            try:
                gold_res = sg.get_results_query(row["sparql"])
            except Exception as e:
                # raise e
                print("Error for gold of ID:", row["id"], "Question:", row["question"], e)

            result_generator = best_results_generator(row, gold_res, qa, skip_empty_queries, all_query_queue, rpc_conn)
            process_single_question(row, result_generator, 0.0, gold_res)
            if use_profiler:
                profiler.dump_stats(f"qald-eval-{tid}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.prof")
            # stats_writer.writerow(stats)
        except queue.Empty:
            print("Row queue timed out!", flush=True)
            continue
        except RuntimeError as e:
            if "row" in locals() and row is not None:
                print("Error for ID:", row["id"], "Question:", row["question"], e)
                #traceback.print_stack()
            else:
                print("Error: Before fetching row?!", e)
            continue

    print(f"Finished query fetching process {tid}!", flush=True)

    while not next_round_queue.empty():
        print(f"Process {tid} Queue Size {next_round_queue.qsize()}", flush=True)
        try:
            if time.time() - total_start > total_timeout:
                print("Total timeout!", flush=True)
                break
            row, result_generator, runtime, gold_res = next_round_queue.get(timeout=10)
            print(f"Process {tid} Further processing ID:", row["id"], "Question:", row["question"], flush=True)
            process_single_question(row, result_generator, runtime, gold_res)
            if use_profiler:
                profiler.dump_stats(f"qald-eval-{tid}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.prof")
        except queue.Empty:
            print("next_round_queue timed out!", flush=True)
            continue
        except RuntimeError as e:
            if "row" in locals() and row is not None:
                print("Error for ID:", row["id"], "Question:", row["question"], e)
                #traceback.print_stack()
            else:
                print("Error: Before fetching row?!", e)
            continue
    if use_profiler:
        profiler.disable()
        profiler.dump_stats(f"qald-eval-{tid}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.prof")
    print(f"Process {tid} Terminated!", flush=True)
    return cache


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--singlethread', action='store_false')
    argparser.add_argument('--test', action='store_true')
    argparser.add_argument("--roundtimeout", type=int, default=60)
    argparser.add_argument("--totaltimeout", type=int, default=3600)
    argparser.add_argument("--exp", type=str, default=None, required=False)
    argparser.add_argument('--profile', action='store_true')

    arguments = argparser.parse_args()

    profiler = cProfile.Profile()
    #h = hpy()
    if arguments.profile:
        profiler.enable()
        #h.heap()
        print("Profiling enabled!", flush=True)


    multithreaded = arguments.singlethread  # default True
    use_test = arguments.test  # default False
    experiment = arguments.exp

    if experiment is not None:
        print("Intermediate: Experiment:", experiment)

    logging.basicConfig(level=logging.INFO)

    thread = dudes.qa.dudes_rpc_service.start_rpc_service()
    print("Trie tagger started!", flush=True)

    # oc = BasicOntology()
    path = os.path.join(
        os.path.dirname(sys.modules["lemon"].__file__),
        "resources",
        "qald",
        "QALD9_train-dataset-raw.csv" if experiment is None else f"QALD9_train-dataset-raw-{experiment}.csv"
    )

    if use_test:
        path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "qald",
            "QALD9_test-dataset-raw.csv" if experiment is None else f"QALD9_test-dataset-raw-{experiment}.csv"
        )
        print("Intermediate: Dataset: test", flush=True)
    else:
        print("Intermediate: Dataset: train", flush=True)

    stanza.download('en')

    # with open(cachepath, "wb") as f:
    #     cache.pop('http://localhost:8890/sparql')
    #     cache = cpl.dump(sg.cache, f, compression="lzma")

    # exit(0)

    rows = []

    with open(path) as csv_file:
        csv_dict = csv.DictReader(csv_file, delimiter=',')
        rows = list(csv_dict)

    cpu_count = 4 # multiprocessing.cpu_count()

    full_cache: Dict = defaultdict(dict)
    nstats_results: Dict[str, Dict[str, EvalStats]] = defaultdict(dict)
    interm_result_stats: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    stats_path = datetime.today().strftime('%Y-%m-%d') + f"-eval-stats-{socket.gethostname()}{'-' + experiment if experiment is not None else ''}" + ("-test" if use_test else "-train") + ".csv"
    strat_stats_path = datetime.today().strftime('%Y-%m-%d') + f"-strategy-eval-stats-{socket.gethostname()}{'-' + experiment if experiment is not None else ''}" + ("-test" if use_test else "-train") + ".csv"
    fieldnames = ['id', 'question', "Strategy", "Gold SPARQL", "Generated SPARQL", "Generated SPARQL Full", "True Positive",
                  "False Positive", "False Negative", "Precision", "Recall", "F1", "Exact matches", "Runtime",
                  "Combinations", "DUDES"]

    strat_fns = ['Strategy',
                 'Micro F1', 'Micro TP', 'Micro FP', 'Micro FN', 'Micro EM', 'Micro Precision', 'Micro Recall',
                 'Macro F1', 'Macro Precision', 'Macro Recall',
                 'Really finished', 'Total results', 'Total questions']

    all_queries: List[Dict] = []

    all_queries_path = datetime.today().strftime('%Y-%m-%d') + f"-all-queries-{socket.gethostname()}{'-' + experiment if experiment is not None else ''}" + ("-test" if use_test else "-train") + ".csv"


    def refresh_stats():
        stats_writer: csv.DictWriter

        with open(stats_path, "w", newline='') as stats_file:
            stats_writer = csv.DictWriter(stats_file, fieldnames=fieldnames)
            stats_writer.writeheader()
            for strat in interm_result_stats.keys():
                for stats in interm_result_stats[strat].values():
                    stats_writer.writerow(stats)
            stats_file.flush()

        with open(strat_stats_path, "w", newline='') as strat_stats_file:
            strat_stats_writer = csv.DictWriter(strat_stats_file, fieldnames=strat_fns)
            strat_stats_writer.writeheader()
            for strat in interm_result_stats.keys():
                really_finished = [val for val in interm_result_stats[strat].values() if val["Combinations"] != "0/0/0"]
                micro = sum(nstats_results[strat].values(), EvalStats()).to_dict()
                macro = utils.macro_stats(nstats_results[strat].values())
                strat_stats_writer.writerow({
                    "Strategy": strat,
                    "Micro F1": micro["F1"] if micro["F1"] is not None else 0.0,
                    "Micro TP": micro["True Positives"],
                    "Micro FP": micro["False Positives"],
                    "Micro FN": micro["False Negatives"],
                    "Micro EM": micro["Exact matches"],
                    "Micro Precision": micro["Precision"] if micro["Precision"] is not None else 0.0,
                    "Micro Recall": micro["Recall"] if micro["Recall"] is not None else 0.0,
                    "Macro F1": macro["F1"] if macro["F1"] is not None else 0.0,
                    "Macro Precision": macro["Precision"] if macro["Precision"] is not None else 0.0,
                    "Macro Recall": macro["Recall"] if macro["Recall"] is not None else 0.0,
                    "Really finished": len(really_finished),
                    "Total results": len(nstats_results[strat]),
                    "Total questions": len(rows)
                })
            strat_stats_file.flush()

        # with open(all_queries_path, "w", newline='') as stats_file:
        #     stats_writer = csv.DictWriter(stats_file, fieldnames=fieldnames)
        #     stats_writer.writeheader()
        #     for stats in all_queries:
        #         stats_writer.writerow(stats)
        #     stats_file.flush()


    def print_eval(final=False):
        if len(interm_result_stats) > 0:
            for strat in interm_result_stats.keys():
                really_finished = [val for val in interm_result_stats[strat].values() if val["Combinations"] != "0/0/0"]
                print(f"{'Intermediate' if not final else 'Final'} eval:", strat, prettier_print(sum(nstats_results[strat].values(), EvalStats()).to_dict()),
                      "Macro:", prettier_print(utils.macro_stats(nstats_results[strat].values())),
                      "Processed rows:", len(really_finished), "/", len(nstats_results[strat]), "/", len(rows),
                      flush=True)
            refresh_stats()


    with open(all_queries_path, "w", newline='') as allq_file:
        allq_writer = csv.DictWriter(allq_file, fieldnames=fieldnames)
        allq_writer.writeheader()

        if multithreaded:
            with multiprocessing.Manager() as manager:
                q = manager.Queue()
                res_q = manager.Queue()
                all_q = manager.Queue()
                for row in rows:
                    q.put(row)
                with Pool(cpu_count) as p:
                    logging.basicConfig(level=logging.INFO)
                    ares = p.map_async(dudes_process,
                                       [(q, res_q, all_q, arguments.totaltimeout, arguments.roundtimeout, False, tid, arguments.profile) for tid in range(cpu_count)],
                                       chunksize=1)

                    while True:
                        try:
                            res = ares.get(timeout=5)
                            break
                        except multiprocessing.TimeoutError:
                            print(f"Processing not finished, queue size: {q.qsize()} empty: {q.empty()}", flush=True)

                            while not res_q.empty():
                                try:
                                    row, nstats, query, str_dudes, query_full, stats_row = res_q.get(timeout=5)

                                    #if row["id"] not in interm_result_stats or nstats > nstats_results[row["id"]]:
                                    interm_result_stats[stats_row["Strategy"]][row["id"]] = stats_row
                                    nstats_results[stats_row["Strategy"]][row["id"]] = nstats

                                except queue.Empty:
                                    print("Intermediate result queue timed out!", flush=True)
                                    continue
                            while not all_q.empty():
                                try:
                                    stat = all_q.get(timeout=5)

                                    allq_writer.writerow(stat)
                                except queue.Empty:
                                    print("All queries queue timed out!", flush=True)
                                    continue
                            allq_file.flush()
                            if arguments.profile:
                                profiler.dump_stats(f"qald-eval{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.prof")
                            print_eval()

                            continue
                print("Emptying result queue!", flush=True)
                while not res_q.empty():
                    try:
                        row, nstats, query, str_dudes, query_full, stats_row = res_q.get(timeout=5)
                        #if row["id"] not in interm_result_stats or nstats > nstats_results[row["id"]]:
                        interm_result_stats[stats_row["Strategy"]][row["id"]] = stats_row
                        nstats_results[stats_row["Strategy"]][row["id"]] = nstats

                    except queue.Empty:
                        print("Intermediate result queue timed out!", flush=True)
                        continue
                while not all_q.empty():
                    try:
                        stat = all_q.get(timeout=5)

                        allq_writer.writerow(stat)
                        #all_queries.append(stat)
                    except queue.Empty:
                        print("All queries queue timed out!", flush=True)
                        continue
                if arguments.profile:
                    profiler.dump_stats(f"qald-eval{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.prof")
                print_eval()
        else:
            logging.basicConfig(level=logging.INFO)
            q = Queue()
            res_q = Queue()
            all_q = Queue()
            for row in rows:
                q.put(row)

            res = [dudes_process((q, res_q, all_q, arguments.totaltimeout, arguments.roundtimeout, False, tid, arguments.profile)) for tid in range(cpu_count)] #[dudes_process((q, res_q, all_q, arguments.totaltimeout, arguments.roundtimeout, False))]
            while not res_q.empty() or not q.empty():
                try:
                    row, nstats, query, str_dudes, query_full, stats_row = res_q.get(timeout=5)
                    #if row["id"] not in interm_result_stats or nstats > nstats_results[row["id"]]:
                    interm_result_stats[stats_row["Strategy"]][row["id"]] = stats_row
                    nstats_results[stats_row["Strategy"]][row["id"]] = nstats
                    print_eval()
                    if arguments.profile:
                       profiler.dump_stats(f"qald-eval{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.prof")

                except queue.Empty:
                    print("Intermediate result queue timed out!", flush=True)
                    continue
            while not all_q.empty():
                try:
                    stat = all_q.get(timeout=5)

                    allq_writer.writerow(stat)
                    #all_queries.append(stat)
                except queue.Empty:
                    print("All queries queue timed out!", flush=True)
                    continue
            print_eval()
            if arguments.profile:
               profiler.dump_stats(f"qald-eval{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.prof")

    for cache in res:
        full_cache.update(cache)

    if arguments.profile:
        #h.heap()
        profiler.disable()
        profiler.dump_stats(f"qald-eval{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.prof")

    print_eval(final=True)

    refresh_stats()

    with open(cachepath, "wb") as f:
        # cache.pop('http://localhost:8890/sparql')
        cpl.dump(full_cache, f, compression="lzma")
