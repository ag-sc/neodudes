from argparse import ArgumentParser

import numpy as np
import pandas as pd
import re

#Strategy	Micro F1	Micro TP	Micro FP	Micro FN	Micro EM	Micro Precision	Micro Recall	Macro F1	Macro Precision	Macro Recall	Really finished	Total results	Total questions
def agg_stats(path, round_digits = 2):
    data = pd.read_csv(path)
    data["Strategy"] = data["Strategy"].dropna().apply(lambda x: re.sub(r"LLMAccumEval_([0-9]+)_(.*)", r"$\\text{LLMAccumEval}_{\2}$", x) if isinstance(x, str) else x)
    data["Strategy"] = data["Strategy"].dropna().apply(lambda x: re.sub(r"LLMMostWinsEval_([0-9]+)_([0-9]+\.[0-9]+)", r"$\\text{LLMMostWinsEval}_{\2}$", x) if isinstance(x, str) else x)
    data = data[data["Strategy"].notna()]
    #data = data[data.filter(regex=".*_None_.*", axis=1)]
    data = data[data["Strategy"].str.contains("_None_") == False]

    agg_data = []
    

    for gr, ids in data.groupby(["Strategy"]).groups.items():
        print(gr)

        agg_data.append({
            "Strategy": gr.replace("LLM", "").replace("Eval", ""),
            "Micro $F_1$": "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Micro F1"].mean(), round_digits)) + "\\pm " + (f"%.{round_digits}f" % round(data.loc[ids]["Micro F1"].std(), round_digits)) + "$" if not np.isnan(data.loc[ids]["Micro F1"].std()) else "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Micro F1"].mean(), round_digits)) + "$",
            "Micro P": "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Micro Precision"].mean(), round_digits)) + "\\pm " + (f"%.{round_digits}f" % round(data.loc[ids]["Micro Precision"].std(), round_digits)) + "$" if not np.isnan(data.loc[ids]["Micro Precision"].std()) else "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Micro Precision"].mean(), round_digits)) + "$",
            "Micro R": "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Micro Recall"].mean(), round_digits)) + "\\pm " + (f"%.{round_digits}f" % round(data.loc[ids]["Micro Recall"].std(), round_digits)) + "$" if not np.isnan(data.loc[ids]["Micro Recall"].std()) else "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Micro Recall"].mean(), round_digits)) + "$",
            "Macro $F_1$": "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Macro F1"].mean(), round_digits)) + "\\pm " + (f"%.{round_digits}f" % round(data.loc[ids]["Macro F1"].std(), round_digits)) + "$" if not np.isnan(data.loc[ids]["Macro F1"].std()) else "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Macro F1"].mean(), round_digits)) + "$",
            "Macro P": "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Macro Precision"].mean(), round_digits)) + "\\pm " + (f"%.{round_digits}f" % round(data.loc[ids]["Macro Precision"].std(), round_digits)) + "$" if not np.isnan(data.loc[ids]["Macro Precision"].std()) else "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Macro Precision"].mean(), round_digits)) + "$",
            "Macro R": "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Macro Recall"].mean(), round_digits)) + "\\pm " + (f"%.{round_digits}f" % round(data.loc[ids]["Macro Recall"].std(), round_digits)) + "$" if not np.isnan(data.loc[ids]["Macro Recall"].std()) else "$" + (f"%.{round_digits}f" % round(data.loc[ids]["Macro Recall"].mean(), round_digits)) + "$",
        })

        #print(data.loc[ids].describe())
        print(data.loc[ids]["Micro F1"].mean())
        print(data.loc[ids]["Micro F1"].std())
    #data.loc[['$\\text{LLMAccumEval}_logits$']]["Micro F1"].std()
    #data["Strategy"] = data["Strategy"].dropna().drop(data["Strategy"].str.contains("_None_"))

    agg_df = pd.DataFrame(agg_data, columns=["Strategy", "Micro $F_1$", "Micro P", "Micro R", "Macro $F_1$", "Macro P", "Macro R"])
    print(agg_df.to_latex(index=False, escape=False))

    pass


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--path", type=str)
    args = argparser.parse_args()
    agg_stats(args.path)
    # argparser.add_argument("--batchsize", type=int, default=80)
    # argparser.add_argument("--epochs", type=int, default=None)
    # argparser.add_argument("--lr", type=float, default=None)
    # argparser.add_argument("--ld", type=float, default=None)
    # argparser.add_argument("--trials", type=int, default=10)
    # argparser.add_argument("--optunafile", type=str, default=f"gen_optuna_{datetime.now().strftime('%Y-%m-%d')}.log")
    # argparser.add_argument("--studyname", type=str, default=f"Query Score {datetime.now().strftime('%Y-%m-%d')}")
    # argparser.add_argument("--dataset", type=str, default="f1")