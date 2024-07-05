# NeoDUDES - A Compositional Question Answering System Using DUDES
## Experimental Results Files

This folder contains tables with results, generated queries etc. as well as the datasets used for fine-tuning GPT-3.5-Turbo. The files are structured as follows:

- `NeoDUDES/` - Results of our approach
    - `3h-all-test/` - Benchmark results for test dataset with 3 hours timeout and parallel evaluation of all single-model strategies
        - `all-generated-queries-test.csv.zst` - Shows all queries generated during the 3 hours of benchmarking
        - `chosen-queries-per-strategy-test.csv` - Shows evaluation statistics per evaluation strategy and question
        - `strategy-total-eval-test.csv` - Shows total evaluation stats per strategy
    - `29376s-all-train/` - Benchmark results for train dataset with 29376 seconds timeout and parallel evaluation of all single-model strategies
        - Files analogous to `3h-all-test/`
    - `3h-bestonly/` - Benchmark results for test dataset with 3 hours timeout only using BestScore evaluation strategy, reducing overhead
        - Files analogous to `3h-all-test/`
    - `top-n-eval/` - Evaluation results for taking together the results of the top n models based on performance shown in `29376s-all-train/strategy-total-eval-train.csv` and using the queries of `3h-all-test/all-generated-queries-test.csv.zst` to create results comparable to the other results of `3h-all-test/`
        - `top2-models-eval-on-same-queries.txt` - Logs of evaluating the top 2 models with queries of `3h-all-test/all-generated-queries-test.csv.zst`
        - Analogous for remaining files for top3, top5 and top10
- `GPT/` - Results of GPT models
    - `finetuned-gpt-3.5-turbo/` - Results and dataset for finetuned-gpt-3.5-turbo
        - `datasets/` - Datasets used for fine-tuning, differing w.r.t. usage of lexicon and system prompt used
            - `train_prompt1.jsonl` - Training dataset for prompt 1 without lexical entries in prompt
            - `train_prompt1_lexicon.jsonl` - Training dataset for prompt 1 with lexical entries in prompt
            - `valid_prompt1.jsonl` - Validation dataset for prompt 1 without lexical entries in prompt
            - `valid_prompt1_lexicon.jsonl` - Validation dataset for prompt 1 with lexical entries in prompt
            - `test_prompt1.jsonl` - Test dataset for prompt 1 without lexical entries in prompt
            - `test_prompt1_lexicon.jsonl` - Test dataset for prompt 1 with lexical entries in prompt
            - Analogous for other prompts
        - `QALD9_finetuned-gpt-3.5-turbo_0-shot_prompt1_test.csv` - Generated responses/queries for test dataset for model trained/finetuned and validated with prompt 1 dataset without lexical entries in prompt
        - `QALD9_finetuned-gpt-3.5-turbo_0-shot_prompt1_test_lexicon.csv` - Generated responses/queries for test dataset for model trained/finetuned and validated with prompt 1 dataset with lexical entries in prompt
        - Analogous for other prompts
    - `gpt-3.5-turbo/` - Results for gpt-3.5-turbo
        - `QALD9_gpt-3.5-turbo_0-shot_test.csv` - Generated responses/queries for QALD-9 test dataset without adding lexical entries to prompt
        - `QALD9_gpt-3.5-turbo_0-shot_test_lexicon.csv` - Generated responses/queries for QALD-9 test dataset with adding lexical entries to prompt 
        - `QALD9_gpt-3.5-turbo_0-shot_train.csv` - Generated responses/queries for QALD-9 training dataset without adding lexical entries to prompt 
        - `QALD9_gpt-3.5-turbo_0-shot_train_lexicon.csv` - Generated responses/queries for QALD-9 training dataset with adding lexical entries to prompt  
    - `gpt-4/` - Results for gpt-4
        - `QALD9_gpt-4_0-shot_test.csv` - Generated responses/queries for QALD-9 test dataset without adding lexical entries to prompt
        - `QALD9_gpt-4_0-shot_test_lexicon.csv` - Generated responses/queries for QALD-9 test dataset with adding lexical entries to prompt 
        - `QALD9_gpt-4_0-shot_train.csv` - Generated responses/queries for QALD-9 training dataset without adding lexical entries to prompt 
        - `QALD9_gpt-4_0-shot_train_lexicon.csv` - Generated responses/queries for QALD-9 training dataset with adding lexical entries to prompt  

Please note the only postprocessing applied to the generated GPT queries was removing leading _\`\`\`sparql_ and trailing _\`\`\`_ if existing - SPARQL queries which could not be directly executed for other reasons were counted as invalid, resulting in statistics with no true or false positives but all respective false negatives.