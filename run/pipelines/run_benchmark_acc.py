# run_benchmark.py
import os
import shutil
import json
import time
import torch
import random
import logging
import gc
from tqdm import tqdm
import numpy as np

from .benchmarks.utils.eval_acc import run_gsm8k_eval, run_aime_eval, run_livecodebench_eval, run_mmlu_pro_eval, run_longbench_eval
from .benchmarks.gsm8k import load_gsm8k_dataset_answer
from .benchmarks.aime import load_aime_dataset_answer
from .benchmarks.livecodebench import load_livecodebench_dataset_answer
from .benchmarks.mmlu_pro import load_mmlu_pro_dataset_answer
from .benchmarks.narrativeqa import load_narrativeqa_dataset_answer
from .benchmarks.qasper import load_qasper_dataset_answer
from .benchmarks.multifieldqa_en import load_multifieldqa_en_dataset_answer
from .benchmarks.hotpotqa import load_hotpotqa_dataset_answer
from .benchmarks.musique import load_musique_dataset_answer
from .benchmarks._2wikimqa import load_2wikimqa_dataset_answer
from .benchmarks.gov_report import load_gov_report_dataset_answer
from .benchmarks.qmsum import load_qmsum_dataset_answer
from .benchmarks.multi_news import load_multi_news_dataset_answer
from .benchmarks.trec import load_trec_dataset_answer
from .benchmarks.triviaqa import load_triviaqa_dataset_answer
from .benchmarks.samsum import load_samsum_dataset_answer
from .benchmarks.passage_count import load_passage_count_dataset_answer
from .benchmarks.passage_retrieval_en import load_passage_retrieval_en_dataset_answer
from .benchmarks.lcc import load_lcc_dataset_answer
from .benchmarks.repobench_p import load_repobench_p_dataset_answer

DATASET_LOADER = {
    "gsm8k":      load_gsm8k_dataset_answer,
    "aime":       load_aime_dataset_answer,
    "livecodebench": load_livecodebench_dataset_answer,
    "mmlu_pro":   load_mmlu_pro_dataset_answer,
    "narrativeqa": load_narrativeqa_dataset_answer,
    "qasper": load_qasper_dataset_answer,
    "multifieldqa_en": load_multifieldqa_en_dataset_answer,
    "hotpotqa": load_hotpotqa_dataset_answer,
    "2wikimqa": load_2wikimqa_dataset_answer,
    "musique": load_musique_dataset_answer,  
    "gov_report": load_gov_report_dataset_answer,
    "qmsum": load_qmsum_dataset_answer,
    "multi_news": load_multi_news_dataset_answer,
    "trec": load_trec_dataset_answer, 
    "triviaqa": load_triviaqa_dataset_answer, 
    "samsum": load_samsum_dataset_answer,
    "passage_count": load_passage_count_dataset_answer,
    "passage_retrieval_en": load_passage_retrieval_en_dataset_answer,
    "lcc": load_lcc_dataset_answer,
    "repobench_p": load_repobench_p_dataset_answer,
}

BENCHMARK_EVALUATORS = {
    "gsm8k":      run_gsm8k_eval,
    "aime":       run_aime_eval,
    "livecodebench": run_livecodebench_eval,
    "mmlu_pro":   run_mmlu_pro_eval,
    "narrativeqa": run_longbench_eval,
    "qasper": run_longbench_eval,
    "multifieldqa_en": run_longbench_eval,
    "hotpotqa": run_longbench_eval,
    "2wikimqa": run_longbench_eval,
    "musique": run_longbench_eval,  
    "gov_report": run_longbench_eval,
    "qmsum": run_longbench_eval, 
    "multi_news": run_longbench_eval,
    "trec": run_longbench_eval,
    "triviaqa": run_longbench_eval,
    "samsum": run_longbench_eval,  
    "passage_count": run_longbench_eval,  
    "passage_retrieval_en": run_longbench_eval,  
    "lcc": run_longbench_eval,  
    "repobench_p": run_longbench_eval,
}

def main(builder, benchmarks=None, max_samples=None):
    torch.manual_seed(0)
    random.seed(0)
        
    # Enable profiling, disable logging profiling results
    builder.generator_profiling = True
    builder.profiling_verbose = False
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    args = builder.args
    
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)
    
    # Build bench_list and check if all names are valid
    bench_list = benchmarks.split(",") if benchmarks is not None else []
    for b in bench_list:
        if b not in DATASET_LOADER:
            raise ValueError(f"Unknown benchmark: {b}. Available benchmarks: {list(DATASET_LOADER.keys())}")
    print(f"Benchmarks to run: {bench_list}")
    
    # Handle output directories
    if args.out_dir is not None:
        shutil.rmtree(args.out_dir, ignore_errors=True)
        print(f"Deleted old {args.out_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
        
    # Run benchmarks
    log_dir_base = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    for bench_name in tqdm(bench_list, desc="Running benchmarks"):
        # fix random seed to 0 for each benchmark for reproducibility
        torch.manual_seed(0)
        random.seed(0)
        
        # Handle output directories
        log_dir = os.path.join(log_dir_base, bench_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory: {log_dir}")
        
        # Load dataset
        if "longgenbench" in bench_name:
            # longgenbench dataset is loaded differently
            # need "length tag"
            length_tag = bench_name.split("_")[-1]
            dataset = DATASET_LOADER[bench_name](length_tag)
            num_samples = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
            print(f"Running benchmark: {bench_name}, samples: {num_samples}, length_tag: {length_tag}")
    
            random.shuffle(dataset)
            dataset = dataset[:num_samples]
        elif not bench_name == "mmlu_pro":
            dataset = DATASET_LOADER[bench_name]()
            num_samples = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
            print(f"Running benchmark: {bench_name}, samples: {num_samples}")
    
            random.shuffle(dataset)
            dataset = dataset[:num_samples]
        else:
            # MMLU-Pro dataset is loaded differently
            dataset = load_mmlu_pro_dataset_answer(sample_per_category=max_samples, seed=0)
            num_samples = len(dataset)
            print(f"Running benchmark: {bench_name}, samples: {num_samples}")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
    
        # Evaluate
        if BENCHMARK_EVALUATORS[bench_name] == run_longbench_eval:
            metrics_json = BENCHMARK_EVALUATORS[bench_name](generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir, bench_name)
        else:
            metrics_json = BENCHMARK_EVALUATORS[bench_name](generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir)
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        if hasattr(generator, "judge_acc_len_list"):
            # print("acc_list:", generator.judge_acc_len_list)
            tacc_judge_value = np.mean(generator.judge_acc_len_list)
        else:
            tacc_judge_value = 0

        # Write results to file
        # with open(os.path.join(log_dir, "results.jsonl"), 'a+') as f:
        #     json.dump({
        #         bench_name: {
        #             "tput":         f"{tput_mean:.3f}",
        #             "tput_std":     f"{tput_std:.3f}",
        #             "Tacc":         f"{acc_rate_mean:.3f}",
        #             "Tacc_std":     f"{acc_rate_std:.3f}",
        #             "Accuracy":     f"{accuracy:.3f}",         
        #             "avg_draft_time":  f"{avg_draft_time:.3f}",
        #             "avg_target_time": f"{avg_target_time:.3f}",
        #             "peak_memory":     f"{peak_mem:.3f} GiB",
        #             "Tacc_judge" : f"{tacc_judge_value:.3f}",
        #         }
        #     }, f, indent=4)
        #     f.write("\n")

        # reduce float values to 3 decimal places
        for key in metrics_json:
            if isinstance(metrics_json[key], float):
                metrics_json[key] = f"{metrics_json[key]:.3f}"
        with open(os.path.join(log_dir, "results.jsonl"), 'a+') as f:
            json.dump({bench_name: metrics_json}, f, indent=4)
            f.write("\n")