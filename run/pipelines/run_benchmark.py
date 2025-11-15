import os
import shutil
import json
import time
import torch
import random
import logging
import gc
from tqdm import tqdm

from .benchmarks.utils.eval import run_common_eval, run_mtbench_eval
from .benchmarks.mtbench import load_mtbench_dataset
from .benchmarks.humaneval import load_humaneval_dataset
from .benchmarks.gsm8k import load_gsm8k_dataset
from .benchmarks.alpaca import load_alpaca_dataset
from .benchmarks.cnndm import load_cnndm_dataset
from .benchmarks.aime import load_aime_dataset
from .benchmarks.gpqa import load_gpqa_dataset
from .benchmarks.math500 import load_math500_dataset
from .benchmarks.livecodebench import load_livecodebench_dataset
from .benchmarks.hotpotqa import load_hotpotqa_dataset
from .benchmarks.narrativeqa import load_narrativeqa_dataset
from .benchmarks.qasper import load_qasper_dataset
from .benchmarks.multifieldqa_en import load_multifieldqa_en_dataset
from .benchmarks.musique import load_musique_dataset
from .benchmarks._2wikimqa import load_2wikimqa_dataset
from .benchmarks.gov_report import load_gov_report_dataset
from .benchmarks.qmsum import load_qmsum_dataset
from .benchmarks.multi_news import load_multi_news_dataset
from .benchmarks.trec import load_trec_dataset
from .benchmarks.triviaqa import load_triviaqa_dataset
from .benchmarks.samsum import load_samsum_dataset
from .benchmarks.passage_count import load_passage_count_dataset
from .benchmarks.passage_retrieval_en import load_passage_retrieval_en_dataset
from .benchmarks.lcc import load_lcc_dataset
from .benchmarks.repobench_p import load_repobench_p_dataset

DATASET_LOADER = {
    "mt-bench": load_mtbench_dataset,
    "human-eval": load_humaneval_dataset,
    "gsm8k": load_gsm8k_dataset,
    "alpaca": load_alpaca_dataset,
    "cnn-dm": load_cnndm_dataset,
    "aime": load_aime_dataset,
    "gpqa": load_gpqa_dataset,
    "math-500": load_math500_dataset,
    "livecodebench": load_livecodebench_dataset,
    "hotpotqa": load_hotpotqa_dataset,
    "narrativeqa": load_narrativeqa_dataset,
    "qasper": load_qasper_dataset,
    "multifieldqa_en": load_multifieldqa_en_dataset,
    "hotpotqa": load_hotpotqa_dataset,
    "2wikimqa": load_2wikimqa_dataset,
    "musique": load_musique_dataset,  
    "gov_report": load_gov_report_dataset,
    "qmsum": load_qmsum_dataset,
    "multi_news": load_multi_news_dataset,
    "trec": load_trec_dataset, 
    "triviaqa": load_triviaqa_dataset, 
    "samsum": load_samsum_dataset,
    "passage_count": load_passage_count_dataset,
    "passage_retrieval_en": load_passage_retrieval_en_dataset,
    "lcc": load_lcc_dataset,
    "repobench_p": load_repobench_p_dataset,
}

BENCHMARK_EVALUATORS = {
    "mt-bench": run_mtbench_eval,
    "human-eval": run_common_eval,
    "gsm8k": run_common_eval,
    "alpaca": run_common_eval,
    "cnn-dm": run_common_eval,
    "aime": run_common_eval,
    "gpqa": run_common_eval,
    "math-500": run_common_eval,
    "livecodebench": run_common_eval,
    "hotpotqa": run_common_eval,
    "narrativeqa": run_common_eval,
    "qasper": run_common_eval,
    "multifieldqa_en": run_common_eval,
    "2wikimqa": run_common_eval,
    "musique": run_common_eval,  
    "gov_report": run_common_eval,
    "qmsum": run_common_eval, 
    "multi_news": run_common_eval,
    "trec": run_common_eval,
    "triviaqa": run_common_eval,
    "samsum": run_common_eval,  
    "passage_count": run_common_eval,  
    "passage_retrieval_en": run_common_eval,  
    "lcc": run_common_eval,  
    "repobench_p": run_common_eval,
}

# Benchmarks
# common: "mt-bench", "human-eval", "gsm8k", "alpaca", "cnn-dm"
# reasoning: "aime", "gpqa", "math-500", "livecodebench"

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
        dataset = DATASET_LOADER[bench_name]()
        num_samples = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
        print(f"Running benchmark: {bench_name}, samples: {num_samples}")

        # Shuffle dataset and limit to num_samples
        random.shuffle(dataset)
        dataset = dataset[:num_samples]
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        # Evaluate
        metrics_json = BENCHMARK_EVALUATORS[bench_name](generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir)
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
    
        # Write results to file
        # with open(os.path.join(log_dir, "results.jsonl"), 'a+') as f:
        #     json.dump({
        #         bench_name: {
        #             "tput": f"{tput_mean:.3f}",
        #             "tput_std": f"{tput_std:.3f}", 
        #             "Tacc": f"{acc_rate_mean:.3f}",
        #             "Tacc_std": f"{acc_rate_std:.3f}",
        #             "avg_draft_time": f"{avg_draft_time:.3f}",
        #             "avg_target_time": f"{avg_target_time:.3f}",
        #             "peak_memory": f"{peak_mem:.3f} GiB"
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
