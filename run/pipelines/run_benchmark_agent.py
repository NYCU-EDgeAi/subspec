import os
import shutil
import json
import time
import torch
import random
import logging
import gc
from tqdm import tqdm

from .benchmarks.utils.eval_agent import run_agent_eval
from .benchmarks.hotpotqa import load_hotpotqa_dataset

DATASET_LOADER = {
    "hotpotqa": load_hotpotqa_dataset,
}

BENCHMARK_EVALUATORS = {
    "hotpotqa": run_agent_eval,
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
    log_dir_base = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_benchmark_agent")
    for bench_name in tqdm(bench_list, desc="Running benchmarks"):
        # fix random seed to 0 for each benchmark for reproducibility
        torch.manual_seed(0)
        random.seed(0)
        
        # Handle output directories
        log_dir = os.path.join(log_dir_base, bench_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory: {log_dir}")

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
    log_dir_base = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_benchmark_agent")
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
