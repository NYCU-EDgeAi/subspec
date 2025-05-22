import os
import shutil
import json
import time
import torch
import random
import logging
import gc
from tqdm import tqdm
import itertools

from specdecodes.models.utils.utils import DraftParams
from run.pipelines.benchmarks.utils.eval import run_common_eval, run_mtbench_eval
from run.pipelines.benchmarks.mtbench import load_mtbench_dataset

def evaluate_single_param(model, draft_model, tokenizer, builder, args, dataset, log_dir, temperature, max_depth, topk_len):
    builder.draft_params = DraftParams(
        temperature=temperature,
        max_depth=max_depth,
        topk_len=topk_len,
    )
    generator, tokenizer, past_kv, draft_past_kv = builder.build_generator_pipeline(model, draft_model, tokenizer)
    results = run_mtbench_eval(generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir)
    return results


def main(builder, temperature_values, max_depth_values, topk_len_values, max_samples=None):
    # Enable profiling, disable logging profiling results
    builder.generator_profiling = True
    builder.profiling_verbose = False
    model, draft_model, tokenizer = builder.build_models_and_tokenizer()
    args = builder.args
    
    # Set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)
    
    # Process candidate values
    temperature_values = [float(x) for x in temperature_values.split(",")]
    max_depth_values = [int(x) for x in max_depth_values.split(",")]
    topk_len_values = [int(x) for x in topk_len_values.split(",")]
    logging.info(f"Candidate values: temperature={temperature_values}, max_depth={max_depth_values}, topk_len={topk_len_values}")
    
    # Handle output directories
    if args.out_dir is not None:
        shutil.rmtree(args.out_dir, ignore_errors=True)
        logging.info(f"Deleted old {args.out_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Handle log directories
    log_dir_base = os.path.join(args.log_dir, "draft_params")
    log_dir_base = os.path.join(log_dir_base, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir_base, exist_ok=True)
        
    # Prepare the benchmark dataset (mt_bench)
    dataset = load_mtbench_dataset()
    num_samples = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    logging.info(f"Running mt-bench, samples: {num_samples}")
    
    # fix random seed to 0 for each benchmark for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    
    # Shuffle dataset and limit to num_samples
    random.shuffle(dataset)
    dataset = dataset[:num_samples]
    
    # Run benchmark
    for temperature, max_depth, topk_len in tqdm(itertools.product(temperature_values, max_depth_values, topk_len_values), 
                                                 total=len(temperature_values) * len(max_depth_values) * len(topk_len_values), 
                                                 desc="Configurations", 
                                                 leave=True):
        logging.info(f"\nTesting DraftParams: temperature={temperature}, max_depth={max_depth}, topk_len={topk_len}")
        
        # fix random seed to 0 for each iteration for reproducibility
        torch.manual_seed(0)
        random.seed(0)
        
        # Handle output directories
        log_dir = os.path.join(log_dir_base, f't{temperature}_d{max_depth}_k{topk_len}')
        os.makedirs(log_dir, exist_ok=True)
        logging.info(f"Log directory: {log_dir}")
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        # Evaluate
        try:
            tput_mean, tput_std, acc_rate_mean, acc_rate_std, avg_draft_time, avg_target_time, peak_mem = evaluate_single_param(model, draft_model, tokenizer, builder, args, dataset, log_dir, temperature, max_depth, topk_len)

        except Exception as e:
            logging.warning(f"Error during evaluation: {e}")
            logging.warning(f"Skipping this configuration.")
            continue
        
        torch.compiler.reset()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
    
        # Write results to file
        with open(os.path.join(log_dir, "results.jsonl"), 'w') as f:
            json.dump({
                "mt-bench": {
                    "tput": f"{tput_mean:.3f}",
                    "tput_std": f"{tput_std:.3f}", 
                    "Tacc": f"{acc_rate_mean:.3f}",
                    "Tacc_std": f"{acc_rate_std:.3f}",
                    "avg_draft_time": f"{avg_draft_time:.3f}",
                    "avg_target_time": f"{avg_target_time:.3f}",
                    "peak_memory": f"{peak_mem:.3f} GiB"
                }
            }, f, indent=4)
            f.write("\n")