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

def evaluate_single_param(
    model,
    draft_model,
    tokenizer,
    builder,
    args,
    dataset,
    log_dir,
    temperature,
    max_depth,
    topk_len,
    threshold=None,
    window_size=None,
):
    builder.draft_params = DraftParams(
        temperature=temperature,
        max_depth=max_depth,
        topk_len=topk_len,
    )

    builder.generator_kwargs = builder.generator_kwargs or {}
    builder.generator_kwargs["verify_method"] = (
        "lossy"
        if (threshold is not None or window_size is not None)
        else builder.generator_kwargs.get("verify_method", "exact")
    )
    builder.generator_kwargs.setdefault("verify_kwargs", {})
    if threshold is not None:
        builder.generator_kwargs["verify_kwargs"]["threshold"] = float(threshold)
    if window_size is not None:
        builder.generator_kwargs["verify_kwargs"]["window_size"] = int(window_size)
    generator, tokenizer, past_kv, draft_past_kv = builder.build_generator_pipeline(model, draft_model, tokenizer)
    return run_mtbench_eval(generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir)


def main(
    builder,
    temperature_values,
    max_depth_values,
    topk_len_values,
    threshold_values=None,
    window_size_values=None,
    max_samples=None,
):
    # Enable profiling, disable logging profiling results
    builder.generator_profiling = True
    builder.profiling_verbose = False

    model, draft_model, tokenizer = builder.build_models_and_tokenizer()
    args = builder.args

    # Keep a handle to the original forward methods so each config can recompile cleanly.
    # Without this, repeated torch.compile() calls can end up wrapping an already-compiled forward.
    base_target_forward = model.forward
    base_draft_forward = draft_model.forward if draft_model is not None else None
    
    # Set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)
    
    # Process candidate values
    temperature_values = [float(x) for x in temperature_values.split(",")]
    max_depth_values = [int(x) for x in max_depth_values.split(",")]
    topk_len_values = [int(x) for x in topk_len_values.split(",")]

    if threshold_values is None:
        threshold_values = [0.0]
    else:
        threshold_values = [float(x) for x in str(threshold_values).split(",") if str(x).strip()]
        if not threshold_values:
            threshold_values = [0.0]

    if window_size_values is None:
        window_size_values = [1]
    else:
        window_size_values = [int(x) for x in str(window_size_values).split(",") if str(x).strip()]
        if not window_size_values:
            window_size_values = [1]

    logging.info(
        "Candidate values: temperature=%s, max_depth=%s, topk_len=%s, threshold=%s, window_size=%s",
        temperature_values,
        max_depth_values,
        topk_len_values,
        threshold_values,
        window_size_values,
    )
    
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
    combos = list(itertools.product(
        temperature_values,
        max_depth_values,
        topk_len_values,
        threshold_values,
        window_size_values,
    ))
    for temperature, max_depth, topk_len, threshold, window_size in tqdm(
        combos,
        total=len(combos),
        desc="Configurations",
        leave=True,
    ):
        logging.info(
            "\nTesting DraftParams: temperature=%s, max_depth=%s, topk_len=%s, threshold=%s, window_size=%s",
            temperature,
            max_depth,
            topk_len,
            threshold,
            window_size,
        )
        
        # fix random seed to 0 for each iteration for reproducibility
        torch.manual_seed(0)
        random.seed(0)
        
        # Handle output directories
        log_dir = os.path.join(
            log_dir_base,
            f"t{temperature}_d{max_depth}_k{topk_len}_th{threshold}_w{window_size}",
        )
        os.makedirs(log_dir, exist_ok=True)
        logging.info(f"Log directory: {log_dir}")
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        # Each configuration can change shapes (e.g., max_verify_tokens / attention masks).
        # Reset compile caches so torch.compile doesn't reuse shape-specialized graphs.
        try:
            torch.compiler.reset()
        except Exception:
            pass

        # Be more aggressive about clearing Dynamo/Inductor caches.
        # This keeps torch.compile enabled for speed, while ensuring each (d,k) gets
        # a fresh compile and we don't hit shape-guard reuse issues.
        try:
            import torch._dynamo as dynamo  # type: ignore

            dynamo.reset()
            if hasattr(dynamo, "reset_code_caches"):
                dynamo.reset_code_caches()
        except Exception:
            pass

        try:
            import torch._inductor as inductor  # type: ignore

            if hasattr(inductor, "utils") and hasattr(inductor.utils, "clear_inductor_caches"):
                inductor.utils.clear_inductor_caches()
        except Exception:
            pass

        # Restore original forwards so torch.compile wraps a clean graph each iteration.
        model.forward = base_target_forward
        if draft_model is not None and base_draft_forward is not None:
            draft_model.forward = base_draft_forward

        # Evaluate
        try:
            results = evaluate_single_param(
                model,
                draft_model,
                tokenizer,
                builder,
                args,
                dataset,
                log_dir,
                temperature,
                max_depth,
                topk_len,
                threshold,
                window_size,
            )

            tput_mean = float(results.get("tput_mean", 0.0))
            tput_std = float(results.get("tput_std", 0.0))
            acc_rate_mean = float(results.get("tacc_mean", 0.0))
            acc_rate_std = float(results.get("tacc_std", 0.0))
            avg_draft_time = float(results.get("avg_draft_time", 0.0))
            avg_target_time = float(results.get("avg_target_time", 0.0))
            peak_mem = float(results.get("peak_memory_gib", 0.0))

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
                    "peak_memory": f"{peak_mem:.3f} GiB",
                    "threshold": f"{float(threshold):.6g}",
                    "window_size": int(window_size),
                }
            }, f, indent=4)
            f.write("\n")