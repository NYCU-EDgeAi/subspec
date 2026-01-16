import os
import json
import re
import base64
import zlib
import pickle
import subprocess
import tempfile
import time
from typing import Any, List, Dict
import numpy as np
import torch
import gc
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel

from .utils import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)
from specdecodes.models.utils.wandb_logger import wandb_logger
from run.pipelines.utils.eval_utils import reset_kv, maybe_init_cuda_graph_runner

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench_p": code_sim_score,
}

def _run_warmup(
    generator,
    tokenizer,
    past_key_values,
    draft_past_key_values,
    args,
    warmup_prompt,
    max_length=None,
    max_new_tokens=None,
    warmup_iter=None,
    show_progress=False,
):
    original_profiling = generator.profiling
    generator.profiling = False
    n_iter = args.warmup_iter if warmup_iter is None else warmup_iter
    if n_iter <= 0:
        generator.profiling = original_profiling
        return

    iterator = tqdm(range(n_iter), desc="Warming up") if show_progress else range(n_iter)
    for _ in iterator:
        tokenizer.use_default_system_prompt = True
        warmup_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": warmup_prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        gen_kwargs = dict(
            temperature=args.temperature,
            do_sample=args.do_sample,
            past_key_values=past_key_values,
            draft_past_key_values=draft_past_key_values,
        )
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        else:
            gen_kwargs["max_length"] = max_length if max_length is not None else args.max_length

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            generator.generate(warmup_ids, **gen_kwargs)

        reset_kv(past_key_values, draft_past_key_values)

    generator.profiling = original_profiling


def _init_perf():
    return {
        "tput_list": [],
        "tacc_list": [],
        "total_iter": 0,
        "total_draft_time": 0.0,
        "total_target_time": 0.0,
    }


def _accum_perf(perf, record):
    perf["tput_list"].append(record["tput"])
    perf["tacc_list"].append(record["avg_sampled"])
    n_iter = record["n_iter"]
    perf["total_iter"] += n_iter
    perf["total_draft_time"] += record["avg_draft_time"] * n_iter
    perf["total_target_time"] += record["avg_target_time"] * n_iter


def _finalize_perf(perf, generator):
    tput_list = perf["tput_list"]
    tacc_list = perf["tacc_list"]
    total_iter = perf["total_iter"]
    total_draft_time = perf["total_draft_time"]
    total_target_time = perf["total_target_time"]

    tput_mean, tput_std = (np.mean(tput_list), np.std(tput_list)) if tput_list else (0, 0)
    tacc_mean, tacc_std = (np.mean(tacc_list), np.std(tacc_list)) if tacc_list else (0, 0)
    avg_draft_time = (total_draft_time / total_iter) if total_iter > 0 else 0
    avg_target_time = (total_target_time / total_iter) if total_iter > 0 else 0
    peak_memory = torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3)

    return {
        "tput_mean": float(tput_mean),
        "tput_std": float(tput_std),
        "tacc_mean": float(tacc_mean),
        "tacc_std": float(tacc_std),
        "avg_draft_time": float(avg_draft_time),
        "avg_target_time": float(avg_target_time),
        "peak_memory_gib": float(peak_memory),
    }


def _print_summary(title, perf_stats, accuracy=None, correct_q=None, total_q=None):
    print(f"Final {title} Results:")
    print(f"\tThroughput       : {perf_stats['tput_mean']:.3f} ± {perf_stats['tput_std']:.3f} tokens/sec")
    print(f"\tToken Acceptance : {perf_stats['tacc_mean']:.3f} ± {perf_stats['tacc_std']:.3f}")
    if accuracy is not None:
        if correct_q is not None and total_q is not None:
            print(f"\tAnswer Accuracy  : {accuracy:.3f} ({correct_q}/{total_q})")
        else:
            print(f"\tAnswer Accuracy  : {accuracy:.3f}")
    print(f"\tAvg Draft Time   : {perf_stats['avg_draft_time']:.3f} sec")
    print(f"\tAvg Target Time  : {perf_stats['avg_target_time']:.3f} sec")
    print(f"\tPeak Memory      : {perf_stats['peak_memory_gib']:.3f} GiB")

def run_gsm8k_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir):
    """
    Evaluate GSM8K dataset accuracy alongside performance metrics.

    Args:
        generator: the model generator instance
        tokenizer: tokenizer with chat template functionality
        past_key_values: primary past key values for autoregressive generation
        draft_past_key_values: draft past key values for speculative decoding (optional)
        args: namespace containing temperature, max_length, do_sample, warmup_iter
        dataset: list of dicts, each with keys:
            "question": the prompt string
            "answer": full original answer text from GSM8K (with reasoning and final line "Answer: N")
        log_dir: directory path for writing per-sample JSONL logs

    Returns:
        A tuple of metrics:
        (tput_mean, tput_std, tacc_mean, tacc_std,
         answer_accuracy, avg_draft_time, avg_target_time, peak_memory)
    """

    warmup_prompt = "Solve this math problem. Give the reasoning steps ...\nWhat is 1 + 1?"
    _run_warmup(
        generator,
        tokenizer,
        past_key_values,
        draft_past_key_values,
        args,
        warmup_prompt,
        max_length=args.max_length,
    )

    # 2. Main evaluation loop
    log_file = os.path.join(log_dir, "0.jsonl")

    # Lists to accumulate throughput, token acceptance, draft/target times
    perf = _init_perf()

    # Counters for overall question accuracy
    total_q = 0
    correct_q = 0

    # Regex to extract integers from the last line of outputs
    int_regex = re.compile(r"[-+]?\d+")

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating GSM8K"):
        prompt = entry["question"]
        ground_truth_text = entry["answer"]  # includes "Answer: N"

        # 2.1 Generate model output IDs (same as original)
        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        if input_ids.shape[1] > args.max_length:
            # Skip prompts that exceed max_length
            continue

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                temperature=args.temperature,
                max_length=args.max_length,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )

        reset_kv(past_key_values, draft_past_key_values)

        # 2.2 Extract original performance logs
        record = {**wandb_logger.log_data}
        record.update({
            "query": prompt,
            "response": tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip(),
            "answer": ground_truth_text.strip(),
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3)
        })

        # 2.3 Compute per-sample correctness
        output_str = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        lines = output_str.splitlines()
        last_line = lines[-1] if lines else output_str
        m_out = int_regex.search(last_line)
        pred_int = m_out.group(0).lstrip("+").lstrip("0") or "0" if m_out else None

        gt_lines = ground_truth_text.strip().splitlines()
        last_gt = gt_lines[-1]
        m_gt = int_regex.search(last_gt)
        gt_int = m_gt.group(0).lstrip("+").lstrip("0") or "0" if m_gt else None

        is_correct = (pred_int is not None and gt_int is not None and pred_int == gt_int)
        total_q += 1
        if is_correct:
            correct_q += 1

        # Include per-sample Accuracy flag in JSON record
        record["Accuracy"] = int(is_correct)

        # Append metrics lists
        _accum_perf(perf, record)

        # Write JSONL entry
        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        # Clean up
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate overall metrics
    answer_accuracy = correct_q / total_q if total_q > 0 else 0
    perf_stats = _finalize_perf(perf, generator)
    _print_summary("GSM8K", perf_stats, accuracy=answer_accuracy, correct_q=correct_q, total_q=total_q)

    # 5. Return metrics as a JSON-serializable dict for better scalability
    return {
        **perf_stats,
        "accuracy": float(answer_accuracy),
    }

def run_aime_eval(generator, tokenizer,
                  past_key_values, draft_past_key_values,
                  args, dataset, log_dir):
    """
    Evaluate AIME‑2024 accuracy alongside performance metrics.

    Args:
        generator:       model generator instance with .generate and .exp_log
        tokenizer:       tokenizer supporting .apply_chat_template and .decode
        past_key_values: primary past key values for autoregressive generation
        draft_past_key_values: optional speculative-decoding pasts
        args:            namespace with temperature, max_length, do_sample, warmup_iter
        dataset:         list of dicts, each with keys:
                         "question": the full prompt string
                         "answer"  : ground truth string (just the numeric answer)
        log_dir:         directory for per-sample JSONL logs

    Returns:
        (tput_mean, tput_std, tacc_mean, tacc_std,
         answer_accuracy, avg_draft_time, avg_target_time, peak_memory)
    """

    warmup_prompt = "Solve this math problem. Give the reasoning steps ...\nWhat is 1 + 1?"
    _run_warmup(
        generator,
        tokenizer,
        past_key_values,
        draft_past_key_values,
        args,
        warmup_prompt,
        max_length=args.max_length,
    )

    # 2. Main loop
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "aime_eval.jsonl")

    perf = _init_perf()
    total_q, correct_q = 0, 0
    int_regex = re.compile(r"[-+]?\d+")

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating AIME"):
        prompt = entry["question"]
        ground_truth = entry["answer"].strip()

        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(
            [{"role":"user","content":prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        if input_ids.shape[1] > args.max_length:
            continue

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                temperature=args.temperature,
                max_length=args.max_length,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )
        reset_kv(past_key_values, draft_past_key_values)

        response = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        # Build record
        record = {
            **wandb_logger.log_data,
            "query": prompt,
            "response": response,
            "answer": ground_truth,
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024**3)
        }

        # Extract integers
        pred_match = int_regex.search(response.splitlines()[-1])
        gt_match   = int_regex.search(ground_truth.splitlines()[-1])
        pred_int = pred_match.group(0).lstrip("+").lstrip("0") or "0" if pred_match else None
        gt_int   = gt_match.group(0).lstrip("+").lstrip("0") or "0" if gt_match else None

        is_correct = (pred_int is not None and gt_int is not None and pred_int == gt_int)
        total_q += 1
        if is_correct:
            correct_q += 1
        record["Accuracy"] = int(is_correct)

        # Aggregate perf metrics
        _accum_perf(perf, record)

        # Log per sample
        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        # Cleanup
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate overall
    accuracy = correct_q / total_q if total_q else 0
    perf_stats = _finalize_perf(perf, generator)
    _print_summary("AIME", perf_stats, accuracy=accuracy, correct_q=correct_q, total_q=total_q)

    # Return JSON-like dict for scalability
    return {
        **perf_stats,
        "accuracy": float(accuracy),
    }

def run_mmlu_pro_eval(generator, tokenizer,
                      past_key_values, draft_past_key_values,
                      args, dataset, log_dir):
    """
    Evaluate MMLU‑Pro multiple‑choice accuracy + perf metrics.
    `dataset` should be the list from load_mmlu_pro_dataset_answer().
    """
    warmup = "What is 1 + 1?"
    warmup_prompt = f"{warmup}\n\nA. 0\nB. 1\nC. 2\nD. 3\nE. 4\nF. 5\nG. 6\nH. 7\nI. 8\nJ. 9\n\nAnswer:"
    _run_warmup(
        generator,
        tokenizer,
        past_key_values,
        draft_past_key_values,
        args,
        warmup_prompt,
        max_length=args.max_length,
    )

    # 2. Main loop
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "mmlu_pro.jsonl")

    letter_re = re.compile(r"\b([A-J])\b")
    perf = _init_perf()
    total_q, correct_q = 0, 0

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Eval MMLU‑Pro"):
        prompt, gt = entry["question"], entry["answer"]
        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(
            [{"role":"user","content":prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)
        if input_ids.shape[1] > args.max_length:
            continue

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids, temperature=args.temperature,
                max_length=args.max_length, do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )
        reset_kv(past_key_values, draft_past_key_values)

        resp = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        # pick last non‑empty line
        last_line = next((l for l in reversed(resp.splitlines()) if l.strip()), resp)
        m = letter_re.search(last_line)
        pred = m.group(1) if m else None

        is_correct = (pred == gt)
        total_q += 1
        if is_correct: correct_q += 1

        # build record
        record = {
            **wandb_logger.log_data,
            "query": prompt,
            "response": resp,
            "answer": gt,
            "pred": pred,
            "Accuracy": int(is_correct),
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024**3)
        }
        # perf lists
        _accum_perf(perf, record)

        with open(log_file, "a+") as f:
            json.dump(record, f); f.write("\n")

        # cleanup
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate
    accuracy = correct_q/total_q if total_q else 0
    perf_stats = _finalize_perf(perf, generator)
    _print_summary("MMLU‑Pro", perf_stats, accuracy=accuracy, correct_q=correct_q, total_q=total_q)

    # Return JSON-like dict for scalability
    return {
        **perf_stats,
        "accuracy": float(accuracy),
    }



# --- Utility functions consolidated from lcb_runner ---

def _extract_code(text: str) -> str:
    """Extracts code from a ```python ... ``` block."""
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()

def _decode_test_cases(field: Any) -> List[Dict[str, str]]:
    """
    Robustly decodes LiveCodeBench public/private test-cases.
    This logic is critical for handling the various data formats.
    """
    if isinstance(field, list):
        return field

    if isinstance(field, bytes):
        s = field.decode("utf-8", errors="ignore").strip()
    else:
        s = str(field).strip()

    if s.lstrip().startswith("["):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass # Fall through

    try:
        data = base64.b64decode(s)
        if data.startswith(b'\x78\x9c'): # zlib compressed
            data = zlib.decompress(data)
        
        try: # Try JSON first
            return json.loads(data.decode("utf-8"))
        except: # Fall back to pickle
            return pickle.loads(data)
    except Exception as e:
        raise ValueError(f"Could not decode test case data: {e}") from None

def _run_single_test(python_src: str, test_case: dict, timeout: float) -> bool:
    """Runs a single test case against the provided Python source."""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_path = os.path.join(temp_dir, "main.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(python_src)

        try:
            proc = subprocess.run(
                ["python", code_path],
                input=test_case["input"].encode("utf-8"),
                capture_output=True,
                timeout=timeout,
            )
            # Compare stripped stdout to expected output
            return proc.stdout.decode("utf-8").strip() == test_case["output"].strip()
        except (subprocess.TimeoutExpired, Exception):
            return False

# --- Main function to replace the library call ---

def check_correctness(problem: dict, completion: str, timeout: float = 2.0) -> dict:
    """
    Self-contained function to grade a model's completion for a given problem.

    Args:
        problem: The problem dictionary from the dataset.
        completion: The string response generated by the model.
        timeout: Timeout in seconds for each test case.

    Returns:
        A dictionary with a "passed" boolean key.
    """
    solution_code = _extract_code(completion)
    if not solution_code:
        return {"passed": False}

    try:
        public_tests = _decode_test_cases(problem["public_test_cases"])
        private_tests = _decode_test_cases(problem["private_test_cases"])
        all_tests = public_tests + private_tests
    except ValueError:
        return {"passed": False} # Failed to decode tests

    for test_case in all_tests:
        if not _run_single_test(solution_code, test_case, timeout):
            return {"passed": False} # Failed a test case

    return {"passed": True} # Passed all test cases

def run_livecodebench_eval(
    generator,
    tokenizer,
    past_key_values,
    draft_past_key_values,
    args,
    dataset,
    log_dir,
    n_samples=1,
    test_timeout=2.0,
):
    """
    Refactored LiveCodeBench evaluation using the official lcb_runner package.
    """

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "livecodebench_eval_refactored.jsonl")

    # === 1) Warm-up (No changes needed here) ===
    # ... (Your warm-up code remains the same) ...
    print("Warm-up complete.")


    # === 2) Main loop (Simplified) ===
    perf = _init_perf()

    for i, problem in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating LiveCodeBench"):
        prompt = problem["prompt"] # Use the prompt from the loaded data

        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        if input_ids.shape[1] > args.max_length:
            continue

        graded_list = []
        responses = []
        for s in range(n_samples):
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output_ids = generator.generate(
                    input_ids,
                    temperature=args.temperature,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    past_key_values=past_key_values,
                    draft_past_key_values=draft_past_key_values
                )

            reset_kv(past_key_values, draft_past_key_values)

            response = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            responses.append(response)

            result = check_correctness(problem=problem, completion=response, timeout=test_timeout)
            graded_list.append(result["passed"])

        pass1 = int(graded_list[0] if graded_list else 0)
        
        record = {
            **wandb_logger.log_data,
            "query": prompt,
            "responses": responses,
            "graded_list": graded_list,
            "pass@1": pass1,
            "n": n_samples,
            "platform": problem.get("platform"),
            "difficulty": problem.get("difficulty"),
            "contest_date": problem.get("contest_date"),
            "question_id": problem.get("question_id"),
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3)
        }

        _accum_perf(perf, record)

    perf_stats = _finalize_perf(perf, generator)

    # Return JSON-like dict for scalability
    return {
        **perf_stats,
    }

# For longbench
def run_longbench_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir, bench_name):
    """
    Evaluate longbench dataset accuracy alongside performance metrics.
    Ex. "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", 
        "gov_report", "qmsum", "multi_news",  "trec", "triviaqa", "samsum",
        "passage_count", "passage_retrieval_en",  "lcc", "repobench_p"
    """
    print("bench name", bench_name)

    with open("run/pipelines/benchmarks/utils/config/dataset2maxlen.json", "r", encoding="utf-8") as f:
        benchmark_max_len = json.load(f)

    max_new_tokens = benchmark_max_len.get(bench_name, args.max_length)

    warmup_prompt = "Solve this math problem. Give the reasoning steps ...\nWhat is 1 + 1?" * 64
    _run_warmup(
        generator,
        tokenizer,
        past_key_values,
        draft_past_key_values,
        args,
        warmup_prompt,
        max_new_tokens=max_new_tokens,
        show_progress=True,
    )

    # Optional CUDA-graph capture for FlashInfer, after warmup (stabilizes kernels/allocations).
    maybe_init_cuda_graph_runner(generator, past_key_values, draft_past_key_values, args.device, args.warmup_iter)

    log_file = os.path.join(log_dir, "0.jsonl")
    perf = _init_perf()
    total_q = 0
    correct_q = 0

    for _, entry in tqdm(enumerate(dataset), total=len(dataset), desc=f"Evaluating {bench_name}"):
        prompt = entry["question"]
        ground_truth_list = entry["answer"]
        all_classes = entry.get("classes", None)

        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                temperature=args.temperature,
                max_new_tokens=max_new_tokens,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values,
            )

        reset_kv(past_key_values, draft_past_key_values)

        record = {**wandb_logger.log_data}
        record.update({
            "query": prompt,
            "response": tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            ),
            "answer": ground_truth_list,
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3),
        })

        response = record["response"]
        if bench_name in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = response.lstrip("\n").split("\n")[0]
        else:
            prediction = response

        score = 0
        for ground_truth in ground_truth_list:
            score = max(score, dataset2metric[bench_name](prediction, ground_truth, all_classes=all_classes))

        total_q += 1
        correct_q += score
        record["Accuracy"] = score

        _accum_perf(perf, record)

        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    answer_accuracy = round(100 * correct_q / total_q, 2) if total_q > 0 else 0
    perf_stats = _finalize_perf(perf, generator)
    _print_summary(bench_name, perf_stats, accuracy=answer_accuracy, correct_q=correct_q, total_q=total_q)

    return {
        **perf_stats,
    }