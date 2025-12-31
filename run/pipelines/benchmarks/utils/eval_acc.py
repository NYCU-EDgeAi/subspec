import os
import json
import re
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

    # 1. Warm-up (identical to original implementation)
    original_profiling = generator.profiling
    generator.profiling = False
    for _ in range(args.warmup_iter):
        warmup_prompt = "Solve this math problem. Give the reasoning steps ...\nWhat is 1 + 1?"
        tokenizer.use_default_system_prompt = True
        warmup_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": warmup_prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            generator.generate(
                warmup_ids,
                temperature=args.temperature,
                max_length=args.max_length,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )

        reset_kv(past_key_values, draft_past_key_values)
    generator.profiling = original_profiling

    # 2. Main evaluation loop
    log_file = os.path.join(log_dir, "0.jsonl")

    # Lists to accumulate throughput, token acceptance, draft/target times
    tput_list = []
    tacc_list = []  # average token acceptance rate per sample
    draft_times = []
    target_times = []

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
        if record.get("tput") is not None:
            tput_list.append(record.get("tput", 0))
        if record.get("avg_sampled") is not None:
            tacc_list.append(record.get("avg_sampled", 0))
        if record.get("avg_draft_time") is not None:
            draft_times.append(record.get("avg_draft_time", 0))
        if record.get("avg_target_time") is not None:
            target_times.append(record.get("avg_target_time", 0))

        # Write JSONL entry
        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        # Clean up
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate overall metrics
    tput_mean, tput_std = (np.mean(tput_list), np.std(tput_list)) if tput_list else (0, 0)
    tacc_mean, tacc_std = (np.mean(tacc_list), np.std(tacc_list)) if tacc_list else (0, 0)
    answer_accuracy = correct_q / total_q if total_q > 0 else 0
    avg_draft = np.mean(draft_times) if draft_times else 0
    avg_target = np.mean(target_times) if target_times else 0
    peak_memory = torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3)

    # 4. Print summary
    print("Final GSM8K Results:")
    print(f"\tThroughput       : {tput_mean:.3f} ± {tput_std:.3f} tokens/sec")
    print(f"\tToken Acceptance : {tacc_mean:.3f} ± {tacc_std:.3f}")
    print(f"\tAnswer Accuracy  : {answer_accuracy:.3f} ({correct_q}/{total_q})")
    print(f"\tAvg Draft Time   : {avg_draft:.3f} sec")
    print(f"\tAvg Target Time  : {avg_target:.3f} sec")
    print(f"\tPeak Memory      : {peak_memory:.3f} GiB")
    if hasattr(generator, "judge_acc_len_list"):
        print(f"\tTacc_judge       : {np.mean(generator.judge_acc_len_list):.3f}")
    else:
        print("\tTacc_judge       : 0.000 (not available)")

    # 5. Return metrics as a JSON-serializable dict for better scalability
    return {
        "tput_mean": float(tput_mean),
        "tput_std": float(tput_std),
        "tacc_mean": float(tacc_mean),
        "tacc_std": float(tacc_std),
        "accuracy": float(answer_accuracy),
        "avg_draft_time": float(avg_draft),
        "avg_target_time": float(avg_target),
        "peak_memory_gib": float(peak_memory),
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

    # 1. Warm‑up
    original_profiling = generator.profiling
    generator.profiling = False
    for _ in range(args.warmup_iter):
        warmup_prompt = "Solve this math problem. Give the reasoning steps ...\nWhat is 1 + 1?"
        tokenizer.use_default_system_prompt = True
        warmup_ids = tokenizer.apply_chat_template(
            [{"role":"user","content":warmup_prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            generator.generate(
                warmup_ids,
                temperature=args.temperature,
                max_length=args.max_length,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )

        reset_kv(past_key_values, draft_past_key_values)
    generator.profiling = original_profiling

    # 2. Main loop
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "aime_eval.jsonl")

    tput_list, tacc_list = [], []
    draft_times, target_times = [], []
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
        if record.get("tput") is not None:       tput_list.append(record["tput"])
        if record.get("avg_sampled") is not None: tacc_list.append(record["avg_sampled"])
        if record.get("avg_draft_time") is not None: draft_times.append(record["avg_draft_time"])
        if record.get("avg_target_time") is not None: target_times.append(record["avg_target_time"])

        # Log per sample
        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        # Cleanup
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate overall
    tput_mean, tput_std = (np.mean(tput_list), np.std(tput_list)) if tput_list else (0,0)
    tacc_mean, tacc_std = (np.mean(tacc_list), np.std(tacc_list)) if tacc_list else (0,0)
    accuracy = correct_q / total_q if total_q else 0
    avg_draft  = np.mean(draft_times)  if draft_times  else 0
    avg_target = np.mean(target_times) if target_times else 0
    peak_mem   = torch.cuda.max_memory_reserved(generator.device) / (1024**3)

    print("Final AIME Results:")
    print(f"\tThroughput       : {tput_mean:.3f} ± {tput_std:.3f} tokens/sec")
    print(f"\tToken Acceptance : {tacc_mean:.3f} ± {tacc_std:.3f}")
    print(f"\tAnswer Accuracy  : {accuracy:.3f} ({correct_q}/{total_q})")
    print(f"\tAvg Draft Time   : {avg_draft:.3f} sec")
    print(f"\tAvg Target Time  : {avg_target:.3f} sec")
    print(f"\tPeak Memory      : {peak_mem:.3f} GiB")

    # Return JSON-like dict for scalability
    return {
        "tput_mean": float(tput_mean),
        "tput_std": float(tput_std),
        "tacc_mean": float(tacc_mean),
        "tacc_std": float(tacc_std),
        "accuracy": float(accuracy),
        "avg_draft_time": float(avg_draft),
        "avg_target_time": float(avg_target),
        "peak_memory_gib": float(peak_mem),
    }

# WARNING: This function is NOT ready
def run_mmlu_pro_eval(generator, tokenizer,
                      past_key_values, draft_past_key_values,
                      args, dataset, log_dir):
    """
    Evaluate MMLU‑Pro multiple‑choice accuracy + perf metrics.
    `dataset` should be the list from load_mmlu_pro_dataset_answer().
    """
    # 1. Warmup
    orig_prof = generator.profiling
    generator.profiling = False
    warmup = "What is 1 + 1?"
    warmup_prompt = f"{warmup}\n\nA. 0\nB. 1\nC. 2\nD. 3\nE. 4\nF. 5\nG. 6\nH. 7\nI. 8\nJ. 9\n\nAnswer:"
    for _ in range(args.warmup_iter):
        tokenizer.use_default_system_prompt = True
        ids = tokenizer.apply_chat_template(
            [{"role":"user","content":warmup_prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            generator.generate(
                ids, temperature=args.temperature,
                max_length=args.max_length, do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )
        reset_kv(past_key_values, draft_past_key_values)
    generator.profiling = orig_prof

    # 2. Main loop
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "mmlu_pro.jsonl")

    letter_re = re.compile(r"\b([A-J])\b")
    tput_list, tacc_list = [], []
    draft_times, target_times = [], []
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
        if record.get("tput")        is not None: tput_list.append(record["tput"])
        if record.get("avg_sampled") is not None: tacc_list.append(record["avg_sampled"])
        if record.get("avg_draft_time"): draft_times.append(record["avg_draft_time"])
        if record.get("avg_target_time"): target_times.append(record["avg_target_time"])

        with open(log_file, "a+") as f:
            json.dump(record, f); f.write("\n")

        # cleanup
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate
    tput_mean, tput_std = (np.mean(tput_list), np.std(tput_list)) if tput_list else (0,0)
    tacc_mean, tacc_std = (np.mean(tacc_list), np.std(tacc_list)) if tacc_list else (0,0)
    accuracy = correct_q/total_q if total_q else 0
    avg_draft_time  = np.mean(draft_times)  if draft_times  else 0
    avg_target_time = np.mean(target_times) if target_times else 0
    peak_mem   = torch.cuda.max_memory_reserved(generator.device)/(1024**3)

    print("Final MMLU‑Pro Results:")
    print(f"\tThroughput       : {tput_mean:.3f} ± {tput_std:.3f} tokens/sec")
    print(f"\tToken Acceptance : {tacc_mean:.3f} ± {tacc_std:.3f}")
    print(f"\tAnswer Accuracy  : {accuracy:.3f} ({correct_q}/{total_q})")
    print(f"\tAvg Draft Time   : {avg_draft_time:.3f} sec")
    print(f"\tAvg Target Time  : {avg_target_time:.3f} sec")
    print(f"\tPeak Memory      : {peak_mem:.3f} GiB")

    # Return JSON-like dict for scalability
    return {
        "tput_mean": float(tput_mean),
        "tput_std": float(tput_std),
        "tacc_mean": float(tacc_mean),
        "tacc_std": float(tacc_std),
        "accuracy": float(accuracy),
        "avg_draft_time": float(avg_draft_time),
        "avg_target_time": float(avg_target_time),
        "peak_memory_gib": float(peak_mem),
    }



import re
import json
import base64
import zlib
import pickle
import subprocess
import os
import tempfile
from typing import Any, List, Dict
import time

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
    tput_list, tacc_list = [], []
    draft_times, target_times = [], []

    totals, corrects = [], []
    easy_totals, easy_corrects = [], []
    med_totals, med_corrects = [], []
    hard_totals, hard_corrects = [], []

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
        timings = []

        for s in range(n_samples):
            start = time.time()
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output_ids = generator.generate(
                    input_ids,
                    temperature=args.temperature,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    past_key_values=past_key_values,
                    draft_past_key_values=draft_past_key_values
                )
            gen_time = time.time() - start

            reset_kv(past_key_values, draft_past_key_values)

            response = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            responses.append(response)

            result = check_correctness(problem=problem, completion=response, timeout=test_timeout)
            graded_list.append(result["passed"])
            
            timings.append(gen_time)

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

    tput_mean, tput_std = (np.mean(tput_list), np.std(tput_list)) if tput_list else (0, 0)
    tacc_mean, tacc_std = (np.mean(tacc_list), np.std(tacc_list)) if tacc_list else (0, 0)
    avg_draft_time = np.mean(draft_times) if draft_times else 0
    avg_target_time = np.mean(target_times) if target_times else 0
    peak_memory = torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3)

    # Return JSON-like dict for scalability
    return {
        "tput_mean": float(tput_mean),
        "tput_std": float(tput_std),
        "tacc_mean": float(tacc_mean),
        "tacc_std": float(tacc_std),
        "avg_draft_time": float(avg_draft_time),
        "avg_target_time": float(avg_target_time),
        "peak_memory_gib": float(peak_memory),
    }

# For longbench
def run_longbench_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir, bench_name):
    """
    Evaluate longbench dataset accuracy alongside performance metrics.
    Ex. "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", 
        "gov_report", "qmsum", "multi_news",  "trec", "triviaqa", "samsum",
        "passage_count", "passage_retrieval_en",  "lcc", "repobench_p"

    Args:
        generator: the model generator instance
        tokenizer: tokenizer with chat template functionality
        past_key_values: primary past key values for autoregressive generation
        draft_past_key_values: draft past key values for speculative decoding (optional)
        args: namespace containing temperature, max_length, do_sample, warmup_iter
        dataset: list of dicts, each with keys:
            "question": the prompt string
            "answer": full original answer text from longbench (with reasoning and final line "Answer: N")
        log_dir: directory path for writing per-sample JSONL logs
        bench_name: name of benchmarks Ex. "narrativeqa", "qasper"...
        max_len: max_len of LLM Ex. llama3: 127500

    Returns:
        A tuple of metrics:
        (tput_mean, tput_std, tacc_mean, tacc_std,
         answer_accuracy, avg_draft_time, avg_target_time, peak_memory)
    """
    print("bench name", bench_name)
    
    # 0. load max_length limit for longbench eval
    with open("run/pipelines/benchmarks/utils/config/dataset2maxlen.json", "r", encoding="utf-8") as f:
        benchmark_max_len = json.load(f)

    if bench_name in benchmark_max_len:
        max_new_tokens = benchmark_max_len[bench_name]
    else: 
        max_new_tokens = args.max_length

    # 1. Warm-up (identical to original implementation)
    original_profiling = generator.profiling
    generator.profiling = False
    for _ in tqdm(range(args.warmup_iter)):
        warmup_prompt = "Solve this math problem. Give the reasoning steps ...\nWhat is 1 + 1?" * 64
        tokenizer.use_default_system_prompt = True
        
        # if bench_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench_p"]:
        #     warmup_prompt = build_chat(warmup_prompt)

        # tokenized_prompt = tokenizer(warmup_prompt, truncation=False, return_tensors="pt").input_ids[0]

        # warmup_ids = tokenizer(warmup_prompt, truncation=False, return_tensors="pt").input_ids.to(generator.device)\
        warmup_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": warmup_prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            generator.generate(
                warmup_ids,
                temperature=args.temperature,
                max_new_tokens=max_new_tokens,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )

        reset_kv(past_key_values, draft_past_key_values)
    generator.profiling = original_profiling

    # Optional CUDA-graph capture for FlashInfer, after warmup (stabilizes kernels/allocations).
    maybe_init_cuda_graph_runner(generator, past_key_values, draft_past_key_values, args.device, args.warmup_iter)
        
    # 2. Main evaluation loop
    log_file = os.path.join(log_dir, "0.jsonl")

    # Lists to accumulate throughput, token acceptance, draft/target times
    tput_list = []
    tacc_list = []  # average token acceptance rate per sample
    draft_times = []
    target_times = []

    # Counters for overall question accuracy
    total_q = 0
    correct_q = 0

    # Regex to extract integers from the last line of outputs
    int_regex = re.compile(r"[-+]?\d+")

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating "+bench_name):
        prompt = entry["question"]
        ground_truth_list = entry["answer"]  # includes "[Answer: N, ...]"
        if 'classes' in entry:
            all_classes = entry["classes"]
        else:
            all_classes = None

        # 2.1 Generate model output IDs (same as original)
        tokenizer.use_default_system_prompt = True
        
        # old
        # ----------------------------------------------------------
        # if bench_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench_p"]:
        #     prompt = build_chat(prompt)
    
        # tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # min_length = args.min_length if hasattr(args, "min_length") else 0
        # if len(tokenized_prompt) > args.max_length or len(tokenized_prompt) < min_length:
        #     print(f"len(tokenized_prompt) = {len(tokenized_prompt)}, Skip it!")
        #     continue
        # elif len(tokenized_prompt) > max_len:
        #     half = max_len//2
        #     #prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        #     tokenized_prompt = torch.cat([tokenized_prompt[:half], tokenized_prompt[-half:]], dim=0)

        # #input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(generator.device)
        # input_ids = tokenized_prompt.unsqueeze(0).to(generator.device)
        # ----------------------------------------------------------

        # new
        # ----------------------------------------------------------
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(generator.device)
        # ----------------------------------------------------------

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                temperature=args.temperature,
                max_new_tokens=max_new_tokens,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values
            )

        reset_kv(past_key_values, draft_past_key_values)

        # 2.2 Extract original performance logs
        record = {**generator.exp_log}
        record.update({
            "query": prompt,
            "response": tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            ),
            "answer": ground_truth_list,
            "peak_memory": torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3)
        })

        # 2.3 Compute per-sample correctness
        response = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        if bench_name in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = response.lstrip('\n').split('\n')[0]
        else:
            prediction = response
        

        score = 0
        for ground_truth in ground_truth_list:
            score = max(score, dataset2metric[bench_name](prediction, ground_truth, all_classes=all_classes))

        total_q += 1
        correct_q += score

        # Include per-sample Score flag in JSON record
        record["Accuracy"] = score

        # Append metrics lists
        if record.get("tput") is not None:
            tput_list.append(record.get("tput", 0))
        if record.get("avg_sampled") is not None:
            tacc_list.append(record.get("avg_sampled", 0))
        if record.get("avg_draft_time") is not None:
            draft_times.append(record.get("avg_draft_time", 0))
        if record.get("avg_target_time") is not None:
            target_times.append(record.get("avg_target_time", 0))

        # Write JSONL entry
        with open(log_file, "a+") as f:
            json.dump(record, f)
            f.write("\n")

        # Clean up
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate overall metrics
    tput_mean, tput_std = (np.mean(tput_list), np.std(tput_list)) if tput_list else (0, 0)
    tacc_mean, tacc_std = (np.mean(tacc_list), np.std(tacc_list)) if tacc_list else (0, 0)
    answer_accuracy = round(100 * correct_q / total_q, 2) if total_q > 0 else 0
    avg_draft_time = np.mean(draft_times) if draft_times else 0
    avg_target_time = np.mean(target_times) if target_times else 0
    peak_memory = torch.cuda.max_memory_reserved(generator.device) / (1024 ** 3)

    # 4. Print summary
    print(f"Final {bench_name} Results:")
    print(f"\tThroughput       : {tput_mean:.3f} ± {tput_std:.3f} tokens/sec")
    print(f"\tToken Acceptance : {tacc_mean:.3f} ± {tacc_std:.3f}")
    print(f"\tAnswer Accuracy  : {answer_accuracy:.3f} ({correct_q}/{total_q})")
    print(f"\tAvg Draft Time   : {avg_draft_time:.3f} sec")
    print(f"\tAvg Target Time  : {avg_target_time:.3f} sec")
    print(f"\tPeak Memory      : {peak_memory:.3f} GiB")
    if hasattr(generator, "judge_acc_len_list"):
        print(f"\tTacc_judge       : {np.mean(generator.judge_acc_len_list):.3f}")
    else:
        print("\tTacc_judge       : 0.000 (not available)")

    # 5. Return metrics tuple
    # return (
    #     tput_mean,
    #     tput_std,
    #     tacc_mean,
    #     tacc_std,
    #     answer_accuracy,
    #     avg_draft,
    #     avg_target,
    #     peak_memory
    # )
    
    return {
        "tput_mean": float(tput_mean),
        "tput_std": float(tput_std),
        "avg_draft_time": float(avg_draft_time),
        "avg_target_time": float(avg_target_time),
        "peak_memory_gib": float(peak_memory),
    }