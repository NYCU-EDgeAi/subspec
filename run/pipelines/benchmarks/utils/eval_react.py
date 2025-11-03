import os, re, gc, json, logging
import numpy as np
from tqdm import trange, tqdm
import torch
from contextlib import nullcontext

# your imports (SDPA kernel etc.)
from torch.nn.attention import SDPBackend, sdpa_kernel

# === ReAct env & wrappers ===
from ..react.wikienv import WikiEnv                               
from ..react.wrappers import (                                     
    LoggingWrapper,
    HotPotQAWrapper,
    FeverWrapper,
    HistoryWrapper,
)

# ---- helpers ----

ACTION_RE = re.compile(r"(search\[.*?\]|lookup\[.*?\]|finish\[.*?\])", re.IGNORECASE | re.DOTALL)

def parse_action(text: str) -> str:
    """
    Extract the first valid ReAct action from the model output.
    Valid actions look like: search[...], lookup[...], finish[...]
    Falls back to a 'think[...]' no-op if nothing matches (env will reply 'Nice thought.').
    """
    m = ACTION_RE.search(text or "")
    if m:
        # normalize spacing; keep original bracket content
        a = m.group(1)
        # Lowercase the verb but preserve the bracket content
        verb = a.split("[", 1)[0].lower()
        return verb + "[" + a.split("[", 1)[1]
    return "think[let me reflect]"

def build_react_prompt(observation: str, history: str | None = None) -> str:
    sys = (
        """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
        (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
        (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
        (3) Finish[answer], which returns the answer and finishes the task.
        Here are some examples.
        """
    )
    if history:
        return f"{sys}\n{history}\nObservation: {observation}\nAction:"
    return f"{sys}\nObservation: {observation}\nAction:"

def decode_new_tokens(tokenizer, output_ids, prompt_len):
    return tokenizer.decode(output_ids[0][prompt_len:])

def maybe_reset_kv(past_key_values, draft_past_key_values):
    if past_key_values is not None:
        past_key_values.reset()
    if draft_past_key_values is not None:
        draft_past_key_values.reset()

# ---- main ----

def run_react_eval(
    generator,
    tokenizer,
    past_key_values,
    draft_past_key_values,
    args,
    dataset,
    log_dir,
    task="hotpot",           # "hotpot" | "fever"
    split="dev",             # dataset split used by wrappers
    num_episodes=5,        # how many questions to run
    max_steps=8,             # max ReAct steps per episode
    obs_format="history",    # "history" (recommended) | "obs"
    use_default_system_prompt=True,
):
    """
    ReAct evaluation loop using your WikiEnv + dataset wrappers.
    - Keeps your warmup, sdpa_kernel usage, KV resets, and exp_log aggregation
    - Logs one JSON object per episode to {log_dir}/react_{task}_{split}.jsonl
    - Reports throughput-ish stats from generator.exp_log + EM/F1 when available
    """

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"react_{task}_{split}.jsonl")

    # ---- 0) Warmup (same pattern as MT-Bench) ----
    is_profiling = generator.profiling
    generator.profiling = False
    for _ in trange(getattr(args, "warmup_iter", 2), desc="Warming up"):
        input_message = "Write an essay about large language models."
        messages = [{"role": "user", "content": input_message}]
        tokenizer.use_default_system_prompt = use_default_system_prompt
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).cuda(device=args.device)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            gc.collect()
            torch.cuda.empty_cache()
            generator.generate(
                input_ids,
                temperature=args.temperature,
                max_length=args.max_length,
                do_sample=args.do_sample,
                past_key_values=past_key_values,
                draft_past_key_values=draft_past_key_values,
            )
        maybe_reset_kv(past_key_values, draft_past_key_values)
    generator.profiling = is_profiling

    # ---- 1) Build env stack (WikiEnv -> dataset wrapper -> logging -> history view) ----
    base_env = WikiEnv()  # ReAct interface: search[], lookup[], finish[]  :contentReference[oaicite:4]{index=4}

    if task.lower() == "hotpot":
        env = HotPotQAWrapper(base_env, split=split)  # uses data/HOTPOT files     :contentReference[oaicite:5]{index=5}
    elif task.lower() == "fever":
        env = FeverWrapper(base_env, split=split)      # uses data/FEVER files      :contentReference[oaicite:6]{index=6}
    else:
        raise ValueError(f"Unsupported task: {task}")

    env = LoggingWrapper(env, folder=os.path.join(log_dir, "trajs"))            # :contentReference[oaicite:7]{index=7}
    # env = HistoryWrapper(env, obs_format=obs_format, prompt=None)               # :contentReference[oaicite:8]{index=8}

    # ---- 2) Run episodes ----
    tput_list, acc_rate_list, draft_time_list, target_time_list = [], [], [], []
    em_list, f1_list = [], []
    peak_mem = 0.0

    for ep in tqdm(range(num_episodes), desc=f"ReAct {task}/{split}"):
        # reset episode (dataset wrappers select a random idx by default)
        obs = env.reset(idx = ep)
        messages = []

        # step loop
        for step in range(max_steps):
            # format either "history" (previous actions+observations) or just the latest obs
            if obs_format == "history":
                # HistoryWrapper already prepends the history into the observation stream
                # so we can pass the observation directly
                prompt_text = build_react_prompt(obs)
            else:
                prompt_text = build_react_prompt(obs)

            messages = [{"role": "user", "content": prompt_text}]
           
            tokenizer.use_default_system_prompt = use_default_system_prompt
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).cuda(device=args.device)
           
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output_ids = generator.generate(
                    input_ids,
                    temperature=args.temperature,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    past_key_values=past_key_values,
                    draft_past_key_values=draft_past_key_values,
                    stop_strings=["\nObservation", "Observation:"],
                )
            output_message = tokenizer.decode(output_ids[0][input_ids.shape[1]:])
            print(f"Step {step} Output:\n{output_message}\n")
            
            # decode only the newly generated span
            action_text = decode_new_tokens(tokenizer, output_ids, prompt_len=input_ids.shape[1])
            thought, action = action_text.strip().split(f"\n**Action:**")
            print("----------------------------------------------------------------")
            print(f"Step {step} Parsed Action: {action}\n")
            # env step
            
            obs, reward, done, info = env.step(action)
            print("----------------------------------------------------------------")
            print(f"Step {step} Env Response:\nObservation: {obs}\nReward: {reward}\nDone: {done}\nInfo: {info}\n")
            exit(1)

            # accumulate low-level timings from your generator.exp_log (same style as MT-Bench)
            n_iter = generator.exp_log.get('n_iter', 0)
            n_tokens = generator.exp_log.get('n_tokens', 0)
            elapsed_time = generator.exp_log.get('elapsed_time', 0)

            # free tensors
            del input_ids, output_ids
            gc.collect()
            torch.cuda.empty_cache()

            if done:
                break

        # hard stop if still not finished after max_steps
        if info.get("answer") is None:
            # try to finalize with the last observed text to avoid dangling episodes
            env.step("finish[unknown]")

        # update logs for this episode
        env.update_record()  # make sure LoggingWrapper captures the traj  :contentReference[oaicite:9]{index=9}

        # compute metrics if provided by wrapper (EM/F1 on Hotpot; EM on FEVER)
        # wrappers already inject em/f1/reward into info when done
        em = info.get("em", 0)
        f1 = info.get("f1", 0)
        em_list.append(em)
        f1_list.append(f1)

        # roll up per-episode averages like your MT-Bench code
        total_sampled = round(generator.exp_log.get('avg_sampled', 0) * generator.exp_log.get('n_iter', 0))
        total_draft_time = generator.exp_log.get('avg_draft_time', 0) * generator.exp_log.get('n_iter', 0)
        total_target_time = generator.exp_log.get('avg_target_time', 0) * generator.exp_log.get('n_iter', 0)
        total_verify_time = generator.exp_log.get('avg_verify_time', 0) * generator.exp_log.get('n_iter', 0)

        overall = {
            "avg_draft_time": (total_draft_time / n_iter) if n_iter else 0,
            "avg_target_time": (total_target_time / n_iter) if n_iter else 0,
            "avg_verify_time": (total_verify_time / n_iter) if n_iter else 0,
            "n_iter": n_iter,
            "n_tokens": n_tokens,
            "avg_sampled": (total_sampled / n_iter) if n_iter else 0,
            "elapsed_time": elapsed_time,
            "tput": (n_tokens / elapsed_time) if elapsed_time else 0,
        }

        peak_mem = max(peak_mem, torch.cuda.max_memory_reserved(args.device) / (1024**3))
        if overall.get("tput") is not None:
            tput_list.append(overall.get("tput", 0))
        if overall.get("avg_sampled") is not None:
            acc_rate_list.append(overall.get("avg_sampled", 0))
        if overall.get("avg_draft_time") is not None:
            draft_time_list.append(overall.get("avg_draft_time", 0))
        if overall.get("avg_target_time") is not None:
            target_time_list.append(overall.get("avg_target_time", 0))

        # per-episode log line (trajectory + metrics)
        episode_log = {
            "episode": ep,
            "task": task,
            "split": split,
            "question": info.get("question"),
            "gt_answer": info.get("gt_answer"),
            "pred_answer": info.get("answer"),
            "metrics": {"em": em, "f1": f1, "reward": info.get("reward", 0)},
            "overall": overall,
            "peak_mem_GiB": torch.cuda.max_memory_reserved(args.device) / (1024**3),
        }
        with open(log_path, "a+", encoding="utf-8") as f:
            f.write(json.dumps(episode_log, ensure_ascii=False) + "\n")

        # reset caches for next episode
        maybe_reset_kv(past_key_values, draft_past_key_values)

    # ---- 3) Final aggregate printout ----
    tput_mean, tput_std = (np.mean(tput_list) if tput_list else 0.0), (np.std(tput_list) if tput_list else 0.0)
    acc_rate_mean, acc_rate_std = (np.mean(acc_rate_list) if acc_rate_list else 0.0), (np.std(acc_rate_list) if acc_rate_list else 0.0)
    avg_draft_time = np.mean(draft_time_list) if draft_time_list else 0.0
    avg_target_time = np.mean(target_time_list) if target_time_list else 0.0
    em_mean = np.mean(em_list) if em_list else 0.0
    f1_mean = np.mean(f1_list) if f1_list else 0.0

    print("Final Results:")
    print(f"\tThroughput: {tput_mean:.3f} ± {tput_std:.3f} tokens/sec")
    print(f"\tAcceptance rate: {acc_rate_mean:.3f} ± {acc_rate_std:.3f} tokens/iter")
    print(f"\tAverage Draft Time: {avg_draft_time:.3f} sec")
    print(f"\tAverage Target Time: {avg_target_time:.3f} sec")
    if task.lower() == "hotpot":
        print(f"\tExact Match (EM): {em_mean:.3f} | F1: {f1_mean:.3f}")
    elif task.lower() == "fever":
        print(f"\tLabel Accuracy: {em_mean:.3f}")
    print(f"\tPeak Memory: {peak_mem:.3f} GiB")

    return {
        "tput_mean": tput_mean,
        "tput_std": tput_std,
        "acc_rate_mean": acc_rate_mean,
        "acc_rate_std": acc_rate_std,
        "avg_draft_time": avg_draft_time,
        "avg_target_time": avg_target_time,
        "peak_mem_GiB": peak_mem,
        "EM": em_mean,
        "F1": f1_mean,
        "log_path": log_path,
    }
