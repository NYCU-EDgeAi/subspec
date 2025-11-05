from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm, trange
import os
import json
import numpy as np
import torch
import gc
import logging

from smolagents import CodeAgent, ToolCallingAgent
from specdecodes.helpers.wrappers import SpecDecodesModel

def run_common_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir):
    # Warm up the model
    is_profiling = generator.profiling
    generator.profiling = False
    for i in trange(args.warmup_iter, desc='Warming up'):
        input_message = "Write an essay about large language models."
        messages = [{"role": "user", "content": input_message}]
        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda(device=args.device)
        torch.cuda.empty_cache()
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            gc.collect()
            torch.cuda.empty_cache()
            generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)

        past_key_values.reset()
        
        if draft_past_key_values is not None:
            draft_past_key_values.reset()
    generator.profiling = is_profiling
    
    # Evaluate dataset
    log_file = os.path.join(log_dir, f"0.jsonl")
    tput_list, acc_rate_list, draft_time_list, target_time_list = [], [], [], []
    for idx, query in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating", leave=True):
        messages = [{"role": "user", "content": query}]
        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(generator.device)
        # logging.info(f"Check shape {input_ids.shape[1]}")
        
        if input_ids.shape[1] > args.max_length:
            logging.info(f"Skipping query No.{idx} due to length {input_ids.shape[1]} > {args.max_length}")
            continue
        
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)
            
        past_key_values.reset()
        if draft_past_key_values is not None:
            draft_past_key_values.reset()

        output_message = tokenizer.decode(output_ids[0][input_ids.shape[1]:])
        exp_log = {**generator.exp_log, "query": query, "response": output_message, "peak_mem": torch.cuda.max_memory_reserved(args.device)/(1024**3)}
        with open(log_file, 'a+') as f:
            json.dump(exp_log, f, indent=4)
            f.write("\n")

        if exp_log.get("tput", None) is not None:
            tput_list.append(exp_log.get("tput", 0))
        if exp_log.get("avg_sampled", None) is not None:
            acc_rate_list.append(exp_log.get("avg_sampled", 0))
        if exp_log.get("avg_draft_time", None) is not None:
            draft_time_list.append(exp_log.get("avg_draft_time", 0))
        if exp_log.get("avg_target_time", None) is not None:
            target_time_list.append(exp_log.get("avg_target_time", 0))
            
        del input_ids, output_ids
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"Final Results:")
    tput_mean, tput_std = np.mean(tput_list), np.std(tput_list)
    acc_rate_mean, acc_rate_std = np.mean(acc_rate_list), np.std(acc_rate_list) if acc_rate_list else 0
    avg_draft_time, avg_target_time = np.mean(draft_time_list), np.mean(target_time_list)
    peak_mem = torch.cuda.max_memory_reserved(args.device)/(1024**3)
    
    print(f"\tThroughput: {tput_mean:.3f} ± {tput_std:.3f} tokens/sec")
    print(f"\tAcceptance rate: {acc_rate_mean:.3f} ± {acc_rate_std:.3f} tokens/iter")
    print(f"\tAverage Draft Time: {avg_draft_time:.3f} sec")
    print(f"\tAverage Target Time: {avg_target_time:.3f} sec")
    print(f"\tPeak Memory: {peak_mem:.3f} GiB")
    
    return tput_mean, tput_std, acc_rate_mean, acc_rate_std, avg_draft_time, avg_target_time, peak_mem

def run_mtbench_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir):
    # Warm up the model
    is_profiling = generator.profiling
    generator.profiling = False
    for i in trange(args.warmup_iter, desc='Warming up'):
        input_message = "Write an essay about large language models."
        messages = [{"role": "user", "content": input_message}]
        tokenizer.use_default_system_prompt = True
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda(device=args.device)
        
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            gc.collect()
            torch.cuda.empty_cache()
            generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)

        past_key_values.reset()
        
        if draft_past_key_values is not None:
            draft_past_key_values.reset()
    generator.profiling = is_profiling

    # Evaluate dataset
    log_file = os.path.join(log_dir, f"0.jsonl")
    tput_list, acc_rate_list, draft_time_list, target_time_list = [], [], [], []
    for idx, turns in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating", leave=True):
        # org_len = 0
        exp_log = {}
        tmp_exp_log = {'total_sampled': 0, 'total_draft_time': 0, 'total_target_time': 0, 'total_verify_time': 0, 'n_iter': 0, 'n_tokens': 0, 'elapsed_time': 0}
        messages = []
        for tid, query in enumerate(turns):
            # print(f"Turn {tid+1}/{len(turns)} -> {query}"
            messages.append({"role": "user", "content": query})
            tokenizer.use_default_system_prompt = True
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda(device=args.device)
            # logging.info(f"Check shape {input_ids.shape[1]}")
                    
            if input_ids.shape[1] > args.max_length:
                logging.info(f"Skipping query No.{idx} (turn {tid}) due to length {input_ids.shape[1]} > {args.max_length}")
                continue
            
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                output_ids = generator.generate(input_ids, temperature=args.temperature, max_length=args.max_length, do_sample=args.do_sample, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values)
            
            output_message = tokenizer.decode(output_ids[0][input_ids.shape[1]:])

            n_iter = generator.exp_log.get('n_iter', 0)
            n_tokens = generator.exp_log.get('n_tokens', 0)
            elapsed_time = generator.exp_log.get('elapsed_time', 0)
            
            tmp_exp_log['n_iter'] += n_iter
            tmp_exp_log['n_tokens'] += n_tokens
            tmp_exp_log['elapsed_time'] += elapsed_time
            
            tmp_exp_log['total_sampled'] += np.round(generator.exp_log.get('avg_sampled', 0) * n_iter, decimals=0)
            tmp_exp_log['total_draft_time'] += generator.exp_log.get('avg_draft_time', 0) * n_iter
            tmp_exp_log['total_target_time'] += generator.exp_log.get('avg_target_time', 0) * n_iter
            tmp_exp_log['total_verify_time'] += generator.exp_log.get('avg_verify_time', 0) * n_iter
            
            exp_log = {**exp_log, tid: {**generator.exp_log, "query": query, "response": output_message, "peak_mem": torch.cuda.max_memory_reserved(args.device)/(1024**3)}}
            messages.append({"role": "system", "content": output_message})
            
            del input_ids, output_ids
            gc.collect()
            torch.cuda.empty_cache()
        
        past_key_values.reset()
        if draft_past_key_values is not None:
            draft_past_key_values.reset()
        
        # output_message = tokenizer.decode(output_ids[0][input_ids.shape[1]:])
        overall_log = {
            "avg_draft_time": tmp_exp_log['total_draft_time'] / tmp_exp_log['n_iter'] if tmp_exp_log['n_iter'] > 0 else 0,
            "avg_target_time": tmp_exp_log['total_target_time'] / tmp_exp_log['n_iter'] if tmp_exp_log['n_iter'] > 0 else 0,
            "avg_verify_time": tmp_exp_log['total_verify_time'] / tmp_exp_log['n_iter'] if tmp_exp_log['n_iter'] > 0 else 0,
            "n_iter": tmp_exp_log['n_iter'], 
            "n_tokens": tmp_exp_log['n_tokens'], 
            "avg_sampled": tmp_exp_log['total_sampled'] / tmp_exp_log['n_iter'] if tmp_exp_log['n_iter'] > 0 else 0,
            "elapsed_time": tmp_exp_log['elapsed_time'],
            "tput": tmp_exp_log['n_tokens'] / tmp_exp_log['elapsed_time']                    
        }
        
        exp_log = {
            **exp_log,
            "overall": overall_log
        }
        
        with open(log_file, 'a+') as f:
            json.dump(exp_log, f, indent=4)
            f.write("\n")

        if overall_log.get("tput", None) is not None:
            tput_list.append(overall_log.get("tput", 0))
        if overall_log.get("avg_sampled", None) is not None:
            acc_rate_list.append(overall_log.get("avg_sampled", 0))
        if overall_log.get("avg_draft_time", None) is not None:
            draft_time_list.append(overall_log.get("avg_draft_time", 0))
        if overall_log.get("avg_target_time", None) is not None:
            target_time_list.append(overall_log.get("avg_target_time", 0))
    
    print(f"Final Results:")
    tput_mean, tput_std = np.mean(tput_list), np.std(tput_list)
    acc_rate_mean, acc_rate_std = np.mean(acc_rate_list), np.std(acc_rate_list) if acc_rate_list else 0
    avg_draft_time, avg_target_time = np.mean(draft_time_list), np.mean(target_time_list)
    peak_mem = torch.cuda.max_memory_reserved(args.device)/(1024**3)
    
    print(f"\tThroughput: {tput_mean:.3f} ± {tput_std:.3f} tokens/sec")
    print(f"\tAcceptance rate: {acc_rate_mean:.3f} ± {acc_rate_std:.3f} tokens/iter")
    print(f"\tAverage Draft Time: {avg_draft_time:.3f} sec")
    print(f"\tAverage Target Time: {avg_target_time:.3f} sec")
    print(f"\tPeak Memory: {peak_mem:.3f} GiB")
    
    return tput_mean, tput_std, acc_rate_mean, acc_rate_std, avg_draft_time, avg_target_time, peak_mem

def run_agent_eval(generator, tokenizer, past_key_values, draft_past_key_values, args, dataset, log_dir):
    # Build agent
    smolmodel = SpecDecodesModel(generator=generator, tokenizer=tokenizer, past_key_values=past_key_values, draft_past_key_values=draft_past_key_values, max_length=args.max_length, temperature=args.temperature, do_sample=args.do_sample, device=args.device)
    agent = ToolCallingAgent(tools=[], model=smolmodel, add_base_tools=True)
    
    # Warm up the model
    is_profiling = generator.profiling
    generator.profiling = False
    for i in trange(args.warmup_iter, desc='Warming up'):
        input_message = "Write an essay about large language models."
        tokenizer.use_default_system_prompt = True
        torch.cuda.empty_cache()
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            gc.collect()
            torch.cuda.empty_cache()
            agent.run(input_message)

        past_key_values.reset()
        if draft_past_key_values is not None:
            draft_past_key_values.reset()
    generator.profiling = is_profiling
    
    # Evaluate dataset
    log_file = os.path.join(log_dir, f"0.jsonl")
    tput_list, acc_rate_list, draft_time_list, target_time_list = [], [], [], []
    for idx, query in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating", leave=True):
        tokenizer.use_default_system_prompt = True
        
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_message = agent.run(query)
            
        past_key_values.reset()
        if draft_past_key_values is not None:
            draft_past_key_values.reset()

        exp_log = {**generator.exp_log, "query": query, "response": output_message, "peak_mem": torch.cuda.max_memory_reserved(args.device)/(1024**3)}
        with open(log_file, 'a+') as f:
            json.dump(exp_log, f, indent=4)
            f.write("\n")

        if exp_log.get("tput", None) is not None:
            tput_list.append(exp_log.get("tput", 0))
        if exp_log.get("avg_sampled", None) is not None:
            acc_rate_list.append(exp_log.get("avg_sampled", 0))
        if exp_log.get("avg_draft_time", None) is not None:
            draft_time_list.append(exp_log.get("avg_draft_time", 0))
        if exp_log.get("avg_target_time", None) is not None:
            target_time_list.append(exp_log.get("avg_target_time", 0))
            
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"Final Results:")
    tput_mean, tput_std = np.mean(tput_list), np.std(tput_list)
    acc_rate_mean, acc_rate_std = np.mean(acc_rate_list), np.std(acc_rate_list) if acc_rate_list else 0
    avg_draft_time, avg_target_time = np.mean(draft_time_list), np.mean(target_time_list)
    peak_mem = torch.cuda.max_memory_reserved(args.device)/(1024**3)
    
    print(f"\tThroughput: {tput_mean:.3f} ± {tput_std:.3f} tokens/sec")
    print(f"\tAcceptance rate: {acc_rate_mean:.3f} ± {acc_rate_std:.3f} tokens/iter")
    print(f"\tAverage Draft Time: {avg_draft_time:.3f} sec")
    print(f"\tAverage Target Time: {avg_target_time:.3f} sec")
    print(f"\tPeak Memory: {peak_mem:.3f} GiB")
    
    return tput_mean, tput_std, acc_rate_mean, acc_rate_std, avg_draft_time, avg_target_time, peak_mem