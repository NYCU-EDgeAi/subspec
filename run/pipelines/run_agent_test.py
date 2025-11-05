import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import trange
import logging
import warnings
import os
import nvtx

from smolagents import CodeAgent, ToolCallingAgent, TransformersModel, InferenceClientModel, Model
from specdecodes.helpers.wrappers import SpecDecodesModel

logger = logging.getLogger(__name__)
    

def main(builder):
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    args = builder.args
    
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    # deterministic
    torch.manual_seed(args.seed)
    
    # Build agent
    model = SpecDecodesModel(generator=generator, tokenizer=tokenizer, past_key_values=past_kv, draft_past_key_values=draft_past_kv, max_length=args.max_length, temperature=args.temperature, do_sample=args.do_sample, device=args.device)
    agent = ToolCallingAgent(tools=[], model=model, add_base_tools=True)

    # warm up
    if args.warmup_iter > 0:
        print("Warming up... It will take some time for the first few iterations to run.")
        with nvtx.annotate("Warming up"):
            is_profiling = generator.profiling
            generator.profiling = False
            for i in trange(args.warmup_iter, desc='Warming up'):
                input_message = "Write an essay about large language models."
                messages = [{"role": "user", "content": input_message}]
                tokenizer.use_default_system_prompt = True
                with nvtx.annotate("Warm up"):
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        output = agent.run(input_message)
                
                past_kv.reset()
                if draft_past_kv is not None:
                    draft_past_kv.reset()
            generator.profiling = is_profiling

    # input message
    input_message = "Do you know what is Beyblade? What is the best strategy to build the strongest Beyblade?"
    # input_message = "Describe what is large language models to me."
    messages = [{"role": "user", "content": input_message}]
    tokenizer.use_default_system_prompt = True
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(args.device)
    prompt = tokenizer.decode(input_ids[0])
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
                  
    # generate response
    print("Generating response...")
    torch.cuda.cudart().cudaProfilerStart() # start profiling from here
    start_event.record()
    with nvtx.annotate("Generate"):
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output = agent.run(input_message)
    end_event.record()
    
    # Ensure all CUDA kernels are done.
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    
    total_time_s = start_event.elapsed_time(end_event) / 1000.0

    if args.print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
    
    if args.print_time:
        print("Time:", total_time_s)