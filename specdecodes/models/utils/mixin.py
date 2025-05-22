import torch
import json
import logging
import os
import time
import prettytable as pt
import csv

class ProfilingMixin:
    def __init__(self, *args, profiling: bool = True, profiling_verbose: bool = False, **kwargs):
        self.profiling = profiling
        self.profiling_verbose = profiling_verbose
        self.exp_log = {}
        super().__init__(*args, **kwargs)

    def _generate(self, input_ids: torch.LongTensor, *args, **kwargs):
        if not self.profiling:
            # If profiling is disabled, behave exactly like the original generator.
            return super()._generate(input_ids, *args, **kwargs)

        # Record the original token count (assumes batch_size=1)
        org_input_len = input_ids.shape[1]

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output_ids = super()._generate(input_ids, *args, **kwargs)
        end_event.record()

        # Ensure all CUDA ops are finished before measuring the elapsed time
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = elapsed_time_ms / 1000.0

        n_generated_tokens = output_ids.shape[1] - org_input_len
        throughput = n_generated_tokens / elapsed_time_s if elapsed_time_s > 0 else 0

        self.exp_log['n_tokens'] = n_generated_tokens
        self.exp_log['elapsed_time'] = elapsed_time_s
        self.exp_log['tput'] = throughput

        if self.profiling_verbose:
            logging.info(
                f"Generated {n_generated_tokens} tokens in {elapsed_time_s:.2f}s, "
                f"throughput: {throughput:.2f} tokens/s"
            )

        return output_ids

class SDProfilingMixin:
    def __init__(self, *args, profiling: bool = True, profiling_verbose: bool = False, out_dir=None, prefix="sd", **kwargs):
        self.out_dir = out_dir
        self.prefix = prefix
        self.profiling_verbose = profiling_verbose
        
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.draft_events = []
        self.target_events = []
        self.verify_events = []
        
        self.profiling = profiling
        self.exp_log = {}
        super().__init__(*args, **kwargs)
        
    def _speculate(self, *model_args, **kwargs):
        if not self.profiling:
            return super()._speculate(*model_args, **kwargs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        root = super()._speculate(*model_args, **kwargs)
        end_event.record()
        
        self.draft_events.append((start_event, end_event))
        return root
    
    def _tree_decoding(self, *model_args, **kwargs):
        if not self.profiling:
            return super()._tree_decoding(*model_args, **kwargs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        outputs = super()._tree_decoding(*model_args, **kwargs)
        end_event.record()
        
        self.target_events.append((start_event, end_event))
        return outputs
    
    def _verify(self, tree, *model_args, **kwargs):
        if not self.profiling:
            return super()._verify(tree, *model_args, **kwargs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        sampled_tokens, hidden_indices, (total_len, accept_len) = super()._verify(tree, *model_args, **kwargs)
        end_event.record()
        
        self.verify_events.append((start_event, end_event))
        
        # tokenize id to text for visualization
        # nodes = list(preorder_iter(root))
        # for node in nodes:
        #     node.id = self.tokenizer.decode(torch.tensor([node.id]), clean_up_tokenization_spaces=False)
        
        # profile data
        # json_graph = tree_to_nested_dict(root, name_key="name", attr_dict={"id": "id", "prob": "prob", "global_prob": "global_prob"})
        # sampled_tokens_list = sampled_tokens.squeeze(0).tolist()
        # self.profile_data[self.iter_count] = {}
        # self.profile_data[self.iter_count]["draft_tree"] = json_graph
        # self.profile_data[self.iter_count]["sampled_tokens"] = sampled_tokens_list
        
        # create profile data if not exist
        self.profile_data['iter'] = self.profile_data.get('iter', [])
        self.profile_data['total_len'] = self.profile_data.get('total_len', [])
        self.profile_data['accept_len'] = self.profile_data.get('accept_len', [])
            
        sampled_tokens_list = sampled_tokens.squeeze(0).tolist()
        self.profile_data['iter'].append(sampled_tokens_list)
        self.profile_data['total_len'].append(total_len)
        self.profile_data['accept_len'].append(accept_len)
        # logging
        logging.debug(
            f"Total: {tree.size()},"\
            f"\tPredicted ({accept_len}/{total_len}): {self.tokenizer.batch_decode(sampled_tokens.squeeze(0), clean_up_tokenization_spaces=False)}"
        )
        
        # update stats
        self.sampled_count += len(sampled_tokens[0])
        self.iter_count += 1
        
        return sampled_tokens, hidden_indices, (total_len, accept_len)
    
    def compute_average_times(self):
        """
        Synchronize once at the end, then compute average
        draft and target times from the recorded CUDA events.
        """
        # Ensure all CUDA kernels are done
        torch.cuda.synchronize()

        # Compute total time for draft iterations
        draft_time_total_ms = 0.0
        for (start_event, end_event) in self.draft_events:
            draft_time_total_ms += start_event.elapsed_time(end_event)  # returns time in ms

        # Compute total time for target iterations
        target_time_total_ms = 0.0
        for (start_event, end_event) in self.target_events:
            target_time_total_ms += start_event.elapsed_time(end_event)
            
        # Compute total time for verify iterations
        verify_time_total_ms = 0.0
        for (start_event, end_event) in self.verify_events:
            verify_time_total_ms += start_event.elapsed_time(end_event)

        # Average times (in milliseconds)
        draft_avg_ms = draft_time_total_ms / max(len(self.draft_events), 1)
        target_avg_ms = target_time_total_ms / max(len(self.target_events), 1)
        verify_avg_ms = verify_time_total_ms / max(len(self.verify_events), 1)

        # Convert to seconds if you prefer
        draft_avg_s = draft_avg_ms / 1000.0
        target_avg_s = target_avg_ms / 1000.0
        verify_avg_s = verify_avg_ms / 1000.0

        return draft_avg_s, target_avg_s, verify_avg_s
    
    def _generate(self, input_ids: torch.LongTensor, *model_args, **kwargs):
        if not self.profiling:
            return super()._generate(input_ids, *model_args, **kwargs)
        
        self.profile_data = {}
        self.sampled_count = 1 # assume first token is sampled (prefill stage)
        self.iter_count = 1 # assume first step is done (prefill stage)
        
        self.exp_log = {}
        self.draft_events = []
        self.target_events = []
        self.verify_events = []
        
        cur_time = time.strftime("%Y%m%d-%H%M%S")
        # prepare output directory
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, f"{self.prefix}_{cur_time}.json")
        else:
            out_path = None
        
        # run generation
        org_input_len = len(input_ids[0])
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        input_ids = super()._generate(input_ids, *model_args, **kwargs)
        end_event.record()
        
        # Make sure all CUDA ops have finished before measuring
        torch.cuda.synchronize()
        
        # Elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = elapsed_time_ms / 1000.0
        
        # compute stats
        total_sampled = self.sampled_count
        total_iterations = self.iter_count
        avg_sampled = total_sampled / total_iterations
        depth = max(self.profile_data['total_len']) + 1
        
        # alpha (node)
        total_lens = torch.bincount( torch.tensor(self.profile_data['total_len']), minlength=depth)
        accept_lens = torch.bincount( torch.tensor(self.profile_data['accept_len']), minlength=depth)
        depth_total_cnt = total_lens + total_lens.sum() - total_lens.cumsum(dim=-1) # reverse cumsum
        depth_total_cnt = depth_total_cnt[1:] # remove first element
        depth_accept_cnt = accept_lens + accept_lens.sum() - accept_lens.cumsum(dim=-1) # reverse cumsum
        depth_accept_cnt = depth_accept_cnt[1:] # remove first element
        alpha_per_node = depth_accept_cnt.float() / depth_total_cnt.float()
        
        # aLive ratio
        depth_alive_rate = depth_total_cnt.float() / depth_total_cnt[0]
        
        # alpha (depth)
        sampled_lens = torch.tensor([len(sampled_tokens) for sampled_tokens in self.profile_data["iter"]])
        sampled_len_bins = torch.bincount(sampled_lens, minlength=depth+1)
        depth_total_cnt = sampled_len_bins + sampled_len_bins.sum() - sampled_len_bins.cumsum(dim=-1) # reverse cumsum
        depth_accept_cnt = depth_total_cnt - sampled_len_bins
        depth_total_cnt = depth_total_cnt[1:depth]
        depth_accept_cnt = depth_accept_cnt[1:depth]
        alpha_per_depth = depth_accept_cnt.float() / depth_total_cnt.float()
        
        # log stats
        if self.profiling_verbose:
            tb = pt.PrettyTable()
            tb.field_names = [ "Summary \ Depth" ] + [ f"{i}" for i in range(1, depth) ]
            tb.add_row([ "Trials count" ] + [ f"{val}" for val in depth_total_cnt.tolist() ])
            tb.add_row([ "Accept count" ] + [ f"{val}" for val in depth_accept_cnt.tolist() ])
            tb.add_row([ "Alpha (node)" ] + [ f"{val:.2f}" for val in alpha_per_node.tolist() ])
            tb.add_row([ "Alpha (depth)" ] + [ f"{val:.2f}" for val in alpha_per_depth.tolist() ])
            tb.add_row([ "Alive ratio" ] + [ f"{val:.2f}" for val in depth_alive_rate.tolist() ])
            logging.info(
                f"Total sampled: {total_sampled},"\
                f"\tTotal iterations: {total_iterations},"\
                f"\tAverage sampled: {avg_sampled:.2f}"\
                f"\n{tb}"
            )
        
        # save profile data
        self.profile_data["total_sampled"] = total_sampled
        self.profile_data["total_iterations"] = total_iterations
        self.profile_data["average_sampled"] = avg_sampled
        if self.out_dir is not None:
            with open(out_path, "w") as f:
                json.dump(self.profile_data, f)
                
        # save exp_log
        avg_draft_s, avg_target_s, avg_verify_s = self.compute_average_times()
        self.exp_log['avg_draft_time'] = avg_draft_s
        self.exp_log['avg_target_time'] = avg_target_s
        self.exp_log['avg_verify_time'] = avg_verify_s
        
        self.exp_log['avg_sampled'] = avg_sampled
        self.exp_log['n_iter'] = total_iterations
        self.exp_log['n_tokens'] = len(input_ids[0][org_input_len:])
        self.exp_log['elapsed_time'] = elapsed_time_s
        self.exp_log['tput'] = len(input_ids[0][org_input_len:]) / elapsed_time_s
        if self.profiling_verbose:
            logging.info(
                f"Average draft time: {self.exp_log['avg_draft_time']:.4f},"\
                f"\tAverage target time: {self.exp_log['avg_target_time']:.4f},"\
                f"\tAverage verify time: {self.exp_log['avg_verify_time']:.4f}"
                f"\nGenerated {self.exp_log['n_tokens']} tokens in {elapsed_time_s:.2f}s, throughput: {self.exp_log['tput']:.2f} tokens/s"
            )
        return input_ids