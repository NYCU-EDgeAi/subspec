import json

QUERY_TEMPLATE = """
{Prompt}
""".strip()

# MTBENCH
def load_mtbench_dataset():
    with open("run/pipelines/benchmarks/data/mt_bench.jsonl") as f:
        dataset = [json.loads(line)['turns'] for line in f]  # list of list of prompts
    return dataset