# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """
Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{Question}
""".strip()

# GSM8K
def load_gsm8k_dataset():
    dataset = load_dataset("openai/gsm8k", "main")
    formatted_dataset = [QUERY_TEMPLATE.format(Question=entry['question']) for entry in dataset['test']]
    return formatted_dataset