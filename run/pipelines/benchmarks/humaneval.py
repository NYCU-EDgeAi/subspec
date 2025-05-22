# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """
Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.

{Question}
""".strip()

# HUMANEVAL
def load_humaneval_dataset():
    dataset = load_dataset("openai/openai_humaneval")
    formatted_dataset = [QUERY_TEMPLATE.format(Question=entry['prompt']) for entry in dataset['test']]
    return formatted_dataset