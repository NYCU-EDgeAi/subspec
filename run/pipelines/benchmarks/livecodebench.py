# LIVECODEBENCH_QUERY_TEMPLATE is adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
from datasets import load_dataset

QUERY_TEMPLATE = """
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {Question}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.

```python\n# YOUR CODE HERE\n```
""".strip()

# LIVECODEBENCH
def load_livecodebench_dataset():

    dataset = load_dataset("livecodebench/code_generation_lite", "v4_v5", trust_remote_code=True) # problems released between Aug 2024 and Jan 2025. The deepseek eval dataset setting
    formatted_dataset = [QUERY_TEMPLATE.format(Question=entry['question_content']) for entry in dataset['test']]

    return formatted_dataset
