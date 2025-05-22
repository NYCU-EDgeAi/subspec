# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """
{Instruction}
{Input}
""".strip()

# ALPACA
def load_alpaca_dataset():
    dataset = load_dataset("tatsu-lab/alpaca")
    formatted_dataset = [QUERY_TEMPLATE.format(Instruction=entry['instruction'], Input=entry['input']) for entry in dataset['train']]
    return formatted_dataset