# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

def load_hotpotqa_dataset():
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    formatted_dataset = [entry['question'] for entry in dataset]
    return formatted_dataset