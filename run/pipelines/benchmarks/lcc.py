# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """Please complete the code given below. \n{context}Next line of code:\n"""

# lcc
def load_lcc_dataset():
    dataset = load_dataset("THUDM/LongBench", "lcc", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context']) for entry in dataset]

    return formatted_dataset

def load_lcc_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "lcc", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str,
        })

    return examples
