# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """Please complete the code given below. 
{context}{input}Next line of code:
"""

# repobench_p
def load_repobench_p_dataset():
    dataset = load_dataset("THUDM/LongBench", "repobench-p", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context'], input=entry['input']) for entry in dataset]

    return formatted_dataset

def load_repobench_p_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "repobench-p", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'], input=entry['input'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str,
        })

    return examples