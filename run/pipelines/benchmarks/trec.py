# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """Please determine the type of the question below. Here are some examples of questions.

{context}
{input}"""

# trec
def load_trec_dataset():
    dataset = load_dataset("THUDM/LongBench", "trec", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context'], input=entry['input']) for entry in dataset]

    return formatted_dataset

def load_trec_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "trec", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'], input=entry['input'])
        a_str = entry['answers']
        c_str = entry['all_classes']
        examples.append({
            "question": q_str,
            "answer": a_str,
            "classes": c_str,
        })

    return examples