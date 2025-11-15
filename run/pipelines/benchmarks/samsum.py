# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """Summarize the dialogue into a few short sentences. The following are some examples.

{context}

{input}"""

# samsum
def load_samsum_dataset():
    dataset = load_dataset("THUDM/LongBench", "samsum", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context'], input=entry['input']) for entry in dataset]

    return formatted_dataset

def load_samsum_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "samsum", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'], input=entry['input'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str,
        })

    return examples