# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.

Transcript:
{context}

Now, answer the query based on the above meeting transcript in one or more sentences.

Query: {input}
Answer:"""

# qmsum
def load_qmsum_dataset():
    dataset = load_dataset("THUDM/LongBench", "qmsum", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context'], input=entry['input']) for entry in dataset]

    return formatted_dataset

def load_qmsum_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "qmsum", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'], input=entry['input'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str
        })

    return examples
