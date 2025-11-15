# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.

Story: {context}

Now, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.

Question: {input}

Answer:"""

# narrativeqa
def load_narrativeqa_dataset():
    dataset = load_dataset("THUDM/LongBench", "narrativeqa", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context'], input=entry['input']) for entry in dataset]

    return formatted_dataset

def load_narrativeqa_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "narrativeqa", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'], input=entry['input'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str
        })

    return examples