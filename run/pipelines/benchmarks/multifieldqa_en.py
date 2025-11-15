# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """Read the following text and answer briefly.

{context}

Now, answer the following question based on the above text, only give me the answer and do not output any other words.

Question: {input}
Answer:"""

# multifieldqa_en
def load_multifieldqa_en_dataset():
    dataset = load_dataset("THUDM/LongBench", "multifieldqa_en", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context'], input=entry['input']) for entry in dataset]

    return formatted_dataset

def load_multifieldqa_en_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "multifieldqa_en", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'], input=entry['input'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str
        })

    return examples