# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.

{context}

{input}"""

# triviaqa
def load_triviaqa_dataset():
    dataset = load_dataset("THUDM/LongBench", "triviaqa", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context'], input=entry['input']) for entry in dataset]

    return formatted_dataset

def load_triviaqa_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "triviaqa", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'], input=entry['input'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str,
        })

    return examples