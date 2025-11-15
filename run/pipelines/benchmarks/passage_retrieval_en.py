# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.

{context}

The following is an abstract.

{input}

Please enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.

The answer is: """

# passage_retrieval_en
def load_passage_retrieval_en_dataset():
    dataset = load_dataset("THUDM/LongBench", "passage_retrieval_en", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context'], input=entry['input']) for entry in dataset]

    return formatted_dataset

def load_passage_retrieval_en_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "passage_retrieval_en", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'], input=entry['input'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str,
        })

    return examples