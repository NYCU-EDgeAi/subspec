# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?

{context}

Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.

The final answer is: """

# passage_count
def load_passage_count_dataset():
    dataset = load_dataset("THUDM/LongBench", "passage_count", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context']) for entry in dataset]

    return formatted_dataset

def load_passage_count_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "passage_count", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str,
        })

    return examples
