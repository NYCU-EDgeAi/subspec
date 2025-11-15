# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """You are given several news passages. Write a one-page summary of all news. 

News:
{context}

Now, write a one-page summary of all the news.

Summary:"""

# multi_news
def load_multi_news_dataset():
    dataset = load_dataset("THUDM/LongBench", "multi_news", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context']) for entry in dataset]

    return formatted_dataset

def load_multi_news_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "multi_news", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str
        })

    return examples