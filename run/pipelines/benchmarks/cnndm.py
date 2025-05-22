# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """
Please summarize the following article:

{Articles}
""".strip()

# CNN/DailyMail 
def load_cnndm_dataset():
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    formatted_dataset = [QUERY_TEMPLATE.format(Articles=entry['article']) for entry in dataset['test']]
    
    return formatted_dataset