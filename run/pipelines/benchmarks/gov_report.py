# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """You are given a report by a government agency. Write a one-page summary of the report.

Report:
{context}

Now, write a one-page summary of the report.

Summary:"""

# gov_report
def load_gov_report_dataset():
    dataset = load_dataset("THUDM/LongBench", "gov_report", split='test')
    
    formatted_dataset = [QUERY_TEMPLATE.format(context=entry['context']) for entry in dataset]

    return formatted_dataset

def load_gov_report_dataset_answer():
    dataset = load_dataset("THUDM/LongBench", "gov_report", split='test')
    examples = []

    for entry in dataset:
        q_str = QUERY_TEMPLATE.format(context=entry['context'])
        a_str = entry['answers']
        examples.append({
            "question": q_str,
            "answer": a_str
        })

    return examples