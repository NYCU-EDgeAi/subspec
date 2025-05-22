# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

# MATH-500
def load_math500_dataset():
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    formatted_dataset = [QUERY_TEMPLATE.format(Question=entry['problem']) for entry in dataset['test']]
    return formatted_dataset