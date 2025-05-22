# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset
import random

QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# GPQA
def load_gpqa_dataset():
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    
    # Prepare formatted dataset
    formatted_dataset = []
    for entry in dataset:
        question = entry["Question"]
        correct_answer = entry["Correct Answer"]
        incorrect_answers = [
            entry["Incorrect Answer 1"], 
            entry["Incorrect Answer 2"], 
            entry["Incorrect Answer 3"]
        ]
        
        # Shuffle answer choices
        choices = [correct_answer] + incorrect_answers
        random.shuffle(choices)
        
        formatted_dataset.append(QUERY_TEMPLATE.format(
            Question=question,
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3]
        ))

    return formatted_dataset