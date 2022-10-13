from ice.paper import Paper
from ice.paper import Paragraph
from ice.recipe import recipe
from exercises.qa import answer
from ice.utils import map_async
import numpy as np
from structlog.stdlib import get_logger
from transformers import GPT2TokenizerFast

log = get_logger()

TOKEN_LIMIT = 2048
TOKEN_QA_EXISTING = 250

def get_relevance_prompt(paragraph: Paragraph, question: str) -> str:
    return f"""
    The following paragraph was drawn from the {paragraph.section_type} section of a paper. Answer whether the paragraph could be used to answer the question {question}.

    {paragraph}

    Answer (either Yes or No):
    """

async def paragraph_relevance(paragraph: Paragraph, question: str) -> float:
    '''
    For a given paragraph, return a number between 0 and 1 indicating how relevant it is to the question.
    '''
    relevance_prompt = get_relevance_prompt(paragraph, question)
    choice_probs, _ = await recipe.agent().classify(prompt=relevance_prompt, choices=(" Yes", " No"))
    return choice_probs.get(" Yes", 0.0)

def get_comparison_prompt(a: Paragraph, b: Paragraph, question: str) -> str:

    return f"""
Which of paragraphs A and B better answers the question "{question}"?

Paragraph A: {a}

Paragraph B: {b}

Question: Which of paragraphs A and B better answers the question '{question}'? Answer with "Paragraph A" or "Paragraph B".

Answer: Paragraph""".strip()


async def compare_paragraph_relevance(a: Paragraph, b: Paragraph, question: str) -> int:
    '''
    Return 1 if a is more relevant than b, -1 if b is more relevant than a, and 0 otherwise
    '''

    comparison_prompt = get_comparison_prompt(a, b, question)    
    completion = await recipe.agent().complete(prompt=comparison_prompt,
                    stop=["\n"],
                    max_tokens=1)
    completion = completion.strip()
    if completion == "A":
        return 1
    elif completion == "B":
        return -1
    else:
        log.warning(f"Unrecognized choice: {completion}")
        return 0        

async def partition_array(array: list[Paragraph], pivot: any, question: str) -> tuple[list, list]:
    '''
    Partition the array into two arrays, one containing elements less than the pivot and one containing elements greater than the pivot.
    '''
    left_array = []
    right_array = []

    left_or_right = await map_async(array, lambda p: compare_paragraph_relevance(p, pivot, question))
    for i, p in enumerate(array):
        if left_or_right[i] == 1:
            left_array.append(p)
        else:
            right_array.append(p)

    return left_array, right_array

async def rank_relevant_paragraphs(paragraphs: list[Paragraph], question: str) -> list[Paragraph]:
    '''
    Sort the paragraphs in the paper by relevance to the question.
    '''

    pivot = paragraphs[len(paragraphs) // 2]
    parapraphs_without_pivot = [p for p in paragraphs if p != pivot]
    left_array, right_array = await partition_array(parapraphs_without_pivot, pivot, question)

    if len(left_array) > 0:
        left_array = await rank_relevant_paragraphs(left_array, question)
    
    if len(right_array) > 0:
        right_array = await rank_relevant_paragraphs(right_array, question)

    return left_array + [pivot] + right_array

def relevant_paragraphs_to_context(relevant_paragraphs: list[Paragraph]) -> str:
    '''
    Combine the relevant paragraphs into a context string.
    '''
    return "\n\n".join(map(str, relevant_paragraphs))

async def get_token_count(text: str, tokenizer: any) -> int:
    return len(tokenizer.encode(text))

async def fill_context_window(paragraphs: list[Paragraph], window_size: int = TOKEN_LIMIT - TOKEN_QA_EXISTING) -> list[Paragraph]:
    '''
    Subset the paragraphs to fit within the token limit.
    '''
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens_per_paragraph = await map_async(paragraphs, lambda p: get_token_count(str(p), tokenizer))
    cumulative_token_lengths = np.cumsum(tokens_per_paragraph)
    cutoff = np.argmax(cumulative_token_lengths > window_size)
    return paragraphs[:cutoff]

async def paper_qa(paper: Paper, question: str = "What was the study population?") -> str:
    relevant_paragraphs = await rank_relevant_paragraphs(paper.nonempty_paragraphs(), question)
    relevant_paragraphs_subset = await fill_context_window(relevant_paragraphs)
    context = relevant_paragraphs_to_context(relevant_paragraphs_subset)
    return await answer(context, question)

recipe.main(paper_qa)