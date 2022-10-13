from ice.paper import Paper
from ice.paper import Paragraph
from ice.recipe import recipe
from exercises.qa import answer
from ice.utils import map_async
import numpy as np
from transformers import GPT2TokenizerFast

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

async def rank_relevant_paragraphs(paper: Paper, question: str) -> list[Paragraph]:
    '''
    Sort the paragraphs in the paper by relevance to the question.
    '''
    relevance_per_paragraph = await map_async(paper.paragraphs, lambda p: paragraph_relevance(p, question))
    sorted_pairs = sorted(
        zip(paper.paragraphs, relevance_per_paragraph), key=lambda x: x[1], reverse=True
    )
    return [p for p, _ in sorted_pairs]

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
    relevant_paragraphs = await rank_relevant_paragraphs(paper, question)
    relevant_paragraphs_subset = await fill_context_window(relevant_paragraphs)
    context = relevant_paragraphs_to_context(relevant_paragraphs_subset)
    return await answer(context, question)

recipe.main(paper_qa)