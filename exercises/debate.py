import itertools
from ice.agents.base import Agent
from ice.recipe import recipe
from ice.recipes.primer.debate.prompt import *
from ice.utils import map_async
import numpy as np

def judge_debate_prompt(debate_history: Debate) -> Name:
    rendered_debate = render_debate(debate_history)
    prompt = f"""
    You are judging the debate. You are trying to decide who won the debate using reason and evidence. You disapprove of repetition. Return the name of the victor, and explain why they won.

    *Debate*
    {rendered_debate}

    *Judgment*
    The name of the victor is:
    """.strip()
    return prompt


async def get_victory_prob(answer: str, debate_history: Debate, name: Name, other_name: Name) -> float:
    '''
    Return the probability that the answer will result in the agent winning the debate.
    '''
    debate_history_copy = debate_history.copy()
    debate_history_copy.append((name, answer))
    judgment_prompt = judge_debate_prompt(debate_history_copy)
    choice_probs, _ = await recipe.agent().classify(prompt=judgment_prompt, choices=(name, other_name))
    return choice_probs.get(name, 0.0)

async def debate(question: str = "Should we ban guns?", number_of_turns: int = 8, best_of_n: int = 5, agent_names: tuple[Name, Name] = ("Alice", "Bob")):
    agents = [recipe.agent(agent_name="instruct-stochastic"), recipe.agent(agent_name="instruct-stochastic")]
    debate_history = initialize_debate(question, agent_names)
    turns_left = number_of_turns
    cache_id = itertools.count()
    while turns_left > 0:
        for agent, name in zip(agents, agent_names):
            other_name = agent_names[1 - agent_names.index(name)]
            prompt = render_debate_prompt(name, debate_history, turns_left)
            potential_answers = await map_async(range(best_of_n), lambda x: agent.complete(prompt=prompt, stop=["\n"], cache_id=x))
            victory_prob_per_answer = await map_async(potential_answers, lambda a: get_victory_prob(a, debate_history, name, other_name))
            print(dict(zip(potential_answers, victory_prob_per_answer)))
            best_answer = potential_answers[np.argmax(victory_prob_per_answer)].strip('"')
            print(f"{name} says: {best_answer}")
            debate_history.append((name, best_answer))
            turns_left -= 1
    judgment_prompt = judge_debate_prompt(debate_history)
    victor = await recipe.agent().complete(prompt=judgment_prompt)
    return judgment_prompt + ' ' + victor

recipe.main(debate)