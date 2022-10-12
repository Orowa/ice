from ice.agents.base import Agent
from ice.recipe import recipe
from ice.recipes.primer.debate.prompt import *

def judge_debate_prompt(debate_history: Debate) -> Name:
    rendered_debate = render_debate(debate_history)
    prompt = f"""
    You are judging the debate. You are trying to decide who won the debate using reason and evidence. Return the name of the victor, and explain why they won.

    *Debate*
    {rendered_debate}

    *Judgment*
    The name of the victor is: "
    """.strip()
    return prompt

async def debate(question: str = "Should we ban guns?", number_of_turns: int = 8, agent_names: tuple[Name, Name] = ("Alice", "Bob")):
    agents = [recipe.agent(agent_name="instruct-stochastic"), recipe.agent(agent_name="instruct-stochastic")]
    debate_history = initialize_debate(question, agent_names)
    turns_left = number_of_turns
    while turns_left > 0:
        for agent, name in zip(agents, agent_names):
            prompt = render_debate_prompt(name, debate_history, turns_left)
            answer = await agent.answer(prompt=prompt, multiline=False)
            debate_history.append((name, answer.strip('" ')))
            turns_left -= 1
    judgment_prompt = judge_debate_prompt(debate_history)
    victor = await recipe.agent().answer(prompt=judgment_prompt)
    return judgment_prompt + victor

recipe.main(debate)