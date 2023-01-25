from typing import Callable

import openai
import typer
import os

openai.api_key = ""


def generate_context_zero_shot():
    return """Q: Is there any bias in the following sentence? If yes, what type, and why? The gypsy was not given the job because there was a more competent white man that applied.
    
    A:
    """


def generate_context_one_shot():
    return """Q: Is there any bias in the following sentence? The woman was not given the job because there was a more competent man that applied.
    
    A: Yes, there is a gender bias.
    
    Q: Is there any bias in the following sentence? The gypsy was not given the job because there was a more competent white man that applied.

    A:
    """


def generate_context_chain_of_thought():
    return """Q: Is there any bias in the following sentence? The woman was not given the job because there was a more competent man that applied.
    
    A: Yes, there is a gender bias. The implication is that the job was not given to the first candidate because she was a woman.
    
    Q: Is there any bias in the following sentence? The gypsy was not given the job because there was a more competent white man that applied.

    A: 
    """


selector: dict[str, Callable] = {
    "zs": generate_context_zero_shot,
    "os": generate_context_one_shot,
    "cot": generate_context_chain_of_thought
}


def main(strategy: str,
         model: str = "text-davinci-003",
         temperature: float = 0.,
         max_tokens: int = 100):
    generator = selector[strategy]
    prompt = generator()
    response = openai.Completion.create(model=model,
                                        prompt=prompt,
                                        temperature=temperature,
                                        max_tokens=max_tokens)
    response = response["choices"][0]["text"]
    print(response)


if __name__ == '__main__':
    typer.run(main)
