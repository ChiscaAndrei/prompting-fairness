import openai
import typer
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def main(prompt: str,
         model: str = "text-davinci-003",
         temperature: float = 0.,
         max_tokens: int = 200) -> None:
    """
    A simple CLI to interact with GPT-3 via the OpenAI SDK.

    :param prompt: The prompt for the input.
    :param model: The model name (see OpenAI docs).
    :param temperature: Model temperature, between 0 and 1. Higher values make the answers less deterministic.
    :param max_tokens: Maximum number of output tokens (words).
    :return: None.
    """
    response = openai.Completion.create(model=model,
                                        prompt=prompt,
                                        temperature=temperature,
                                        max_tokens=max_tokens)
    response = response["choices"][0]["text"]
    print(response)


if __name__ == '__main__':
    typer.run(main)
