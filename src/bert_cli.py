from typing import Any

import typer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline


def to_string(prediction: dict[str, Any]) -> str:
    """
    Convert a BERT prediction to a more human-readable string.

    :param prediction: A dictionary
    :return:
    """
    return f"{prediction['token_str']} ({prediction['score']:.3f})"


def pretty_print(prediction: list[dict[str, float | str | int]]) -> str:
    return ", ".join([to_string(p) for p in prediction])


def main(model_name: str = "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    pipe = pipeline("fill-mask", tokenizer=tokenizer, model=model, device=0)

    while True:
        template = input()

        prediction = pipe(template)

        print(pretty_print(prediction))


if __name__ == '__main__':
    typer.run(main)
