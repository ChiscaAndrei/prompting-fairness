from statistics import mean
from typing import Any, Callable

import spacy
import typer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline

from src.abs_quant.template import generate_templates

sentiments = ['positive', 'neutral', 'negative']


def pretty_print_sorted_items(sorted_tuples: list[tuple[str, dict]], highlight: str) -> str:
    return "\n".join([f"{t[0]}: {t[1][highlight]}" for t in sorted_tuples])


def pretty_print(prediction: dict) -> str:
    return f"{prediction['token_str']} ({prediction['score']:.2f})"


def get_polarity_simple(prediction: dict, nlp) -> dict:
    # Todo - Weight the sentiment by probability? I.e. polarity = sum(polarity *
    docs = [nlp(p['token_str']) for p in prediction]
    polarities = [dict(doc._.polarity) for doc in docs]  # type: ignore
    average_polarity = {sentiment: mean([polarity[sentiment] for polarity in polarities]) for sentiment in sentiments}
    return average_polarity


def get_polarity_weighted(prediction: dict, nlp) -> dict:
    docs = [nlp(p['token_str']) for p in prediction]

    polarities = [dict(doc._.polarity) for doc in docs]  # type: ignore
    scores = [p['score'] for p in prediction]

    weighted_polarity = {}
    for sentiment in sentiments:
        sentiment_score = sum([polarities[idx][sentiment] * scores[idx] for idx in range(len(prediction))])
        weighted_polarity[sentiment] = sentiment_score

    return weighted_polarity


polarity_strategies = {
    'simple': get_polarity_simple,
    'weighted': get_polarity_weighted
}


def get_polarity_callable(strategy: str) -> Callable[[str, Any], dict]:
    return polarity_strategies[strategy]


def main(model_name: str = "bert-base-uncased",
         minimal: bool = False,
         singular: bool = False,
         strategy: str = "weighted"):
    get_polarity = get_polarity_callable(strategy)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('asent_en_v1')

    pipe = pipeline("fill-mask", tokenizer=tokenizer, model=model, device=0)

    templates = generate_templates(minimal=minimal, singular=singular)

    predictions = pipe(templates)

    polarities = {template: get_polarity(prediction, nlp) for (template, prediction) in zip(templates, predictions)}

    top_negative = sorted(polarities.items(), key=lambda x: x[1]['negative'], reverse=True)

    top_positive = sorted(polarities.items(), key=lambda x: x[1]['positive'], reverse=True)

    print("Positive")
    print(pretty_print_sorted_items(top_positive[:10], "positive"))

    print("Negative")
    print(pretty_print_sorted_items(top_negative[:10], "negative"))


if __name__ == '__main__':
    typer.run(main)
