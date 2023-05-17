from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import pandas as pd
import peft
from peft import TaskType
import typer
import json
import itertools
from enum import Enum
from rich.console import Console
from rich.progress import track as rich_track

from src.prompt_tune_sandbox.bias_template import prepare_dataset_for_masked_model, create_positional_ids,PositionIdAdjustmentType
from src.prompt_tune_sandbox.bias_trainer import BiasTrainerMaskedModel
from src.prompt_tune_sandbox.bias_evaluator import BiasEvaluatorForBert

app = typer.Typer()
console = Console()

def add_results_to_tensorboard(writer: SummaryWriter, results, epoch, prefix=""):
    if type(results) is dict:
        for k, v in results.items():
            new_prefix = f"{prefix}/{k}" if prefix else k
            add_results_to_tensorboard(writer, v, epoch, new_prefix)
    elif type(results) is tuple:
        data_type, data = results
        if data_type == "embedding":
            writer.add_embedding(**data, global_step=epoch, tag=prefix)
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    else:
        # Assume scalar
        writer.add_scalar(prefix, results, epoch)


class LossFunctionType(str, Enum):
    equal_valid_options_mask_logits = "equal_logits_options"
    mean_valid_options_original_probabilities = "original_model_probability_options"


def loss_equal_valid_options_mask_logits(male_mask_logits, female_mask_logits, output_indices):
    """
    Computes a loss function which minimizes the norm:
    ||male_logits_opt-female_logits_opt||
    where male_logits_opt are the logits for the male sentence corresponding o the indices in `output_indices`
    """
    male_mask_valid_options_logits = torch.gather(male_mask_logits, 1, output_indices)
    female_mask_valid_options_logits = torch.gather(female_mask_logits, 1, output_indices)
    # Shape is [batch_size, number_valid_options]

    loss_batch = torch.linalg.vector_norm(male_mask_valid_options_logits - female_mask_valid_options_logits, dim=1)
    # Shape is [batch_size]

    loss = torch.mean(loss_batch)
    return loss


def loss_mean_valid_options_original_probabilities_old(male_mask_logits, female_mask_logits, male_original_model_mask_logits, female_original_model_mask_logits, output_indices):
    male_original_probabilities = torch.nn.functional.softmax(male_original_model_mask_logits, dim=1)
    female_original_probabilities = torch.nn.functional.softmax(female_original_model_mask_logits, dim=1) 
    average_probabilities = 0.5 * (male_original_probabilities + female_original_probabilities)
    average_probabilities_valid_options = torch.gather(average_probabilities, 1, output_indices)

    male_mask_log_probabilities = torch.nn.functional.log_softmax(male_mask_logits, dim=1)
    male_mask_valid_options_log_prob = torch.gather(male_mask_log_probabilities, 1, output_indices)
    female_mask_log_probabilities = torch.nn.functional.log_softmax(female_mask_logits, dim=1)
    female_mask_valid_options_log_prob = torch.gather(female_mask_log_probabilities, 1, output_indices)

    loss_male = torch.nn.functional.kl_div(male_mask_valid_options_log_prob, average_probabilities_valid_options, reduction="batchmean", log_target=False)
    loss_female = torch.nn.functional.kl_div(female_mask_valid_options_log_prob, average_probabilities_valid_options, reduction="batchmean", log_target=False)
    loss = loss_male + loss_female
    return loss


def loss_mean_valid_options_original_probabilities(male_mask_logits, female_mask_logits, male_original_model_mask_logits, female_original_model_mask_logits, output_indices, specific_output_indices_male=None, specific_output_indices_female=None):
    male_original_log_probabilities = torch.nn.functional.log_softmax(male_original_model_mask_logits, dim=1)
    female_original_log_probabilities = torch.nn.functional.log_softmax(female_original_model_mask_logits, dim=1) 
    # We have so far the log probabilities (denote them m and f) and we wish to compute the log of the average probability
    # which is log((e^m+e^f)/2) = log(e^m+e^f) - log 2
    average_log_probabilities = torch.logaddexp(male_original_log_probabilities, female_original_log_probabilities) - torch.log(torch.tensor(2)) 
    average_log_probabilities_valid_options = torch.gather(average_log_probabilities, 1, output_indices)
    # Shape is [batch_size, valid_options]
    # After selecting the valid options, we no longer have a probability distribution since the sum of
    # elements is no longer equal to 1. To fix this, we add another "dummy element" which is equal
    # to 1 minus the sum of the other elements.
    def fix_probability_distribution(log_probs_valid_options):
        invalid_options_log_probs = torch.log(1-torch.exp(torch.logsumexp(log_probs_valid_options, dim=1, keepdim=True)))  # [batch_size, 1]
        return torch.cat((log_probs_valid_options, invalid_options_log_probs), dim=1)

    if specific_output_indices_male is not None and specific_output_indices_female is not None:
        output_indices_male = torch.cat((output_indices, specific_output_indices_male), dim=1)
        output_indices_female = torch.cat((output_indices, specific_output_indices_female), dim=1)
        log_probabilities_male_options = torch.gather(male_original_log_probabilities, 1, specific_output_indices_male)
        log_probabilities_female_options = torch.gather(female_original_log_probabilities, 1, specific_output_indices_female)
        # TODO: check if this really makes sense
        average_log_probabilities_gender_specific_options = torch.logaddexp(log_probabilities_male_options, log_probabilities_female_options) - torch.log(torch.tensor(2)) 
        average_log_probabilities_valid_options = torch.cat(
            (average_log_probabilities_valid_options, average_log_probabilities_gender_specific_options), dim=1)
    else:
        output_indices_male = output_indices
        output_indices_female = output_indices

    # Make sure this adjustment is done AFTER the gender specific probabilities are appended to the input
    average_log_probabilities_valid_options = fix_probability_distribution(average_log_probabilities_valid_options)
    
    male_mask_log_probabilities = torch.nn.functional.log_softmax(male_mask_logits, dim=1)
    male_mask_valid_options_log_prob = torch.gather(male_mask_log_probabilities, 1, output_indices_male)
    male_mask_valid_options_log_prob = fix_probability_distribution(male_mask_valid_options_log_prob)
    female_mask_log_probabilities = torch.nn.functional.log_softmax(female_mask_logits, dim=1)
    female_mask_valid_options_log_prob = torch.gather(female_mask_log_probabilities, 1, output_indices_female)
    female_mask_valid_options_log_prob = fix_probability_distribution(female_mask_valid_options_log_prob)

    loss_male = torch.nn.functional.kl_div(male_mask_valid_options_log_prob, average_log_probabilities_valid_options, reduction="batchmean", log_target=True)
    loss_female = torch.nn.functional.kl_div(female_mask_valid_options_log_prob, average_log_probabilities_valid_options, reduction="batchmean", log_target=True)
    loss = loss_male + loss_female
    return loss

@app.command()
def train_model(
        model_name='bert-large-cased', 
        prompt_length: int = 3,
        experiment_name: str = "test",
        eval_interval: int = 100,
        num_epochs: int = 1000,
        loss_type: LossFunctionType = LossFunctionType.equal_valid_options_mask_logits,
        position_ids_adjustment: PositionIdAdjustmentType = PositionIdAdjustmentType.none,
        gender_specific_options: bool = False,
        ):

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=f"./runs/{experiment_name}")
    console.log(f"Writing tensorboard logs to {writer.log_dir}")

    with console.status(f"Loading model under bias investigation {model_name}..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load training dataset
    with console.status(f"Loading training dataset..."):
        occupation_dataset = prepare_dataset_for_masked_model(tokenizer, False, model)
        occupation_dataset.set_format("torch")
        console.log("Training dataset loaded")

    # Evaluate the model before any changes
    with console.status(f"Evaluating original model..."):
        bias_evaluator = BiasEvaluatorForBert()
        model.eval()
        evaluation_initial_results = bias_evaluator.evaluate(model, tokenizer)
        add_results_to_tensorboard(writer, evaluation_initial_results, 0)
        dump_predictions_on_training_dataset(model, tokenizer, 10, 0, out_file_sents=f"./runs/{experiment_name}/predictions_before.html")
        console.log("Finished evaluation of initial model")

    # Convert the model to use prompt tuning
    with console.status("Converting the model for prompt tuning..."):
        peft_config = peft.PromptTuningConfig(
            task_type="SEQ_CLS", num_virtual_tokens=prompt_length)
        model = peft.get_peft_model(model, peft_config)
        model = model.to(device)
        console.log("Converted model for prompt tuning")
        model.train()
        model.print_trainable_parameters()

    NUM_EPOCHS = num_epochs
    dataset_split = occupation_dataset.train_test_split(test_size=0.1)
    
    train_loader = DataLoader(dataset_split["train"], batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset_split["test"], batch_size=16)
    optim = AdamW(model.parameters(), lr=1e-2)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=0.06 * (len(train_loader) * NUM_EPOCHS),
        num_training_steps=(len(train_loader) * NUM_EPOCHS),
    )

    def compute_loss(batch):
        input_ids_male = batch["input_ids_male"].to(device)
        input_ids_female = batch["input_ids_female"].to(device)
        attention_mask_male = batch["attention_mask_male"].to(device)
        attention_mask_female = batch["attention_mask_female"].to(device)
        output_indices = batch["output_indices"].to(device)
        mask_token_index_male = batch["mask_token_idx_male"].to(device)
        mask_token_index_female = batch["mask_token_idx_female"].to(device)
        male_original_model_mask_logits = batch["male_original_model_mask_logits"].to(device)
        female_original_model_mask_logits = batch["female_original_model_mask_logits"].to(device)

        if gender_specific_options:
            output_indices_male = batch["output_indices_male"].to(device)
            output_indices_female = batch["output_indices_female"].to(device)
        else:
            output_indices_male = None
            output_indices_female = None

        position_ids_male = create_positional_ids(input_ids_male.size(0), input_ids_male.size(1), prompt_length, device, position_ids_adjustment)
        position_ids_female = create_positional_ids(input_ids_female.size(0), input_ids_female.size(1), prompt_length, device, position_ids_adjustment)

        output_male = model(input_ids_male, attention_mask=attention_mask_male, position_ids=position_ids_male)
        output_female = model(input_ids_female, attention_mask=attention_mask_female, position_ids=position_ids_female)
        # size: torch.Size([batch, 27, 28996])
        # shape i: [batch_size, seq_len, vocab_size]
        # The model will also return the logits for the prompt tokens, so the output seq_len is larger than the input

        # Adjust the mask token index and output indices to take into account the size of the prompt
        mask_token_index_male = mask_token_index_male + prompt_length
        mask_token_index_female = mask_token_index_female + prompt_length

        male_logits = output_male.logits
        male_mask_logits = male_logits[torch.arange(male_logits.size(0)), mask_token_index_male,:]
        female_logits = output_female.logits
        female_mask_logits = female_logits[torch.arange(female_logits.size(0)), mask_token_index_female,:]
        # Shape is [batch_size, vocab_size]

        # Compute loss
        loss = None
        if loss_type == LossFunctionType.equal_valid_options_mask_logits:
            loss = loss_equal_valid_options_mask_logits(male_mask_logits, female_mask_logits, output_indices)
        elif loss_type == LossFunctionType.mean_valid_options_original_probabilities:
            loss = loss_mean_valid_options_original_probabilities(
                male_mask_logits, female_mask_logits,
                male_original_model_mask_logits, female_original_model_mask_logits,
                output_indices, output_indices_male, output_indices_female)

        if loss is None:
            raise RuntimeError("Loss was not computed")
        return loss


    console.rule("START OF TRAINING")

    for epoch in rich_track(range(1,NUM_EPOCHS+1), description="Training...", console=console):
        losses_in_batch = []
        test_losses_in_batch = []
        for batch in train_loader:
            optim.zero_grad()
            loss = compute_loss(batch)
            losses_in_batch.append(loss.item())
            loss.backward()

            optim.step()
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            for batch in test_loader:
                loss_test = compute_loss(batch)
                test_losses_in_batch.append(loss_test.item())
            model.train()
            

        writer.add_scalar("Loss/train", np.mean(losses_in_batch), epoch)
        writer.add_scalar("Loss/test", np.mean(test_losses_in_batch), epoch)

        if epoch % 10 == 0:
            console.log(f"Loss in epoch {epoch}: {np.mean(losses_in_batch)}")
        if epoch % eval_interval == 0:
            model.eval()
            evaluation = bias_evaluator.evaluate(
                model, tokenizer, prompt_length=prompt_length, return_embeddings=True, position_id_adjustment=position_ids_adjustment)
            dump_predictions_on_training_dataset(model, tokenizer, 10, prompt_length, out_file_sents=f"./runs/{experiment_name}/predictions_after_{epoch}.html")
            model.train()
            #for test_category, results_for_category in evaluation.items():
            add_results_to_tensorboard(writer, evaluation, epoch)
    console.rule("FINISHED TRAINING")

@app.command()
def inspect_dataset(
    model_name='bert-large-cased', 
    n_options: int = 10,
    out_file_sents="",
    out_file_predictions="",
    ):
    pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    dump_predictions_on_training_dataset(model, tokenizer, n_options, 0, out_file_sents, out_file_predictions)

def dump_predictions_on_training_dataset(
    model, 
    tokenizer,
    n_options: int = 10,
    prompt_length = 0,
    out_file_sents="",
    out_file_predictions="",
    ):
    occupation_dataset, male_sentences, female_sentences = prepare_dataset_for_masked_model(tokenizer, return_unencoded_sentences=True)
    occupation_dataset.set_format("torch")
    device = model.device
    input_ids_male = occupation_dataset["input_ids_male"].to(device)
    input_ids_female = occupation_dataset["input_ids_female"].to(device)
    attention_mask_male = occupation_dataset["attention_mask_male"].to(device)
    attention_mask_female = occupation_dataset["attention_mask_female"].to(device)
    output_indices = occupation_dataset["output_indices"].to(device)
    mask_token_index_male = occupation_dataset["mask_token_idx_male"].to(device)
    mask_token_index_female = occupation_dataset["mask_token_idx_female"].to(device)
    # Adjust the mask token index and output indices to take into account the size of the prompt
    mask_token_index_male = mask_token_index_male + prompt_length
    mask_token_index_female = mask_token_index_female + prompt_length

    with torch.no_grad():
        output_male = model(input_ids_male, attention_mask=attention_mask_male)
        output_female = model(input_ids_female, attention_mask=attention_mask_female)
        male_logits = output_male.logits
        male_mask_logits = male_logits[torch.arange(male_logits.size(0)), mask_token_index_male,:]
        female_logits = output_female.logits
        female_mask_logits = female_logits[torch.arange(female_logits.size(0)), mask_token_index_female,:]

        male_mask_logits = male_mask_logits.cpu().numpy()
        female_mask_logits = female_mask_logits.cpu().numpy()
        # Shape is [batch_size, vocab_size]
    top_indices_male = np.argsort(male_mask_logits, axis=1)[:,-n_options:]
    top_indices_female = np.argsort(female_mask_logits, axis=1)[:,-n_options:]
    # Shape is [batch_size, n_options]

    dataframe_records = []
    for male_sentence, female_sentence, male_indices, female_indices in zip(male_sentences, female_sentences, top_indices_male, top_indices_female):
        # reverse indices so that they are in reverse order
        male_indices = male_indices[::-1]
        female_indices = female_indices[::-1]
        male_predictions = tokenizer.convert_ids_to_tokens(male_indices)
        female_predictions = tokenizer.convert_ids_to_tokens(female_indices)
        dataframe_records.append(
            {
                "MaleSentence": male_sentence,
                "FemaleSentence": female_sentence,
                "MaleIndices": male_indices,
                "FemaleIndices": female_indices,
                "MalePredictions": male_predictions,
                "FemalePredictions": female_predictions 
            }
        )
    df = pd.DataFrame.from_records(dataframe_records)
    print(df)
    if out_file_sents:
        if out_file_sents.endswith(".html"):
            with open(out_file_sents,"w") as fout:
                df.to_html(fout)
        elif out_file_sents.endswith(".xlsx"):
            df.to_excel(out_file_sents)

    dataframe_top_predictions_records = []
    all_predictions = list(itertools.chain(*df["MalePredictions"].to_list())) + list(itertools.chain(*df["FemalePredictions"].to_list()))
    all_predictions.sort()
    for word, word_instances in itertools.groupby(all_predictions):
        count = len(list(word_instances))
        dataframe_top_predictions_records.append(
            {
                "Word": word,
                "Count": count,
            }
        )
    df_top_predictions = pd.DataFrame.from_records(dataframe_top_predictions_records)
    df_top_predictions.sort_values("Count", ascending=False, inplace=True)
    print(df_top_predictions)

    if out_file_predictions:
        if out_file_predictions.endswith(".html"):
            with open(out_file_predictions,"w") as fout:
                df_top_predictions.to_html(fout)
        elif out_file_predictions.endswith(".xlsx"):
            df_top_predictions.to_excel(out_file_predictions)


@app.command()
def experiment_position_ids(
    model_name='bert-large-cased'
):
    pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    bias_evaluator = BiasEvaluatorForBert()

    with console.status("Evaluating initial model"):
        results_initial = bias_evaluator.evaluate(model ,tokenizer, 0, False, PositionIdAdjustmentType.none)
    with console.status("Evaluating position ids shifter by 1"):
        results_start_1 = bias_evaluator.evaluate(model ,tokenizer, 0, False, PositionIdAdjustmentType.start_1)
    with console.status("Evaluating position ids shifter by 2"):
        results_start_2 = bias_evaluator.evaluate(model ,tokenizer, 0, False, PositionIdAdjustmentType.start_2)
    with console.status("Evaluating position ids shifter by 3"):
        results_start_3 = bias_evaluator.evaluate(model ,tokenizer, 0, False, PositionIdAdjustmentType.start_3)
    with console.status("Evaluating initial model a second time"):
        results_initial2 = bias_evaluator.evaluate(model ,tokenizer, 0, False, PositionIdAdjustmentType.none)

    console.print("\n[b]Initial results:[/b]", results_initial)
    console.print("\n[b]Start 1 results:[/b]", results_start_1)
    console.print("\n[b]Start 2 results:[/b]", results_start_2)
    console.print("\n[b]Start 3 results:[/b]", results_start_3)
    console.print("\n[b]Initial results (second try):[/b]", results_initial2)



if __name__ == '__main__':
    app()
