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

from src.prompt_tune_sandbox.bias_template import prepare_dataset_for_masked_model
from src.prompt_tune_sandbox.bias_trainer import BiasTrainerMaskedModel
from src.prompt_tune_sandbox.bias_evaluator import BiasEvaluatorForBert

app = typer.Typer()

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

@app.command()
def train_model(
        model_name='bert-large-cased', 
        prompt_length: int = 3,
        experiment_name: str = "test",
        eval_interval: int = 100,
        num_epochs: int = 1000,
        ):

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=f"./runs/{experiment_name}")

    typer.echo(f"Loading model under bias investigation {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Evaluate the model before any changes
    bias_evaluator = BiasEvaluatorForBert()
    model.eval()
    evaluation_initial_results = bias_evaluator.evaluate(model, tokenizer)
    add_results_to_tensorboard(writer, evaluation_initial_results, 0)
    dump_predictions_on_training_dataset(model, tokenizer, 10, 0, out_file_sents=f"./runs/{experiment_name}/predictions_before.html")

    # Convert the model to use prompt tuning
    typer.echo("Converting the model for prompt tuning...")
    peft_config = peft.PromptTuningConfig(
        task_type="SEQ_CLS", num_virtual_tokens=prompt_length)
    model = peft.get_peft_model(model, peft_config)
    model = model.to(device)
    model.train()
    model.print_trainable_parameters()

    # Load training dataset
    occupation_dataset = prepare_dataset_for_masked_model(tokenizer)
    occupation_dataset.set_format("torch")

    NUM_EPOCHS = num_epochs
    train_loader = DataLoader(occupation_dataset, batch_size=16, shuffle=True)
    optim = AdamW(model.parameters(), lr=1e-2)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=0.06 * (len(train_loader) * NUM_EPOCHS),
        num_training_steps=(len(train_loader) * NUM_EPOCHS),
    )

    for epoch in range(1,NUM_EPOCHS+1):
        losses_in_batch = []
        for batch in train_loader:
            optim.zero_grad()
            input_ids_male = batch["input_ids_male"].to(device)
            input_ids_female = batch["input_ids_female"].to(device)
            attention_mask_male = batch["attention_mask_male"].to(device)
            attention_mask_female = batch["attention_mask_female"].to(device)
            output_indices = batch["output_indices"].to(device)
            mask_token_index_male = batch["mask_token_idx_male"].to(device)
            mask_token_index_female = batch["mask_token_idx_female"].to(device)

            output_male = model(input_ids_male, attention_mask=attention_mask_male)
            output_female = model(input_ids_female, attention_mask=attention_mask_female)
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

            male_mask_valid_options_logits = torch.gather(male_mask_logits, 1, output_indices)
            female_mask_valid_options_logits = torch.gather(female_mask_logits, 1, output_indices)
            # Shape is [batch_size, number_valid_options]

            loss_batch = torch.linalg.vector_norm(male_mask_valid_options_logits - female_mask_valid_options_logits, dim=1)
            # Shape is [batch_size]

            loss = torch.mean(loss_batch)
            losses_in_batch.append(loss.item())

            loss.backward()

            optim.step()
            lr_scheduler.step()
        writer.add_scalar("Loss/train", np.mean(losses_in_batch), epoch)

        if epoch % 10 == 0:
            typer.echo(f"Loss in epoch {epoch}: {np.mean(losses_in_batch)}")
        if epoch % eval_interval == 0:
            model.eval()
            evaluation = bias_evaluator.evaluate(model, tokenizer, prompt_length=prompt_length)
            dump_predictions_on_training_dataset(model, tokenizer, 10, prompt_length, out_file_sents=f"./runs/{experiment_name}/predictions_after_{epoch}.html")
            model.train()
            #for test_category, results_for_category in evaluation.items():
            add_results_to_tensorboard(writer, evaluation, epoch)

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





if __name__ == '__main__':
    app()
    #typer.run(experiment)
