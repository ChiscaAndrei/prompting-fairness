import json
import os
import torch
import numpy as np
from src.bookcorpus.seat_utils import run_test
from src.prompt_tune_sandbox.bias_template import prepare_dataset_for_masked_model

class BiasEvaluatorForBert:
    
    def __init__(self) -> None:
        current_dir = os.path.dirname(__file__)
        self.seat_file_paths = {
            "sent-weat6"  : f"{current_dir}/gender_seat_templates/sent-weat6.jsonl",
            "sent-weat6b" : f"{current_dir}/gender_seat_templates/sent-weat6b.jsonl",
            "sent-weat7"  : f"{current_dir}/gender_seat_templates/sent-weat7.jsonl",
            "sent-weat7b" : f"{current_dir}/gender_seat_templates/sent-weat7b.jsonl",
            "sent-weat8"  : f"{current_dir}/gender_seat_templates/sent-weat8.jsonl",
            "sent-weat8b" : f"{current_dir}/gender_seat_templates/sent-weat8b.jsonl",
        }
        self.seat_templates = {}

        for name, file_path in self.seat_file_paths.items():
            with open(file_path, "r") as json_file:
                self.seat_templates[name] = json.load(json_file)


    def evaluate(self, model, tokenizer, prompt_length=0, return_embeddings=True):
        seat_results = {}
        seat_embeddings = {}
        for seat_test_name, seat_template in self.seat_templates.items():
            seat_results[seat_test_name], seat_embeddings[seat_test_name] = self.evaluate_seat(
                seat_template, model, tokenizer, prompt_length)
        
        occupation_dataset_statistics = self.evaluate_training_occupation_dataset(model, tokenizer, prompt_length)

        return_value = {
            "seat_results": seat_results,
            "occupation_dataset": occupation_dataset_statistics,
        }

        if return_embeddings:
            return_value["seat_embeddings"] = seat_embeddings

        return return_value


    def encode(self,
               text: str,
               model,
               tokenizer,
               prompt_length):
        tokens = tokenizer(text, return_tensors='pt').to(model.device)
        outputs = model(**tokens, output_hidden_states=True)

        # hidden_states [batch, seq_len, d_hidden]
        hidden_states = outputs.hidden_states[-1]

        return hidden_states[:, prompt_length, :].squeeze().detach().cpu().numpy()

    def evaluate_seat(self, seat_template, model, tokenizer, prompt_length):
        with torch.no_grad():
            targ1_embeddings = np.array(
                [self.encode(sample, model, tokenizer, prompt_length) for sample in seat_template["targ1"]["examples"]])
            targ2_embeddings = np.array(
                [self.encode(sample, model, tokenizer, prompt_length) for sample in seat_template["targ2"]["examples"]])
            attr1_embeddings = np.array(
                [self.encode(sample, model, tokenizer, prompt_length) for sample in seat_template["attr1"]["examples"]])
            attr2_embeddings = np.array(
                [self.encode(sample, model, tokenizer, prompt_length) for sample in seat_template["attr2"]["examples"]])

        labels = (targ1_embeddings.shape[0] * [seat_template["targ1"]["category"]+"(targ1)"]) +\
            (targ2_embeddings.shape[0] * [seat_template["targ2"]["category"]+"(targ2)"]) +\
            (attr1_embeddings.shape[0] * [seat_template["attr1"]["category"]+"(attr1)"]) +\
            (attr2_embeddings.shape[0] * [seat_template["attr2"]["category"]+"(attr2)"])
        embeddings = np.vstack((targ1_embeddings, targ2_embeddings, attr1_embeddings, attr2_embeddings))

        effect_size, p_value = run_test(x=targ1_embeddings, y=targ2_embeddings,
                                    a=attr1_embeddings, b=attr2_embeddings,
                                    n_samples=10000, parametric=False)
        p_value = p_value if effect_size > 0.0 else 1.0-p_value

        seat_result = {
            "effect_size": effect_size,
            "p_value" : p_value
        }
        embedding_result = {
            "mat": embeddings,
            "metadata": labels
        }
        return seat_result, ("embedding", embedding_result)

    @torch.no_grad()
    def evaluate_training_occupation_dataset(self, model, tokenizer, prompt_length):
        dataset = prepare_dataset_for_masked_model(tokenizer)
        dataset.set_format("torch")
        device = model.device

        input_ids_male = dataset["input_ids_male"].to(device)
        input_ids_female = dataset["input_ids_female"].to(device)
        attention_mask_male = dataset["attention_mask_male"].to(device)
        attention_mask_female = dataset["attention_mask_female"].to(device)
        output_indices = dataset["output_indices"].to(device)
        mask_token_index_male = dataset["mask_token_idx_male"].to(device)
        mask_token_index_female = dataset["mask_token_idx_female"].to(device)

        batch_size = len(dataset)

        output_male = model(input_ids_male, attention_mask=attention_mask_male)
        output_female = model(input_ids_female, attention_mask=attention_mask_female)
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
        assert male_mask_logits.size(0) == batch_size
        assert female_mask_logits.size(0) == batch_size

        male_mask_valid_options_logits = torch.gather(male_mask_logits, 1, output_indices)
        female_mask_valid_options_logits = torch.gather(female_mask_logits, 1, output_indices)
        # Shape is [batch_size, number_valid_options]
        assert male_mask_valid_options_logits.size(0) == batch_size
        assert female_mask_valid_options_logits.size(0) == batch_size
        assert male_mask_valid_options_logits.size(1) == output_indices.size(1)
        assert female_mask_valid_options_logits.size(1) == output_indices.size(1)

        male_prediction_index = torch.argmax(male_mask_logits, dim=1)
        female_prediction_index = torch.argmax(female_mask_logits, dim=1)
        male_prediction_valid_options_index = torch.argmax(male_mask_valid_options_logits, dim=1)
        female_prediction_valid_options_index = torch.argmax(female_mask_valid_options_logits, dim=1)
        assert male_prediction_index.size(0) == batch_size
        assert female_prediction_index.size(0) == batch_size
        assert male_prediction_valid_options_index.size(0) == batch_size
        assert female_prediction_valid_options_index.size(0) == batch_size

        number_of_same_predictions = torch.sum(male_prediction_index == female_prediction_index).item()
        number_of_same_predictions_valid_options = torch.sum(male_prediction_valid_options_index == female_prediction_valid_options_index).item()
        number_predictions_in_valid_options_male = sum([i in output_indices for i in male_prediction_index])
        number_predictions_in_valid_options_female = sum([i in output_indices for i in female_prediction_index])
        return {
            "ratio_different_predictions": float(batch_size-number_of_same_predictions)/batch_size,
            "ratio_different_predictions_valid_options": float(batch_size-number_of_same_predictions_valid_options)/batch_size,
            "ratio_predictions_in_valid_options_male": float(number_predictions_in_valid_options_male)/batch_size,
            "ratio_predictions_in_valid_options_female": float(number_predictions_in_valid_options_female)/batch_size,
        }
        
