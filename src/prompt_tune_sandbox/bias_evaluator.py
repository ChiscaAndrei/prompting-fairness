import json
import os
import torch
import numpy as np
import spacy
from scipy.spatial import distance
from src.bookcorpus.seat_utils import run_test
from src.prompt_tune_sandbox.bias_template import prepare_dataset_for_masked_model
from src.prompt_tune_sandbox.bias_template import create_positional_ids, PositionIdAdjustmentType
from src.prompt_tune_sandbox.stereoset_evaluator.stereoset import StereoSetRunner, ScoreEvaluator as StereoSetScoreEvaluator
from datasets import load_dataset

from .console import console

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

    @torch.inference_mode()
    def evaluate(self, model, tokenizer,
                 prompt_length=0, 
                 return_embeddings=True, 
                 position_id_adjustment=PositionIdAdjustmentType.none, 
                 return_seat_results=True,
                 return_occupation_dataset_results=True,
                 return_words_close_to_prompts=False, 
                 return_stereo_set_results=False,
                 stereoset_only_evaluate_gender=True,
                 return_perplexity=False,
                 ):

        console.log("[b]Starting evaluation...")
        return_value = {}

        if return_seat_results:
            console.log("Evaluating SEAT...")
            seat_results = {}
            seat_embeddings = {}
            for seat_test_name, seat_template in self.seat_templates.items():
                seat_results[seat_test_name], seat_embeddings[seat_test_name] = self.evaluate_seat(
                    seat_template, model, tokenizer, prompt_length, position_id_adjustment)
            return_value["seat_results"] = seat_results
            if return_embeddings:
                return_value["seat_embeddings"] = seat_embeddings
        
        if return_occupation_dataset_results:
            console.log("Evaluating training dataset")
            occupation_dataset_statistics = self.evaluate_training_occupation_dataset(model, tokenizer, prompt_length, position_id_adjustment)
            return_value["occupation_dataset"] = occupation_dataset_statistics

        if return_perplexity:
            console.log("Evaluating perplexity...")
            return_value["perplexity"] = self.evaluate_perplexity(model, tokenizer, prompt_length, position_id_adjustment)
        
        if return_words_close_to_prompts:
            console.log("Finding k closest words...")
            return_value["closest_k_words"] = ("prompt_words_list", self.find_k_closest_words(model, tokenizer))

        if return_stereo_set_results:
            console.log("Evaluating stereoset...")
            return_value["stereo_set"] = self.evaluate_stereo_set(
                model, tokenizer, prompt_length, only_evaluate_gender=stereoset_only_evaluate_gender, position_id_adjustment=position_id_adjustment)

        return return_value


    def encode(self,
               text: str,
               model,
               tokenizer,
               prompt_length,
               position_id_adjustment=PositionIdAdjustmentType.none,
               use_cls_only=False,
               ):
        tokens = tokenizer(text, return_tensors='pt').to(model.device)
        batch_size = tokens["input_ids"].size(0)
        seq_length = tokens["input_ids"].size(1)
        position_ids = create_positional_ids(batch_size, seq_length, prompt_length, model.device, position_id_adjustment)
        outputs = model(**tokens, position_ids=position_ids, output_hidden_states=True)

        # hidden_states [batch, seq_len, d_hidden]
        hidden_states = outputs.hidden_states[-1]
        if use_cls_only:
            # Only use the embedding corresponding to the CLS token
            return hidden_states[:, prompt_length, :].squeeze().detach().cpu().numpy()
        else:
            # Average over the last layer's hidden representations and normalize the vector
            # (same as in github.com/McGill-NLP/bias-bench/blob/main/bias_bench/benchmark/seat/seat.py)
            # Only tokens which are not part of the prompt are taken into account
            encoding = hidden_states[:, prompt_length:, :].mean(dim=1).squeeze().detach().cpu().numpy()
            encoding /= np.linalg.norm(encoding)
            return encoding

    @torch.inference_mode()
    def evaluate_seat(self, seat_template, model, tokenizer, prompt_length, position_id_adjustment=PositionIdAdjustmentType.none):
        with torch.no_grad():
            targ1_embeddings = np.array(
                [self.encode(sample, model, tokenizer, prompt_length, position_id_adjustment) for sample in seat_template["targ1"]["examples"]])
            targ2_embeddings = np.array(
                [self.encode(sample, model, tokenizer, prompt_length, position_id_adjustment) for sample in seat_template["targ2"]["examples"]])
            attr1_embeddings = np.array(
                [self.encode(sample, model, tokenizer, prompt_length, position_id_adjustment) for sample in seat_template["attr1"]["examples"]])
            attr2_embeddings = np.array(
                [self.encode(sample, model, tokenizer, prompt_length, position_id_adjustment) for sample in seat_template["attr2"]["examples"]])

        labels = (targ1_embeddings.shape[0] * [seat_template["targ1"]["category"]+"(targ1)"]) +\
            (targ2_embeddings.shape[0] * [seat_template["targ2"]["category"]+"(targ2)"]) +\
            (attr1_embeddings.shape[0] * [seat_template["attr1"]["category"]+"(attr1)"]) +\
            (attr2_embeddings.shape[0] * [seat_template["attr2"]["category"]+"(attr2)"])
        embeddings = np.vstack((targ1_embeddings, targ2_embeddings, attr1_embeddings, attr2_embeddings))

        effect_size, p_value = run_test(x=targ1_embeddings, y=targ2_embeddings,
                                    a=attr1_embeddings, b=attr2_embeddings,
                                    n_samples=4000, parametric=True)
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
    def evaluate_training_occupation_dataset(self, model, tokenizer, prompt_length, position_id_adjustment=PositionIdAdjustmentType.none):
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

        position_ids_male = create_positional_ids(input_ids_male.size(0), input_ids_male.size(1), prompt_length, device, position_id_adjustment)
        position_ids_female = create_positional_ids(input_ids_female.size(0), input_ids_female.size(1), prompt_length, device, position_id_adjustment)

        output_male = model(input_ids_male, attention_mask=attention_mask_male, position_ids=position_ids_male)
        output_female = model(input_ids_female, attention_mask=attention_mask_female, position_ids=position_ids_female)
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

    @torch.no_grad()
    def find_k_closest_words(self, model, tokenizer, k=5):
        word_embeddings : torch.nn.Embedding = model.bert.embeddings.word_embeddings
        embedding_values = word_embeddings.weight.detach().cpu().numpy()
        # shape: [vocab_size, hidden_size]
        prompt_embeddings = model.get_prompt_embedding_to_save().numpy()
        # shape: [prompt_length, hidden_size]
        closest_k_words_list = []
        for prompt_element in prompt_embeddings:
            dists = distance.cdist(np.expand_dims(prompt_element,0), embedding_values, 'cosine')[0]
            # shape: [vocab_size]
            closest_k_indices = np.argpartition(dists, list(range(k)))[:k]
            closest_k_words = tokenizer.convert_ids_to_tokens(closest_k_indices)
            closest_k_words_list.append(closest_k_words)
        return closest_k_words_list
    
    @torch.no_grad()
    def evaluate_stereo_set(self, model, tokenizer, prompt_length, only_evaluate_gender=True, position_id_adjustment=PositionIdAdjustmentType.none):
        current_dir = os.path.dirname(__file__)
        stereo_set_dataset_path = f"{current_dir}/stereoset_evaluator/data/stereoset/test.json"
        bias_types = ["gender"] if only_evaluate_gender else None
        runner = StereoSetRunner(
            model,
            tokenizer,
            input_file=stereo_set_dataset_path,
            batch_size=1,
            max_seq_length=128,
            prompt_tuning_prompt_length=prompt_length,
            bias_types_to_load=bias_types,
            prompt_tuning_positional_ids_type=position_id_adjustment,
        )
        predictions = runner()
        console.log("   Computing stereoset score...")
        score_evaluator = StereoSetScoreEvaluator(stereo_set_dataset_path, predictions=predictions, bias_types_to_load=bias_types)
        return score_evaluator.get_overall_results()["intrasentence"]

    @torch.inference_mode()
    def evaluate_perplexity(self, model, tokenizer, prompt_length, position_id_adjustment=PositionIdAdjustmentType.none, sentencize=True):
        # Load the dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        if sentencize:
            # Create a pipeline which divides the text into sentences
            nlp = spacy.load("en_core_web_md")
            nlp.disable_pipes("tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer")
            nlp.add_pipe('sentencizer')

        n_elements_total = len(dataset["text"])
        n_elements_processed = 0
        n_elements_since_last_update = 0
        total_score = 0
        total_n_tokens = 0
        for text in dataset["text"][:n_elements_total]:
            n_elements_processed += 1
            n_elements_since_last_update += 1
            if n_elements_since_last_update >= 0.05 * n_elements_total:
                n_elements_since_last_update = 0
                console.log(
                    f"    [Perplexity] Processed {int(100*(n_elements_processed/n_elements_total))}% of dataset. Total score: {total_score}. Total tokens: {total_n_tokens}. Perplexity:{np.exp(total_score / total_n_tokens)}")
            if not text:
                # skip empty strings
                continue
            if sentencize:
                doc = nlp(text)
                sentences = doc.sents
            else:
                sentences = [text]
            for sentence in sentences:
                text_input = sentence.text if sentencize else sentence
                if not text_input:
                    continue
                try:
                    score, n_tokens = self.pseudo_perplexity_score(text_input, model, tokenizer,
                                                                prompt_length=prompt_length, position_id_adjustment=position_id_adjustment)
                    if not n_tokens:
                        continue
                    # console.log(f"Score: {score}     NTokens:{n_tokens}")
                    total_score += score
                    total_n_tokens += n_tokens
                except Exception as e:
                    console.log(f"[ERROR] Message:{str(e)}\nSentence: {text_input}")
                    
        pseudo_perplexity = np.exp(total_score / total_n_tokens)
        return pseudo_perplexity

    def pseudo_perplexity_score(self, sentence, model, tokenizer, prompt_length=0, position_id_adjustment=PositionIdAdjustmentType.none):
        input_ids = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
        nr_nonspecial_tokens = input_ids.size(-1) - 2  # ignore [CLS] and [SEP]
        if nr_nonspecial_tokens <= 0:
            return 0, 0
        # mask each not-special token one by one, in a separate sentence
        input_ids_batch = input_ids.repeat(nr_nonspecial_tokens, 1)
        position_ids = create_positional_ids(input_ids_batch.size(0), input_ids_batch.size(1), prompt_length, model.device, position_id_adjustment)
        labels = torch.full((input_ids_batch.size(0), input_ids_batch.size(1)+prompt_length), -100).to(model.device)  # positions with value -100 are ignored
        labels[:,1+prompt_length:-1][torch.eye(nr_nonspecial_tokens).bool()] = input_ids_batch[:,1:-1].diagonal()
        input_ids_batch[:,1:-1].fill_diagonal_(tokenizer.mask_token_id)

        total_loss = 0.0
        # Divide the batch into smaller batches to avoid out of memory exceptions
        max_minibatch_size = 32
        for start_idx in range(0, nr_nonspecial_tokens, max_minibatch_size):
            input_ids_minibatch = input_ids_batch[start_idx:start_idx+max_minibatch_size,:]
            labels_minibatch = labels[start_idx:start_idx+max_minibatch_size,:]
            position_ids_minibatch = position_ids[start_idx:start_idx+max_minibatch_size,:] if position_ids is not None else None
            n_elements_minibatch = input_ids_minibatch.size(0)

            output = model(input_ids_minibatch, labels=labels_minibatch, position_ids=position_ids_minibatch)
            # The BERT loss function uses 'mean' reduction by default,
            # so we have to multiply by the nr of masks in the minibatch
            total_loss += n_elements_minibatch * output.loss.item()
        return total_loss, nr_nonspecial_tokens




        
        

