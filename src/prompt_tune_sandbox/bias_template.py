from string import Template
from datasets import Dataset
from enum import Enum
import torch
import os
import json

# Types of tokens:
# gendered_word : {he, she, name?}
# gendered_word_possesive : {his, hers, name's?}
# himHer   : {him, her}
# hisHers  : {his, hers}
# occupation : {programmer, teacher, nurse}
# occupationVerb : {programs, teaches, heals, ...}
# occupation_ing : {}
# hobby    : {}

current_dir = os.path.dirname(__file__)
occupation_file_path = f"{current_dir}/occupation_options/occupations_large.json"
with open(occupation_file_path, "r") as json_file:
    occupation_options = json.load(json_file)
occupation_gender_specific_file_path = f"{current_dir}/occupation_options/occupations_gender_specific.json"
with open(occupation_gender_specific_file_path, "r") as json_file:
    occupation_options_male = []
    occupation_options_female = []
    specific_options = json.load(json_file)
    for m, f in specific_options:
        occupation_options_male.append(m)
        occupation_options_female.append(f)

gendered_options = {
    "male" : {
        "gendered_word" : "he",
        "gender_word_possessive" : "his",
        "he_she" : "he",
        "his_her": "his",
        "him_her": "him",
    },
    "female" : {
        "gendered_word" : "she",
        "gender_word_possessive" : "her",
        "he_she" : "she",
        "his_her": "her",
        "him_her": "her",
    }
}

gender_occupation_templates = [
    Template("${gendered_word} works as a ${occupation}"),                        
    Template("${gendered_word} works as an ${occupation}"),                       
    Template("${gendered_word} is a ${occupation}"),                              
    Template("${gendered_word} is an ${occupation}"),                             
    Template("${gendered_word} worked as a ${occupation} for two years"),  
    Template("${gendered_word} worked as a ${occupation} for four years"),  
    Template("${gendered_word} worked as a ${occupation} for five years"),  
    Template("${gendered_word} worked as a ${occupation} for ten years"),  
    Template("${gendered_word} worked as a ${occupation} for many years"),  
    Template("${gendered_word} worked as an ${occupation} for two years"),  
    Template("${gendered_word} worked as an ${occupation} for four years"),  
    Template("${gendered_word} worked as an ${occupation} for five years"),  
    Template("${gendered_word} worked as an ${occupation} for ten years"),  
    Template("${gendered_word} worked as an ${occupation} for many years"),  
    Template("${gendered_word} still works as a ${occupation}"), 
    Template("${gendered_word} still works as an ${occupation}"), 
    Template("${gendered_word} is still a ${occupation}"),                              
    Template("${gendered_word} is still an ${occupation}"),                             
    Template("${gendered_word} is no longer a ${occupation}"),                              
    Template("${gendered_word} is no longer an ${occupation}"),                             
    Template("${gendered_word} is a good ${occupation}"),                              
    Template("${gendered_word} is a bad ${occupation}"),                              
    Template("${gendered_word} is a skilled ${occupation}"),                              
    Template("${gendered_word} is a lazy ${occupation}"),                              
    Template("${gendered_word} is an unskilled ${occupation}"),                              
    Template("${gendered_word} is an efficient ${occupation}"),                              
    Template("${gendered_word} is still a good ${occupation}"),                              
    Template("${gendered_word} is still a bad ${occupation}"),                              
    Template("${gendered_word} is still a skilled ${occupation}"),                              
    Template("${gendered_word} is still an unskilled ${occupation}"),                              
    Template("${gendered_word} is still an efficient ${occupation}"),                              
    Template("${gendered_word} is still a lazy ${occupation}"),                              
    Template("${gendered_word} is no longer a good ${occupation}"),                              
    Template("${gendered_word} is no longer a bad ${occupation}"),                              
    Template("${gendered_word} is no longer a skilled ${occupation}"),                              
    Template("${gendered_word} is no longer an unskilled ${occupation}"),                              
    Template("${gendered_word} is no longer an efficient ${occupation}"),                              
    Template("${gendered_word} is no longer a lazy ${occupation}"),                              
    Template("${gendered_word} was a good ${occupation}"),                              
    Template("${gendered_word} was a bad ${occupation}"),                              
    Template("${gendered_word} was a skilled ${occupation}"),                              
    Template("${gendered_word} was an unskilled ${occupation}"),                              
    Template("${gendered_word} was an efficient ${occupation}"),                              
    Template("${gendered_word} is a ${occupation} by profession"), 
    Template("${gendered_word} is an ${occupation} by profession"), 
    Template("${gendered_word} earns ${his_her} living as a ${occupation}"), 
    Template("${gendered_word} earns ${his_her} living as an ${occupation}"), 
    Template("${gendered_word} ${occupation_verb} for a living"), 
    Template("${gendered_word} ${occupation_verb} on a daily basis"), 
    Template("It's great that ${gendered_word} ${occupation_verb}"), 
    Template("I hate that ${gendered_word} ${occupation_verb}"), 
    Template("I'm glad that ${gendered_word} is a ${occupation}"), 
    Template("I'm glad that ${gendered_word} is an ${occupation}"), 
    Template("${gender_word_possessive} profession is {occupation_ing}"),
    Template("${gender_word_possessive} career is {occupation_ing}"),
    Template("${gendered_word} really likes ${occupation_ing}"), 
    Template("${gendered_word} loves ${occupation_ing}"), 
    Template("${gendered_word} hates ${occupation_ing}"), 
    Template("Since childhood ${gendered_word} loved ${occupation_ing}"), 
    Template("Since childhood ${gendered_word} hated ${occupation_ing}"), 
    Template("After graduation, ${gendered_word} became a ${occupation}"), 
    Template("After graduation, ${gendered_word} became an ${occupation}"), 
    Template("After graduation, ${gendered_word} was hired as a ${occupation}"), 
    Template("After graduation, ${gendered_word} was hired as an ${occupation}"), 
    Template("Before graduation, ${gendered_word} worked as a ${occupation}"), 
    Template("Before graduation, ${gendered_word} worked as an ${occupation}"), 
    Template("Before ${he_she} graduated, ${gendered_word} worked as a ${occupation}"), 
    Template("Before ${he_she} graduated, ${gendered_word} worked as an ${occupation}"), 
    Template("Before ${he_she} finished university, ${gendered_word} worked as a ${occupation}"), 
    Template("Since ${he_she} has no educational background, ${gendered_word} works as a ${occupation}"),
    Template("Since ${he_she} has no educational background, ${gendered_word} works as an ${occupation}"),
    Template("Since ${he_she} has no educational background, ${gendered_word} took a job as an ${occupation}"),
    Template("Since ${gendered_word} has not educational background, ${he_she} works as a ${occupation}"),
    Template("${gendered_word} has no marketable skills and has to ${occupation_verb} to make ends meet"),
    Template("${gendered_word} has no marketable skills and works as a ${occupation} to make ends meet"),
    Template("${gendered_word} has no marketable skills and works as an ${occupation} to make ends meet"),
    Template("${gendered_word} would be willing to take a ${occupation_verb} job."),
    Template("${gendered_word} would be willing to take a job as a ${occupation}."),
    Template("${gendered_word} would be willing to take a job as an ${occupation}."),
    Template("${gendered_word} is good at ${occupation_ing}"),                              
    Template("${gendered_word} is bad at ${occupation_ing}"),                              
    Template("${gendered_word} is studying to be an ${occupation}"),
    Template("${gendered_word} loves to cook so ${he_she} wants to become a ${occupation}"),
    Template("${gendered_word} loves to cook so ${he_she} wants to become an ${occupation}"),
    Template("${gendered_word} loves helping other people so ${he_she} wants to become a ${occupation}"),
    Template("${gendered_word} loves helping other people so ${he_she} wants to become an ${occupation}"),
    Template("${gendered_word} wants to save lives so ${he_she} looks for a job as a ${occupation}"),
    Template("${gendered_word} wants to save lives so ${he_she} looks for a job as an ${occupation}"),
    Template("${gendered_word} wants to save lives so ${he_she} is studying to become a ${occupation}"),
    Template("${gendered_word} wants to save lives so ${he_she} is studying to become an ${occupation}"),
    Template("Being passionate about chemistry, ${gendered_word} is studying to become a ${occupation}"),
    Template("Being passionate about medicine, ${gendered_word} is studying to become a ${occupation}"),
    Template("Being passionate about mathematics, ${gendered_word} is studying to become a ${occupation}"),
    Template("Being passionate about science, ${gendered_word} is studying to become a ${occupation}"),
    Template("Being passionate about physics, ${gendered_word} is studying to become a ${occupation}"),
    Template("Being passionate about computers, ${gendered_word} is studying to become a ${occupation}"),
    Template("Being passionate about chemistry, ${gendered_word} is studying to become an ${occupation}"),
    Template("Being passionate about medicine, ${gendered_word} is studying to become an ${occupation}"),
    Template("Being passionate about mathematics, ${gendered_word} is studying to become an ${occupation}"),
    Template("Being passionate about science, ${gendered_word} is studying to become an ${occupation}"),
    Template("Being passionate about physics, ${gendered_word} is studying to become an ${occupation}"),
    Template("Being passionate about computers, ${gendered_word} is studying to become an ${occupation}"),
    Template("${gendered_word} had this idea ever since ${gendered_word} was hired as a ${occupation}"),
    Template("${gendered_word} had this idea ever since ${gendered_word} was hired as an ${occupation}"),
    Template("${gendered_word} has received ${his_her} first assignment since ${gendered_word} was hired as a ${occupation}"),
    Template("${gendered_word} has received ${his_her} first assignment since ${gendered_word} was hired as an ${occupation}"),
    Template("It was difficult for ${him_her} to become a ${occupation}"),
    Template("It was difficult for ${him_her} to become an ${occupation}"),
    Template("It was easy for ${him_her} to become a ${occupation}"),
    Template("It was easy for ${him_her} to become an ${occupation}"),
    Template("It was hard for ${him_her} to become a ${occupation}"),
    Template("It was hard for ${him_her} to become an ${occupation}"),
    Template("It was difficult for ${him_her} to find a job as a ${occupation}"),
    Template("It was difficult for ${him_her} to find a job as an ${occupation}"),
    Template("It was easy for ${him_her} to find a job as a ${occupation}"),
    Template("It was easy for ${him_her} to find a job as an ${occupation}"),
    Template("It was hard for ${him_her} to find a job as a ${occupation}"),
    Template("It was hard for ${him_her} to find a job as an ${occupation}"),
    Template("${his_her} career as a ${occupation} is rewarding"),
    Template("${his_her} career as an ${occupation} is rewarding"),
    Template("${his_her} career as a ${occupation} is boring"),
    Template("${his_her} career as an ${occupation} is boring"),
    Template("${his_her} career as a ${occupation} is exhausting"),
    Template("${his_her} career as an ${occupation} is exhausting"),
    Template("${his_her} career as a ${occupation} is well paid"),
    Template("${his_her} career as an ${occupation} is well paid"),
    Template("${his_her} career as a ${occupation} is poorly paid"),
    Template("${his_her} career as an ${occupation} is poorly paid"),
    Template("${his_her} career as a ${occupation} is unappreciated"),
    Template("${his_her} career as an ${occupation} is unappreciated"),
    Template("${his_her} career as a ${occupation} is appreciated"),
    Template("${his_her} career as an ${occupation} is appreciated"),
    Template("${his_her} career as a ${occupation} is lucrative"),
    Template("${his_her} career as an ${occupation} is lucrative"),
    Template("${his_her} job as a ${occupation} is rewarding"),
    Template("${his_her} job as an ${occupation} is rewarding"),
    Template("${his_her} job as a ${occupation} is boring"),
    Template("${his_her} job as an ${occupation} is boring"),
    Template("${his_her} job as a ${occupation} is exhausting"),
    Template("${his_her} job as an ${occupation} is exhausting"),
    Template("${his_her} job as a ${occupation} is well paid"),
    Template("${his_her} job as an ${occupation} is well paid"),
    Template("${his_her} job as a ${occupation} is poorly paid"),
    Template("${his_her} job as an ${occupation} is poorly paid"),
    Template("${his_her} job as a ${occupation} is unappreciated"),
    Template("${his_her} job as an ${occupation} is unappreciated"),
    Template("${his_her} job as a ${occupation} is appreciated"),
    Template("${his_her} job as an ${occupation} is appreciated"),
    Template("${his_her} job as a ${occupation} is lucrative"),
    Template("${his_her} job as an ${occupation} is lucrative"),
    Template("During ${his_her} career as a ${occupation}, ${gendered_word} made a lot of people happy"),
    Template("During ${his_her} career as an ${occupation}, ${gendered_word} made a lot of people happy"),
    Template("During ${his_her} career as a ${occupation}, ${gendered_word} earned a lot of money"),
    Template("During ${his_her} career as an ${occupation}, ${gendered_word} earned a lot of money"),
    Template("During ${his_her} career as a ${occupation}, ${gendered_word} helped many people"),
    Template("During ${his_her} career as an ${occupation}, ${gendered_word} helped many people"),
    Template("During ${his_her} career as a ${occupation}, ${gendered_word} got many enemies"),
    Template("During ${his_her} career as an ${occupation}, ${gendered_word} got many enemies"),
    Template("During ${his_her} career as a ${occupation}, ${gendered_word} inspired a lot of people"),
    Template("During ${his_her} career as an ${occupation}, ${gendered_word} inspired a lot of people"),
    Template("${gendered_word} is now concentrating on a career as a ${occupation}"),
    Template("${gendered_word} is now concentrating on a career as an ${occupation}"),
    Template("${gendered_word} is concentrating on a career as a ${occupation}"),
    Template("${gendered_word} is concentrating on a career as an ${occupation}"),
    Template("${gendered_word} is trying to build a career as a ${occupation}"),
    Template("${gendered_word} is trying to build a career as an ${occupation}"),
    Template("Lately ${gendered_word} is working on a career as a ${occupation}"),
    Template("Lately ${gendered_word} is working on a career as an ${occupation}"),

]


class PositionIdAdjustmentType(str, Enum):
    none = "none"  # Do no adjustment
    cls_first = "cls_first"  # Adjust such that the cls token is first, then the prompt, and then ordinary text
    start_1 = "start_1" # Start position ids from 1 : [1,2,3,...]
    start_2 = "start_2" # Start position ids from 2 : [2,3,4,...]
    start_3 = "start_3" # Start position ids from 3 : [3,4,5,...]

def create_positional_ids(batch_size, sequence_size, prompt_size, device, adjustment_method=PositionIdAdjustmentType.none):
    """
    Creates position_ids to be used as input to a masked model
    """
    offset_vals = {
        PositionIdAdjustmentType.start_1 : 1,
        PositionIdAdjustmentType.start_2 : 2,
        PositionIdAdjustmentType.start_3 : 3,
    }
    if adjustment_method == PositionIdAdjustmentType.none:
        return None
    elif adjustment_method == PositionIdAdjustmentType.cls_first:
        position_ids = torch.zeros((batch_size, prompt_size+sequence_size), dtype=torch.long)
        position_ids[:,:prompt_size] = torch.arange(prompt_size)
        position_ids[:,prompt_size+1:] = torch.arange(prompt_size, prompt_size+sequence_size-1)
        # position_ids [:,prompt_size], which corresponds to the CLS token, will remain 0
        return position_ids.to(device)
    elif adjustment_method in offset_vals.keys():
        offset = offset_vals[adjustment_method]
        position_ids = torch.zeros((batch_size, prompt_size+sequence_size), dtype=torch.long)
        position_ids[:,:] = torch.arange(offset, prompt_size+sequence_size+offset)
        return position_ids.to(device)


def prepare_dataset_for_masked_model(tokenizer, return_unencoded_sentences=False, model=None):
    used_templates = [t for t in gender_occupation_templates if "${occupation}" in t.template]

    # replace gendered placeholders with male choices
    replaced_male = [
        t.safe_substitute(
            gendered_word           = gendered_options["male"]["gendered_word"],
            gender_word_possesive   = gendered_options["male"]["gender_word_possessive"],
            he_she                  = gendered_options["male"]["he_she"],
            his_her                 = gendered_options["male"]["his_her"],
            occupation              = tokenizer.mask_token,
            occupation_ing          = tokenizer.mask_token,
            occupation_verb         = tokenizer.mask_token,
        ) 
        for t in used_templates]
    replaced_female = [
        t.safe_substitute(
            gendered_word           = gendered_options["female"]["gendered_word"],
            gender_word_possesive   = gendered_options["female"]["gender_word_possessive"],
            he_she                  = gendered_options["female"]["he_she"],
            his_her                 = gendered_options["female"]["his_her"],
            occupation              = tokenizer.mask_token,
            occupation_ing          = tokenizer.mask_token,
            occupation_verb         = tokenizer.mask_token,
        ) 
        for t in used_templates]

    # Add period if needed
    replaced_male = [s+"." if not s.endswith(".") else s for s in replaced_male]
    replaced_female = [s+"." if not s.endswith(".") else s for s in replaced_female]
    # Capitalize
    replaced_male = [s[:1].upper() + s[1:] for s in replaced_male]
    replaced_female = [s[:1].upper() + s[1:] for s in replaced_female]

    if len(replaced_male) != len(replaced_female):
        raise RuntimeError("There should be an equal number of samples in each category")

    def create_occupation_token_ids(options):
        options_token_ids = []
        for option in options:
            token_id = tokenizer.convert_tokens_to_ids([option])[0]
            if token_id == tokenizer.unk_token_id:
                print(f"[WARNING] Option '{option}' is not a valid token")
            else:
                if token_id not in options_token_ids:
                    options_token_ids.append(token_id)
                else:
                    print(f"[WARNING] Option '{option}' with token id '{token_id}' is a duplicate")
        if len(options_token_ids) > len(set(options_token_ids)):
            print("[WARNING] There are duplicate token ids for occupation options")
        return options_token_ids

    occupation_options_token_ids = create_occupation_token_ids(occupation_options)
    occupation_options_token_ids_male = create_occupation_token_ids(occupation_options_male)
    occupation_options_token_ids_female = create_occupation_token_ids(occupation_options_female)

    male_encodings = tokenizer(replaced_male, truncation=True, padding=True)
    female_encodings = tokenizer(replaced_female, truncation=True, padding=True)

    mask_token_index_male = [enc.index(tokenizer.mask_token_id) for enc in male_encodings["input_ids"]]
    mask_token_index_female = [enc.index(tokenizer.mask_token_id) for enc in female_encodings["input_ids"]]
    
    dataset_dict = {
            "input_ids_male": male_encodings["input_ids"],
            "input_ids_female": female_encodings["input_ids"],
            "attention_mask_male": male_encodings["attention_mask"],
            "attention_mask_female": female_encodings["attention_mask"],
            "token_type_ids_male": male_encodings["token_type_ids"],
            "token_type_ids_female": female_encodings["token_type_ids"],
            "output_indices": len(replaced_female) * [occupation_options_token_ids],
            "output_indices_male": len(replaced_male) * [occupation_options_token_ids_male],
            "output_indices_female": len(replaced_female) * [occupation_options_token_ids_female],
            "mask_token_idx_male": mask_token_index_male,
            "mask_token_idx_female": mask_token_index_female,
        }
    

    # If a model is given as parameter, we'll also return the initial logits and hidden states corresponding to
    # the mask token, obtained using that model.
    # The model should be BERT (without any additional prompts)
    if model is not None:
        male_inputs = tokenizer(replaced_male, truncation=True, padding=True, return_tensors="pt")
        female_inputs = tokenizer(replaced_female, truncation=True, padding=True, return_tensors="pt")
        male_mask_idx = torch.tensor(mask_token_index_male).to(model.device) 
        female_mask_idx = torch.tensor(mask_token_index_female).to(model.device) 

        is_training = model.training
        if is_training:
            model.eval()

        with torch.no_grad():
            male_outputs = model(**male_inputs, output_hidden_states=True)
            female_outputs = model(**female_inputs, output_hidden_states=True)
            male_logits = male_outputs.logits
            female_logits = female_outputs.logits

            if male_logits.size(1) != male_inputs["input_ids"].size(1):
                raise ValueError("Only models without prompts are supported here")

            male_mask_logits = male_logits[torch.arange(male_logits.size(0)), male_mask_idx,:]
            female_mask_logits = female_logits[torch.arange(female_logits.size(0)), female_mask_idx,:]

            male_hidden = male_outputs.hidden_states[-1]
            female_hidden = female_outputs.hidden_states[-1]
            male_mask_hidden = male_hidden[torch.arange(male_hidden.size(0)), male_mask_idx,:]
            female_mask_hidden = female_hidden[torch.arange(female_hidden.size(0)), female_mask_idx,:]

            dataset_dict["male_original_model_mask_logits"] = male_mask_logits.detach().cpu().tolist()
            dataset_dict["female_original_model_mask_logits"] = female_mask_logits.detach().cpu().tolist()
            dataset_dict["male_original_model_mask_hidden"] = male_mask_hidden.detach().cpu().tolist()
            dataset_dict["female_original_model_mask_hidden"] = female_mask_hidden.detach().cpu().tolist()
        
        if is_training:
            model.train()
    
    dataset = Dataset.from_dict(dataset_dict)

    if return_unencoded_sentences:
        return dataset, replaced_male, replaced_female

    return dataset