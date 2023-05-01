from string import Template
from datasets import Dataset
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

gendered_options = {
    "male" : {
        "gendered_word" : "he",
        "gender_word_possessive" : "his",
        "he_she" : "he",
        "his_her": "his",
    },
    "female" : {
        "gendered_word" : "she",
        "gender_word_possessive" : "her",
        "he_she" : "she",
        "his_her": "her",
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
]


def prepare_dataset_for_masked_model(tokenizer, return_unencoded_sentences=False):
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

    occupation_options_token_ids = []

    for option in occupation_options:
        token_id = tokenizer.convert_tokens_to_ids([option])[0]
        if token_id == tokenizer.unk_token_id:
            print(f"[WARNING] Option '{option}' is not a valid token")
        else:
            occupation_options_token_ids.append(token_id)

    if len(occupation_options_token_ids) > len(set(occupation_options_token_ids)):
        print("[WARNING] There are duplicate token ids for occupation options")

    if len(replaced_male) != len(replaced_female):
        raise RuntimeError("There should be an equal number of samples in each category")

    male_encodings = tokenizer(replaced_male, truncation=True, padding=True)
    female_encodings = tokenizer(replaced_female, truncation=True, padding=True)

    mask_token_index_male = [enc.index(tokenizer.mask_token_id) for enc in male_encodings["input_ids"]]
    mask_token_index_female = [enc.index(tokenizer.mask_token_id) for enc in female_encodings["input_ids"]]
    
    dataset = Dataset.from_dict(
        {
            "input_ids_male": male_encodings["input_ids"],
            "input_ids_female": female_encodings["input_ids"],
            "attention_mask_male": male_encodings["attention_mask"],
            "attention_mask_female": female_encodings["attention_mask"],
            "token_type_ids_male": male_encodings["token_type_ids"],
            "token_type_ids_female": female_encodings["token_type_ids"],
            "output_indices": len(replaced_female) * [occupation_options_token_ids],
            "mask_token_idx_male": mask_token_index_male,
            "mask_token_idx_female": mask_token_index_female,
        }
    )

    if return_unencoded_sentences:
        return dataset, replaced_male, replaced_female

    return dataset