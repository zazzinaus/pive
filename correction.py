# # import json
# # import torch
# # import os
# # from tqdm import tqdm
# # from utils_back import query_model, get_template #load_vllm_model
# # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# # def load_data(dataset_name):
# #     with open(dataset_name, 'r') as f:
# #         data = json.load(f)
# #     return data

# # def preprocess_data(data, template_filename):
# #     inputs = []
# #     gold_fols = []
# #     for example in data:
# #         gold_fol = f"FOL premises: \n{example['fol_context']}\n FOL conclusion: \n{example['gold_conclusion']}"
# #         gen_fol = f"FOL premises: \n{example['generated_fol_premises']}\n FOL conclusion: \n{example['generated_fol_conclusion']}"
# #         template = get_template(
# #             task='train', 
# #             template_filename=template_filename, 
# #             formatting_kwargs={
# #                 'premises': example['nl_context'], 
# #                 'question': example['nl_question'], 
# #                 'generated_fol': gen_fol, 
# #                 'parsing_error': example['prover_status'], 
# #                 'exec_error': example['prover_error'],
# #             }
# #         )
# #         inputs.append(template)
# #         gold_fols.append(gold_fol)
# #     return inputs, gold_fols


# # if __name__ == "__main__":
# #     bnb_config = BitsAndBytesConfig(
# #             load_in_4bit=True,
# #             bnb_4bit_use_double_quant=True,
# #             bnb_4bit_quant_type="nf4",
# #             bnb_4bit_compute_dtype=torch.bfloat16
# #         )
# #     model_name = 'LorMolf/LogicLlama2-chat-direct'
# #     dataset_file = 'provatrans_folio_out.json'
# #     template_filename = 'folio'
# #     batch_size = 1
    
# #     data = load_data(dataset_file)
# #     prep_data, gold_fols = preprocess_data(data, template_filename)

# #     # # Limit the data to only 3 samples
# #     # prep_data = prep_data[:5]
# #     # gold_fols = gold_fols[:5]

# #     tokenizer = AutoTokenizer.from_pretrained(model_name) # Llamatokenizer
# #     tokenizer.add_special_tokens({
# #         "eos_token": "</s>",
# #         "bos_token": "<s>",
# #         "unk_token": '<unk>',
# #         "pad_token": '<unk>',
# #     })
# #     tokenizer.padding_side = "left"  # Allow batched inference
    
# #     model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')

    
# #     # correct FOLs using the model
# #     batch_corrected_fols = []
# #     for i in tqdm(range(1, len(prep_data) + 1, batch_size)):
# #         batch_inputs = prep_data[i:i + batch_size]
# #         batch_gold_fol = gold_fols[i:i + batch_size]

# #         for input_text in batch_inputs:
# #             corrected = query_model(tokenizer, model, input_prompts=input_text)
# #             # print(corrected)
# #             # print("\n-------------------------------------\n")
# #             corrected_fol = corrected.split("### Comments:\n")[3].strip()
# #             # print(corrections)

# #             # # Get the content after the third occurrence
# #             # corrected_content = corrections[3:]

# #             # # Join the content with '\n'.join
# #             # corrected_fol = ('\n').join(corrected_content)
# #             # #print(f"NOT FOL:\n{corrections}")
# #             # print(corrected_fol)
# #             #print("\n appesa nel batch \n")
# #             batch_corrected_fols.append(corrected_fol)
# #     #print(batch_corrected_fols)
# #     #compute loss
# #     with open ('best_out.json', 'w', encoding='utf-8') as f:
# #         json.dump(batch_corrected_fols, f, indent=2, ensure_ascii=False)
            








# # huggingface-cli login

# import json
# import torch
# import os
# from tqdm import tqdm
# from utils_back import query_model, get_template
# from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# def load_data(dataset_name):
#     with open(dataset_name, 'r') as f:
#         data = json.load(f)
#     return data

# def preprocess_data(data, template_filename):
#     inputs = []
#     gold_fols = []
#     for example in data:
#         gold_fol = f"FOL premises: \n{example['fol_context']}\n FOL conclusion: \n{example['gold_conclusion']}"
#         gen_fol = f"### FOL premises: \n{example['generated_fol_premises']}\n### FOL conclusion: \n{example['generated_fol_conclusion']}"
#         template = get_template(
#             task='train', 
#             template_filename=template_filename, 
#             formatting_kwargs={
#                 'premises': example['nl_context'], 
#                 'question': example['nl_question'], 
#                 'generated_fol': gen_fol, 
#                 'gold_answer': example['gold_answer'],
#                 'predicted_answer': example['prover_answer'],
#                 'parsing_error': example['prover_status'], 
#                 'exec_error': example['prover_error'],
#                 'is_correct': example['is_correct']
#             }
#         )
#         inputs.append(template)
#         gold_fols.append(gold_fol)
#     return inputs, gold_fols


# if __name__ == "__main__":
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
#     model_name = 'LorMolf/LogicLlama2-chat-direct'
#     dataset_file = 'prova_cont_folio_out.json'
#     template_filename = 'good_folio'
#     batch_size = 1
    
#     data = load_data(dataset_file)
#     prep_data, gold_fols = preprocess_data(data, template_filename)

#     tokenizer = LlamaTokenizer.from_pretrained(model_name)
#     tokenizer.add_special_tokens({
#         "eos_token": "</s>",
#         "bos_token": "<s>",
#         "unk_token": '<unk>',
#         "pad_token": '<unk>',
#     })
#     tokenizer.padding_side = "left"
    
#     model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')

#     # correct FOLs using the model
#     batch_corrected_fols = []
#     for i in tqdm(range(1, len(prep_data) + 1, batch_size)):
#         batch_inputs = prep_data[i:i + batch_size]
#         batch_gold_fol = gold_fols[i:i + batch_size]

#         for input_text in batch_inputs:
#             corrected = query_model(tokenizer, model, input_prompts=input_text)
            
#             # Print the full output produced by the LLM
#             #print(f"LLM Output for input {i}: \n{corrected}\n")
            
#             # Extract the corrected FOL from the LLM output
#             corrected_fol = corrected.split("### FOL Correction:\n")[1].strip()
            
#             batch_corrected_fols.append(corrected_fol)
    
#     # Save the corrected FOLs to a JSON file
#     with open('best_out.json', 'w', encoding='utf-8') as f:
#         json.dump(batch_corrected_fols, f, indent=2, ensure_ascii=False)



import json
import torch
from utils_back import query_model, get_template
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

class FOLRefiner:
    def __init__(self, model_name, template_filename, max_seq_length=2048):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.template_filename= template_filename
        # Change the variable names to avoid conflicts with TranslatorEngine
        self.refiner_tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.refiner_tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": '<unk>',
            "pad_token": '<unk>',
        })
        self.refiner_tokenizer.padding_side = "left"
        
        # Use a distinct model variable name
        self.refiner_model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=self.bnb_config, device_map='auto')
        self.refiner_model.max_seq_length = max_seq_length  # Set max_seq_length for the refiner model

    def preprocess_sample(self, example):
        prover_status = example.get('prover_status', '')  # Default to an empty string if not found
        prover_error = example.get('prover_error', '')  # Default to an empty string if not found
        is_correct = example.get('is_correct', None)  # Default to None if not found

        gold_fol = f"FOL premises: \n{example['fol_context']}\n FOL conclusion: \n{example['gold_conclusion']}"
        gen_fol = f"### FOL premises: \n{example['generated_fol_premises']}\n### FOL conclusion: \n{example['generated_fol_conclusion']}"

        template = get_template(
            task='train', 
            template_filename=self.template_filename, 
            formatting_kwargs={
                'premises': example['nl_context'], 
                'question': example['nl_question'], 
                'generated_fol': gen_fol, 
                'parsing_error': prover_status, 
                'exec_error': prover_error,
                'is_correct': is_correct
            }
        )
        return template, gold_fol

    def correct_fol(self, input_text):
        # Use the refiner model and tokenizer
        print(f"Using model: {self.refiner_model}")
        corrected = query_model(self.refiner_tokenizer, self.refiner_model, input_prompts=input_text)
        corrected_fol = corrected.split("### FOL Correction:\n")[1].strip()
        return corrected_fol

    def process_single_sample(self, sample):
        # Preprocess the data
        input_text = self.preprocess_sample(sample)
        
        # Correct the FOL using the refiner model
        corrected_fol = self.correct_fol(input_text)
        
        # Output the corrected FOL
        return corrected_fol

# Example usage:
if __name__ == "__main__":
    model_name = 'LorMolf/LogicLlama2-chat-direct'
    template_filename = 'good_folio'
    
    # Instantiate the translator
    refiner = FOLRefiner(model_name, template_filename)

    # Load a single sample from a dataset
    dataset_file = 'prova_cont_folio_out.json'
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    # Process only one sample (for example, the first one)
    single_sample = data[0]
    
    corrected_fol = refiner.process_single_sample(single_sample)
    
    print("Corrected FOL:", corrected_fol)
