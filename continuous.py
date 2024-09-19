import os
import shutil
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import TextStreamer, TrainingArguments
from unsloth import FastLanguageModel
## pip install evaluate / python-Levenshtein
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program

def safe_execute_program(premises, conclusion):
    """Safely execute the logic program."""
    program = FOL_Prover9_Program(premises, conclusion)
    if not program.flag:
        return None, 'parsing error', '', None
    
    answer, error_message, unification_stack = program.execute_program()
    if answer is None:
        return answer, 'execution error', error_message, unification_stack
    
    return program.answer_mapping(answer), '', '', unification_stack 

def inference_on_generated_fol(generated_premises, generated_conclusion, gold_answer):
    """Perform inference after translation."""
    answer, flag, error_message, unification_stack = safe_execute_program(generated_premises, generated_conclusion)
    
    # Mapping the prover's answer
    answer_mapping = {
        'A': "True",
        'B': "False",
        'C': 'Uncertain'
    }
    prover_answer = answer_mapping.get(answer, None)  # Default to None if not found

    # Check if prover_answer matches gold_answer
    is_correct = prover_answer == gold_answer

    return prover_answer, flag, error_message, is_correct



class BatchData:
    """Helper class to store and manage batch data."""
    def __init__(self, batch):
        self.context_ids = [item['story_id'] for item in batch]
        self.contexts = [item['premises'] for item in batch]
        self.fol_contexts = [item['premises-FOL'] for item in batch]
        self.questions = [item['conclusion'] for item in batch]
        self.fol_gold_questions = [item['conclusion-FOL'] for item in batch]
        self.gold_answers = [item['label'] for item in batch]
        self.example_ids = [item['example_id'] for item in batch]


class TranslatorEngine:
    def __init__(self, dataset, split, output_filename, batch_size, sample_limit=None, continuous=False):
        self.dataset = dataset
        self.split = split
        self.output_filename = output_filename
        self.batch_size = batch_size
        self.sample_limit = sample_limit
        self.continuous = continuous
        
        # Model configuration
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        
        # Load model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        
        # Apply PEFT (Parameter-Efficient Fine-Tuning)
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Enable faster inference
        FastLanguageModel.for_inference(self.model)

    def load_data(self):
        if self.dataset == 'folio':
            dataset = load_dataset(f'tasksource/{self.dataset}', split=self.split)
            if self.sample_limit > 0:
                dataset = dataset.select(range(self.sample_limit))
            return dataset
        else:
            raise ValueError('Dataset not found')
        
    def format_prompt(self, nl_premise, nl_conclusion):
        prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Translate the following natural language (NL) premises and conclusions to first-order logic (FOL) rules.
Only translate the sentences to FOL, do not include any other information. Only include the FOL rule in the answer box.
Do not add any extra symbols or words to the FOL rule. Do not include any kind of explanation or description in the answer box.
Respond only in the format:
FOL premises: 
NL premises translated into FOL

FOL conclusion: 
NL conclusion translated into FOL.

Take the following NL to FOL as an example:

# Example 1:
NL premises:
All people who regularly drink coffee are dependent on caffeine.
People regularly drink coffee, or they don't want to be addicted to caffeine, or both.
No one who doesn't want to be addicted to caffeine is unaware that caffeine is a drug.
Rina is either a student who is unaware that caffeine is a drug, or she is not a student and is she aware that caffeine is a drug.
Rina  is either a student who is dependent on caffeine, or she is not a student and not dependent on caffeine

NL conclusion:
Rina either doesn't want to be addicted to caffeine or is unaware that caffeine is a drug.

FOL premises:
∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))
∀x (DrinkRegularly(x, coffee)  ∨ (¬WantToBeAddictedTo(x, caffeine)))
∀x (¬WantToBeAddictedTo(x, caffeine) → ¬AwareThatDrug(x, caffeine))
¬(Student(rina) ⊕  ¬AwareThatDrug(rina, caffeine))
¬(IsDependentOn(rina, caffeine) ⊕ Student(rina))<|eot_id|>

FOL conclusion:
¬WantToBeAddictedTo(rina, caffeine) ⊕ ¬AwareThatDrug(rina, caffeine)

# Example 2:
NL premises:
All people who regularly drink coffee are dependent on caffeine.
People regularly drink coffee, or they don't want to be addicted to caffeine, or both.
No one who doesn't want to be addicted to caffeine is unaware that caffeine is a drug.
Rina is either a student who is unaware that caffeine is a drug, or she is not a student and is she aware that caffeine is a drug.
Rina  is either a student who is dependent on caffeine, or she is not a student and not dependent on caffeine

NL conclusion:
Rina either doesn't want to be addicted to caffeine or is unaware that caffeine is a drug.

FOL premises:
∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))
∀x (DrinkRegularly(x, coffee)  ∨ (¬WantToBeAddictedTo(x, caffeine)))
∀x (¬WantToBeAddictedTo(x, caffeine) → ¬AwareThatDrug(x, caffeine))
¬(Student(rina) ⊕  ¬AwareThatDrug(rina, caffeine))
¬(IsDependentOn(rina, caffeine) ⊕ Student(rina))

FOL conclusion:
DrinkRegularly(rina, coffee)  ⊕ IsUnawareThatCaffeineIsADrug(rina)
You are a helpful assistant.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
NL premises:
{}
NL conclusion:
{}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{}"""
        return prompt.format(nl_premise, nl_conclusion, "") + self.tokenizer.eos_token

    def extract_response(self, text):
            """Extract the assistant's response from the generated text."""
            return text.split("assistant")[2].strip()


    def translate_batch(self, batch_data, folio):
        """Processes a batch of data by generating FOL translations and running inference iteratively, inserting refinement string before '<|start_header_id|>user<|end_header_id|>'."""
        batch_prompt_inputs = [
            self.format_prompt(premises, conclusion)
            for premises, conclusion in zip(batch_data.contexts, batch_data.questions)
        ]

        batch_fol_gen_questions = []
        json_folio = []

        for idx, prompt in enumerate(batch_prompt_inputs):
            refined_response = None
            base_prompt = prompt  # Store the original prompt structure
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

            # Split the base prompt to identify where to insert the refinement string
            prompt_before_user = base_prompt.split("<|start_header_id|>user<|end_header_id|>")[0]
            prompt_after_user = "<|start_header_id|>user<|end_header_id|>" + base_prompt.split("<|start_header_id|>user<|end_header_id|>")[1]

            # Repeat the process three times for iterative refinement
            for i in range(3):
                # Print the prompt that will be fed into the LLM
                print(f"Iteration {i + 1} - Prompt being fed to the LLM:\n{prompt}\n")

                outputs = self.model.generate(**inputs, max_new_tokens=256, use_cache=True)
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                response = self.extract_response(generated_texts)

                # Split the generated_fol text to extract premises and conclusion
                if "FOL premises:" in response and "FOL conclusion:" in response:
                    parts = response.split("FOL conclusion:")
                    generated_premises = parts[0].split("FOL premises:")[1].strip()
                    generated_conclusion = parts[1].strip()
                else:
                    generated_premises = ""  # No FOL premises found
                    generated_conclusion = ""  # No FOL conclusion found

                # Initialize default prover fields
                prover_answer, prover_status, prover_error, is_correct = None, None, None, None

                # Only run inference if continuous flag is True
                if self.continuous:
                    prover_answer, prover_status, prover_error, is_correct = inference_on_generated_fol(
                        generated_premises, generated_conclusion, batch_data.gold_answers[idx]
                    )
                    
                    # Add iterative refinement string to the prompt for the next iteration
                    refinement_string = f"these are the results for inference (round {i + 1}): {prover_answer}{prover_status}{prover_error}{is_correct}\n\n"

                    # Insert the refinement string before "<|start_header_id|>user<|end_header_id|>"
                    prompt = prompt_before_user + refinement_string + prompt_after_user

                # Use the current response as input for the next round (but don't change the base prompt)
                inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

                # Save the latest refined response
                refined_response = response

            # Save the output in a dictionary with the refined response
            output_dict = {
                'generated_fol_premises': generated_premises,
                'fol_context': batch_data.fol_contexts[idx],
                'generated_fol_conclusion': generated_conclusion,
                'gold_conclusion': batch_data.fol_gold_questions[idx],
                'gold_answer': batch_data.gold_answers[idx],
                'example_id': batch_data.example_ids[idx],
                'nl_context': batch_data.contexts[idx],
                'nl_question': batch_data.questions[idx],
                'context_id': batch_data.context_ids[idx],
                'prover_answer': prover_answer,
                'prover_status': prover_status,
                'prover_error': prover_error,
                'is_correct': is_correct,
                'refined_response': refined_response  # Store the final refined response after 3 iterations
            }

            json_folio.append(output_dict)

        return json_folio
       
    def translate(self):
        """Main translation method with optional inference."""
        folio = self.load_data()
        print(f"\nLoaded {len(folio)} examples for Translation\n")

        json_folio = []
        iterations = len(folio) // self.batch_size

        print(f'\nTranslation from NL to FOL...\n')
        for i in tqdm(range(iterations + 1)):
            b_start = i * self.batch_size
            b_end = min((i + 1) * self.batch_size, len(folio))
            batch = [folio[j] for j in range(b_start, b_end)]
            batch_data = BatchData(batch)

            # Generate FOL translations and optionally run inference for the batch
            batch_fol_json = self.translate_batch(batch_data, folio)
            json_folio.extend(batch_fol_json)

            # Periodic save (save every 5 batches)
            if i % 5 == 0:
                with open(self.output_filename, 'w', encoding='utf-8') as f:
                    json.dump(json_folio, f, indent=4, ensure_ascii=False)

        # Final save
        with open(self.output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_folio, f, indent=4, ensure_ascii=False)

        self.cleanup()

    def cleanup(self):
        """Clean up temporary files."""
        compiled_krb_dir = './compiled_krb'
        if os.path.exists(compiled_krb_dir):
            print('Removing compiled_krb directory...')
            shutil.rmtree(compiled_krb_dir)

# Example usage:
if __name__ == '__main__':
    dataset = 'folio'
    split = 'train'
    output = 'prova_cont_folio_out.json'
    batch_size = 1
    sample_limit = 10 # Process all samples, or set this to limit it to a number of samples
    continuous = True  # Set this flag to True or False

    engine = TranslatorEngine(dataset, split, output, batch_size, sample_limit, continuous)
    engine.translate() 

