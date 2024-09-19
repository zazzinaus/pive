import os
import shutil
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import TextStreamer, TrainingArguments
from unsloth import FastLanguageModel
## pip install evaluate / python-Levenshtein

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
    def __init__(self, dataset, split, output_filename, batch_size, sample_limit=None):
        self.dataset = dataset
        self.split = split
        self.output_filename = output_filename
        self.batch_size = batch_size
        self.sample_limit = sample_limit
        
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
        """Processes a batch of data by generating FOL translations."""
        batch_prompt_inputs = [
            self.format_prompt(premises, conclusion)
            for premises, conclusion in zip(batch_data.contexts, batch_data.questions)
        ]

        batch_fol_gen_questions = []
        for prompt in batch_prompt_inputs:
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=256, use_cache=True)
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            response = self.extract_response(generated_texts)
            batch_fol_gen_questions.append(response)

        return batch_fol_gen_questions
    
    def translate(self):
        """Main translation method."""
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

            # Generate FOL translations for the batch
            batch_fol_gen_questions = self.translate_batch(batch_data, folio)

            # Save every variable from the batch in the final output
            for j in range(len(batch)):
                generated_fol = batch_fol_gen_questions[j]
                gold_fol = batch_data.fol_gold_questions[j]
                context_id = batch_data.context_ids[j]
                nl_context = batch_data.contexts[j]
                nl_question = batch_data.questions[j]
                fol_context = batch_data.fol_contexts[j]  # FOL premises
                gold_answer = batch_data.gold_answers[j]
                example_id = batch_data.example_ids[j]

                # Split the generated_fol text to extract premises and conclusion
                if "FOL premises:" in generated_fol and "FOL conclusion:" in generated_fol:
                    parts = generated_fol.split("FOL conclusion:")
                    generated_premises = parts[0].split("FOL premises:")[1].strip()
                    generated_conclusion = parts[1].strip()
                else:
                    generated_premises = [] # ""
                    generated_conclusion = ""

                # Print generated premises and conclusion
                print(f"Generated Premises: {generated_premises}")
                print(f"Generated Conclusion: {generated_conclusion}")

                # Save the output in a dictionary
                output_dict = {
                    'generated_fol_premises': generated_premises,
                    'fol_context': fol_context,
                    'generated_fol_conclusion': generated_conclusion,
                    'gold_conclusion': gold_fol,
                    'gold_answer': gold_answer,
                    'example_id': example_id,
                    'nl_context': nl_context,
                    'nl_question': nl_question,
                    'context_id': context_id
                }
                json_folio.append(output_dict)

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
    output = 'trans_folio_out.json'
    batch_size = 1
    sample_limit = 50  # Process all samples, or set this to limit it to a number of samples

    engine = TranslatorEngine(dataset, split, output, batch_size, sample_limit)
    engine.translate()
