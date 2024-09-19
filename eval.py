import os
import json
from tqdm import tqdm
from metrics import UniversalMetrics  # Assuming you have this for evaluation metrics like BLEU

class EvalEngine:
    def __init__(self, input_filename, output_filename, batch_size):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.batch_size = batch_size
        self.metric = UniversalMetrics()

    def load_data(self):
        if os.path.isfile(self.input_filename):
            with open(self.input_filename, 'r', encoding='utf-8') as input_file:
                data = json.load(input_file)
                return data
        else:
            raise FileNotFoundError(f"Input file '{self.input_filename}' not found.")
        
    def evaluation(self):
        """Main evaluation method."""
        folio = self.load_data()
        print(f"\nLoaded {len(folio)} examples for Evaluation\n")

        json_folio = []
        iterations = len(folio) // self.batch_size

        print(f'\nComparing generated FOL to gold FOL...\n')
        for i in tqdm(range(iterations + 1)):
            b_start = i * self.batch_size
            b_end = min((i + 1) * self.batch_size, len(folio))
            batch = [folio[j] for j in range(b_start, b_end)]

            for example in batch:
                # Debugging print statements to inspect the data
                print(f"\nExample ID: {example['example_id']}")
                print(f"Generated FOL Premises: {example['generated_fol_premises']}")
                print(f"FOL Context: {example['fol_context']}")
                
                generated_fol_premises = example['generated_fol_premises']
                fol_context = example['fol_context']
                
                # Handle empty premises
                if not generated_fol_premises or not fol_context:
                    print(f"Premises are empty for example ID {example['example_id']}. Assigning 0 to BLEU and LE scores.")
                    json_folio.append({
                        'example_id': example['example_id'],
                        'generated_premise': "N/A",
                        'gold_premise': "N/A",
                        'bleu': 0.,
                        'LE': 0.
                    })
                else:
                    # If it's already a list, no need to split it
                    if isinstance(generated_fol_premises, str):
                        generated_premises_list = generated_fol_premises.strip().split("\n")
                    else:
                        generated_premises_list = generated_fol_premises

                    if isinstance(fol_context, str):
                        fol_context_list = fol_context.strip().split("\n")
                    else:
                        fol_context_list = fol_context

                    for j, (gen_premise, gold_premise) in enumerate(zip(generated_premises_list, fol_context_list)):
                        print(f"Evaluating FOL Premise {j+1}...")
                        try:
                            eval_result = self.metric.evaluate(None, gold_premise, None, gen_premise)
                            bleu = eval_result.FOL_bleu
                            LE = eval_result.FOL_LE
                        except AssertionError as e:
                            print(f"Error evaluating FOL Premise: {e}")
                            bleu = 0.
                            LE = 0.

                        # Log the result for each premise
                        json_folio.append({
                            'example_id': example['example_id'],
                            'generated_premise': gen_premise,
                            'gold_premise': gold_premise,
                            'bleu': bleu,
                            'LE': LE
                        })

                # Handle empty conclusions
                generated_fol_conclusion = example['generated_fol_conclusion']
                gold_conclusion = example['gold_conclusion']

                if not generated_fol_conclusion or not gold_conclusion:
                    print(f"Conclusion is empty for example ID {example['example_id']}. Assigning 0 to BLEU and LE scores.")
                    json_folio.append({
                        'example_id': example['example_id'],
                        'generated_conclusion': "N/A",
                        'gold_conclusion': "N/A",
                        'bleu_conclusion': 0.,
                        'LE_conclusion': 0.
                    })
                else:
                    print(f"Evaluating FOL Conclusion...")
                    try:
                        eval_result_conclusion = self.metric.evaluate(None, gold_conclusion, None, generated_fol_conclusion)
                        bleu_conclusion = eval_result_conclusion.FOL_bleu
                        LE_conclusion = eval_result_conclusion.FOL_LE
                    except AssertionError as e:
                        print(f"Error evaluating FOL Conclusion: {e}")
                        bleu_conclusion = 0.
                        LE_conclusion = 0.

                    # Log the result for the conclusion
                    json_folio.append({
                        'example_id': example['example_id'],
                        'generated_conclusion': generated_fol_conclusion,
                        'gold_conclusion': gold_conclusion,
                        'bleu_conclusion': bleu_conclusion,
                        'LE_conclusion': LE_conclusion
                    })

        # Periodic save (save every 10 batches)
        if i % 10 == 0:
            with open(self.output_filename, 'w', encoding='utf-8') as f:
                json.dump(json_folio, f, indent=4, ensure_ascii=False)

        # Final save
        with open(self.output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_folio, f, indent=4, ensure_ascii=False)

        print("Evaluation completed!")

if __name__ == '__main__':
    input_filename = 'trans_folio_out.json'
    output_filename = 'evaluation_results.json'
    batch_size = 10

    engine = EvalEngine(input_filename, output_filename, batch_size)
    engine.evaluation()
