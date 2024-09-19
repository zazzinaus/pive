

# import json
# import os
# from tqdm import tqdm
# from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
# import argparse

# class LogicInferenceEngine:
#     def __init__(self, dataset_name, input_filename, output_filename):
#         self.dataset_name = dataset_name
#         self.input_file_name = input_filename
#         self.output_filename = output_filename
#         self.dataset = self.load_logic_programs()

#     def load_logic_programs(self):
#         if not self.input_file_name:
#             raise ValueError("Input file name is not set.")
#         with open(self.input_file_name) as f:
#             dataset = json.load(f)
#         print(f"\nLoaded {len(dataset)} examples for logic inference.")
#         return dataset

#     def safe_execute_program(self, premises, conclusion):
#         program = FOL_Prover9_Program(premises, conclusion)
#         if not program.flag:
#             return None, 'parsing error', '', None
        
#         answer, error_message, unification_stack = program.execute_program()
#         if answer is None:
#             return answer, 'execution error', error_message, unification_stack
        
#         return program.answer_mapping(answer), '', '', unification_stack 

#     def inference_on_dataset(self):
#         outputs = []
#         match_count = 0

#         print("\nLogic Inference on translations...\n")

#         answer_mapping = {
#             'A': "True",
#             'B': "False",
#             'C': 'Uncertain'
#         }

#         for example in tqdm(self.dataset):
#             generated_premises = example['generated_fol_premises']
#             generated_conclusion = example['generated_fol_conclusion']

#             answer, flag, error_message, stack = self.safe_execute_program(
#                 generated_premises, generated_conclusion
#             )

#             # Get the gold answer from the example
#             gold_answer = example['gold_answer']
            
#             # Map prover_answer to gold_answer using answer_mapping
#             prover_answer = answer_mapping.get(answer, None)  # Default to None if not found

#             # Check if prover_answer matches gold_answer
#             if prover_answer == gold_answer:
#                 match_count += 1  # Increment match counter

#             output_dict = {
#                 'generated_fol_premises': generated_premises,
#                 'fol_context': example['fol_context'],
#                 'generated_fol_conclusion': generated_conclusion,
#                 'gold_conclusion': example['gold_conclusion'],
#                 'gold_answer': gold_answer,  # Ensure we store the original gold_answer
#                 'example_id': example['example_id'],
#                 'nl_context': example['nl_context'],
#                 'nl_question': example['nl_question'],
#                 'context_id': example['context_id'],
#                 'prover_answer': answer,
#                 'prover_status': flag,
#                 'prover_error': error_message,
#                 'is_correct': prover_answer == gold_answer  # Optional: add correctness flag
#             }
#             outputs.append(output_dict)

#         with open(self.output_filename, 'w') as f:
#             json.dump(outputs, f, indent=2, ensure_ascii=False)
        
#         print(f"Total matches: {match_count}")

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_name', type=str)
#     parser.add_argument('--input_filename', type=str)
#     parser.add_argument('--save_path', type=str, default='./outputs/logic_inference.json')
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()
#     engine = LogicInferenceEngine(args.dataset_name, args.input_filename, args.save_path)
#     engine.inference_on_dataset()
import json
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program

def load_single_logic_program(input_filename, example_id):
    """Load a specific example by ID from the dataset."""
    if not input_filename:
        raise ValueError("Input file name is not set.")
    with open(input_filename) as f:
        dataset = json.load(f)
    
    for example in dataset:
        if example['example_id'] == example_id:
            return example
    raise ValueError(f"Example ID {example_id} not found in the dataset.")

def safe_execute_program(premises, conclusion):
    """Safely execute the logic program."""
    program = FOL_Prover9_Program(premises, conclusion)
    if not program.flag:
        return None, 'parsing error', '', None
    
    answer, error_message, unification_stack = program.execute_program()
    if answer is None:
        return answer, 'execution error', error_message, unification_stack
    
    return program.answer_mapping(answer), '', '', unification_stack 

def inference_on_single_example(input_filename, example_id, output_filename):
    """Perform inference on a single example."""
    example = load_single_logic_program(input_filename, example_id)
    
    generated_premises = example['generated_fol_premises']
    generated_conclusion = example['generated_fol_conclusion']

    # Execute the program for premises and conclusion
    answer, flag, error_message, stack = safe_execute_program(generated_premises, generated_conclusion)

    # Gold answer
    gold_answer = example['gold_answer']

    # Mapping the prover's answer
    answer_mapping = {
        'A': "True",
        'B': "False",
        'C': 'Uncertain'
    }
    prover_answer = answer_mapping.get(answer, None)  # Default to None if not found

    # Check if prover_answer matches gold_answer
    is_correct = prover_answer == gold_answer

    # Prepare output
    output_dict = {
        'generated_fol_premises': generated_premises,
        'fol_context': example['fol_context'],
        'generated_fol_conclusion': generated_conclusion,
        'gold_conclusion': example['gold_conclusion'],
        'gold_answer': gold_answer,
        'example_id': example['example_id'],
        'nl_context': example['nl_context'],
        'nl_question': example['nl_question'],
        'context_id': example['context_id'],
        'prover_answer': prover_answer,
        'prover_status': flag,
        'prover_error': error_message,
        'is_correct': is_correct
    }

    # Save the output to a file
    with open(output_filename, 'w') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Output saved to {output_filename}")
    print(f"Prover Answer: {prover_answer}, Gold Answer: {gold_answer}, Match: {is_correct}")

# Example usage
if __name__ == "__main__":
    input_filename = "trans_folio_out.json"
    example_id = 1192  # ID of the example you want to run inference on
    output_filename = "single_logic_inference.json"
    
    inference_on_single_example(input_filename, example_id, output_filename)
