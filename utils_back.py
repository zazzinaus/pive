# prendi question id e question strippato dopo "?" da dev e fai un json da passare a logicllama
import os
import torch
import re
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

# huggingface-cli login

# from huggingface_hub import login

# load_dotenv()
# login(token=os.getenv('HF_TOKEN')) # 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

#torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
translation_base_model = 'meta-llama/Llama-2-7b-chat-hf'
translation_peft_model = 'yuan-yang/LogicLLaMA-7b-direct-translate-delta-v0.1'

#inference_base_model = 'microsoft/Phi-3-mini-128k-instruct'

def parse_fol(llama_output):
    fol_translated = llama_output.split('### FOL:')[-1].strip()
    return fol_translated

def parse_inference(phi_output):
    return phi_output.split('### Answer')[-1].strip()

def get_template(task, template_filename, formatting_kwargs):
    base_template = open(f'./templates/{task}/{template_filename}.txt', 'r').read()
    template = base_template.format(**formatting_kwargs)
    return template


def load_vllm_model():
    model = LLM('LorMolf/LogicLlama2-chat-direct', dtype='auto') # 'meta-llama/Meta-Llama-3-8B-Instruct'
    return model

# def load_vllm_inference_model():
#     model = LLM(inference_base_model, dtype='auto', trust_remote_code=True)
# #     return model


def query_vllm_model(model, input_prompts, max_tokens=256):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens, n=1)
    output_text = model.generate(input_prompts, sampling_params=sampling_params)
    texts = [output.outputs[0].text.strip() for output in output_text]
    return texts


def load_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        #bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(inference_base_model)
    tokenizer.padding_side = "left"  # Allow batched inference
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')
    # model.eval()
    # model.to(device)
    return model, tokenizer


def query_model(tokenizer, model, input_prompts, max_tokens=256):
    input_ids = tokenizer(input_prompts, return_tensors='pt', padding=True).input_ids
    input_ids = input_ids.to(device)

    gen_config = GenerationConfig(
        do_sample= True,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens = max_tokens
    )

    output = model.generate(input_ids, generation_config=gen_config)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text

