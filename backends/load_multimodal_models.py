# Refactor 1 - Handle Multiple model loading (Blip2ForConditionalGeneration, LlavaForConditionalGeneration) - Done
# AutoModelForCausalLM does not work here

# Refactor 2 - Handle EOS/culling, Add a JSON to handle SUPPORTED_MODELS list and map
from transformers import (LlavaForConditionalGeneration,
                          Blip2ForConditionalGeneration, 
                          Blip2Processor,
                          AutoProcessor,
                          AutoTokenizer)

# Define Model IDs 
LLAVA_1_5_13B = "llava-1.5-13b-hf" 
LLAVA_1_5_7B = "llava-1.5-7b-hf"
BLIP_2_OPT_2_7B = "blip2-opt-2.7b"
LLAVA_1_6_7B_M = "llava-v1.6-mistral-7b"
LLAVA_1_6_13B_V = "llava-v1.6-vicuna-13b"
LLAVA_1_6_34B = "llava-v1.6-34b"
# VIP_LLAVA_7B = "vip-llava-7b-hf"

# Individual Model Loading Functions
def load_blip2_model(hf_id_str, cache_dir):
    hf_id_str = f"Salesforce/{hf_id_str}" 
    return Blip2ForConditionalGeneration.from_pretrained(hf_id_str, cache_dir=cache_dir, device_map="auto"), AutoProcessor.from_pretrained(hf_id_str, cache_dir = cache_dir, device_map = 'auto')

def load_llava_model(hf_id_str, cache_dir):
    hf_id_str = f"llava-hf/{hf_id_str}" 
    return LlavaForConditionalGeneration.from_pretrained(hf_id_str, cache_dir=cache_dir, device_map="auto"), AutoProcessor.from_pretrained(hf_id_str, cache_dir = cache_dir, device_map = 'auto')

# def load_vipllava_model(hf_id_str, cache_dir):
#     hf_id_str = f"llava-hf/{hf_id_str}" 
#     return VipLlavaForConditionalGeneration.from_pretrained(hf_id_str, cache_dir=cache_dir, device_map="auto"), AutoProcessor.from_pretrained(hf_id_str, cache_dir = cache_dir, device_map = 'auto')

def load_llava16_model(hf_id_str, cache_dir):
    hf_id_str = f"liuhaotian/{hf_id_str}"
    return LlavaForConditionalGeneration.from_pretrained(hf_id_str, cache_dir=cache_dir), AutoTokenizer.from_pretrained(hf_id_str, cache_dir = cache_dir)
 

# Map identifiers to loading functions
model_loaders = {
    LLAVA_1_5_13B: load_llava_model,
    LLAVA_1_5_7B: load_llava_model,
#     VIP_LLAVA_7B: load_vipllava_model,
    BLIP_2_OPT_2_7B: load_blip2_model,
    LLAVA_1_6_7B_M: load_llava16_model,
    LLAVA_1_6_13B_V: load_llava16_model,
    LLAVA_1_6_34B: load_llava16_model
}

# Main function to load a model based on identifier
def load_model(model_id, cache_dir):
    hf_id_str = model_id
    if model_id in model_loaders:
        return model_loaders[model_id](hf_id_str, cache_dir)
    else:
        raise ValueError(f"Model identifier {model_id} is not recognized.")