from typing import List, Dict, Tuple, Any, Optional
import torch
import backends
import transformers
import os
import copy
from PIL import Image
import requests
from io import BytesIO


from transformers import (LlavaForConditionalGeneration, 
                          Blip2ForConditionalGeneration, 
                          Blip2Processor,
                          AutoProcessor,
                          AutoModelForCausalLM)
from templates import apply_qa_template, apply_ua_template

from backends.load_multimodal_models import load_model

logger = backends.get_logger(__name__)

LLAVA_1_5_13B = "llava-1.5-13b-hf" 
LLAVA_1_5_7B = "llava-1.5-7b-hf"
BLIP_2_OPT_2_7B = "blip2-opt-2.7b"
LLAVA_1_6_7B_M = "llava-v1.6-mistral-7b"
LLAVA_1_6_13B_V = "llava-v1.6-vicuna-13b"
LLAVA_1_6_34B = "llava-v1.6-34b"
QWEN_VL_CHAT = "Qwen-VL-Chat"
VIP_LLAVA_7B = "vip-llava-7b-hf"

SUPPORTED_MODELS = [LLAVA_1_5_7B, LLAVA_1_5_13B, BLIP_2_OPT_2_7B, LLAVA_1_6_7B_M, LLAVA_1_6_13B_V, LLAVA_1_6_34B, QWEN_VL_CHAT, VIP_LLAVA_7B]


class MultimodalLocal(backends.Backend):
    def __init__(self):
        # load HF API key:
        creds = backends.load_credentials("huggingface")
        self.api_key = creds["huggingface"]["api_key"]

        self.temperature: float = -1.
        self.model_loaded: bool = False

    def load_model(self, model_name: str):
        assert model_name in SUPPORTED_MODELS, f"{model_name} is not supported, please make sure the model name is correct."
        logger.info(f'Start loading hf model: {model_name}')

        # model cache handling
        root_data_path = os.path.join(os.path.abspath(os.sep), "data")
        # check if root/data exists:
        if not os.path.isdir(root_data_path):
            logger.info(f"{root_data_path} does not exist, creating directory.")
            # create root/data:
            os.mkdir(root_data_path)
        CACHE_DIR = os.path.join(root_data_path, "huggingface_cache")
        OFFLOAD_DIR = os.path.join(root_data_path, "offload")

        # Load model and processor
        self.model, self.processor = load_model(model_name, CACHE_DIR)
  
        # use CUDA if available:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_loaded = True

    def load_image(self, image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def generate_response(self, messages: List[Dict], model: str,
                          max_new_tokens: Optional[int] = 100, top_p: float = 0.9) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name, chat models for chat-completion, otherwise text completion
        :param max_new_tokens: Maximum generation length.
        :param top_p: Top-P sampling parameter. Only applies when do_sample=True.
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"
        assert messages, "Messages passed for generation cannot be empty"

        # load the model to the memory
        if not self.model_loaded:
            self.load_model(model)
            logger.info(f"Finished loading hf model: {model}")
            logger.info(f"Model device map: {self.model.hf_device_map}")

        # greedy decoding:
        if self.temperature <= 0.0:
            self.temperature = 0.001

        # turn off redundant transformers warnings:
        transformers.logging.set_verbosity_error()

        # deepcopy messages to prevent reference issues:
        current_messages = copy.deepcopy(messages)

        # flatten consecutive user messages and extract image from message struct:
        imgs = []
        for msg_idx, message in enumerate(current_messages):
            if 'image' in message:
                imgs.append(message['image'])
                del current_messages[msg_idx]['image']

        # load image
        raw_image = self.load_image(imgs[0])

        if self.model_name in [BLIP_2_OPT_2_7B]:
            prompt_text = apply_ua_template(current_messages)
            # prompt_text = "Question: What is in the image Answer: " # Add a test question to check if model can 'see' the image
            inputs = self.processor(raw_image, prompt_text, return_tensors='pt').to(self.device)
        else:
            prompt_text = apply_ua_template(current_messages) 
            #Different order of text and image
            inputs = self.processor(prompt_text, raw_image, return_tensors='pt').to(self.device)
        
        prompt = {"inputs": current_messages, "max_new_tokens": max_new_tokens,
                    "temperature": self.temperature, "image": imgs[0]}

        output_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True
        ).to(self.device)
            
        decoded_output = self.processor.decode(output_ids[0][2:], skip_special_tokens=True)
        decoded_output = decoded_output.strip()
        response = {"response": decoded_output}
        
        response_text = decoded_output

        # print("Response Text: ")
        # print(response_text)
        # cull prompt from output, Only for LLAVA1.5, BLIP2 has "" empty response
        if not self.model_name in [BLIP_2_OPT_2_7B]:
            response_text = decoded_output.split("ASSISTANT:")[-1].strip()

        # remove EOS token at the end of output:
        if response_text[-4:len(response_text)] == "</s>":
            response_text = response_text[:-4]

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS