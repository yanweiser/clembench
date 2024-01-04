"""https://huggingface.co/llava-hf/llava-1.5-13b-hf"""

from typing import List, Dict, Tuple, Any, Optional
import torch
import backends
import transformers
import os
import copy
from PIL import Image

logger = backends.get_logger(__name__)

LLAVA_1_5_13b = "llava-1.5-13b-hf"


SUPPORTED_MODELS = [LLAVA_1_5_13b]

NAME = "llava1.5-hf"


class Llava15LocalHF(backends.Backend):
    def __init__(self):
        # load HF API key:
        creds = backends.load_credentials("huggingface")
        self.api_key = creds["huggingface"]["api_key"]

        self.temperature: float = -1.
        self.model_loaded: bool = False

    def load_model(self, model_name: str):
        assert model_name in SUPPORTED_MODELS, f"{model_name} is not supported, please make sure the model name is correct."
        logger.info(f'Start loading llava1.5-hf model: {model_name}')

        # model cache handling
        root_data_path = os.path.join(os.path.abspath(os.sep), "data")
        # check if root/data exists:
        if not os.path.isdir(root_data_path):
            logger.info(f"{root_data_path} does not exist, creating directory.")
            # create root/data:
            os.mkdir(root_data_path)
        CACHE_DIR = os.path.join(root_data_path, "huggingface_cache")

        # full HF model id string:
        hf_id_str = f"llava-hf/{model_name.capitalize()}"
        # load processor and model:
        self.processor = AutoProcessor.from_pretrained(hf_id_str, token=self.api_key, device_map="auto",
                                                       cache_dir=CACHE_DIR, verbose=False)
        self.model = LlavaForConditionalGeneration.from_pretrained(hf_id_str, token=self.api_key,
                                                          torch_dtype="auto", device_map="auto",
                                                          cache_dir=CACHE_DIR)
        # use CUDA if available:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_loaded = True

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
            logger.info(f"Finished loading llama2-hf model: {model}")
            logger.info(f"Model device map: {self.model.hf_device_map}")

        # greedy decoding:
        do_sample: bool = False
        if self.temperature > 0.0:
            do_sample = True

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
            if msg_idx > 0 and message['role'] == "user" and current_messages[msg_idx - 1]['role'] == "user":
                current_messages[msg_idx - 1]['content'] += f" {message['content']}"
                del current_messages[msg_idx]
            elif msg_idx > 0 and message['role'] == "assistant" and current_messages[msg_idx - 1]['role'] == "assistant":
                current_messages[msg_idx - 1]['content'] += f" {message['content']}"
                del current_messages[msg_idx]
        assert len(imgs) == 1, 'exactly one Image should be passed to the model'

        # load image
        raw_image = Image.open(imgs[0]) # to-do: should images be read via local path or imternet request?

        # prompt template
        # USER:
        # <image>\n<prompt> 
        # ASSISTANT:
        # <answer1>
        # USER:
        # <response1>
        # ASSISTANT:
        # ...

        # apply prompt template
        #   apply_chat_template is only for cllm tokenizers, so we do it by hand here

        if current_messages[0]['role'] == 'system':
            prompt_text = current_messages[0]['content']
            del current_messages[0]
        else:
            prompt_text = ""

        assert current_messages, "messages cannot only contain a system prompt"
        assert current_messages[0]['role'] == 'user', "You need to start dialogue on a User entry"

        prompt_text += f"USER:\n<image>\n{current_messages[0]['content']}\n\n"

        for msg in current_messages[1:]:
            if msg['role'] == 'user':
                prompt_text = f"USER:\n{msg['content']}\n\n"
            else:
                prompt_text = f"ASSISTANT:\n{msg['content']}\n\n"
        prompt_text += "ASSISTANT:\n"

        # apply processor to get model inputs
        inputs = self.processor(prompt_text, raw_image, return_tensors='pt').to(self.device)
        
        # to do: Image path should be added to inputs in some way for logging purposes
        prompt = {"inputs": prompt_text, "max_new_tokens": max_new_tokens,
                    "temperature": self.temperature}

        if do_sample:
            model_output_ids = self.model.generate(
                **inputs,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=top_p
            )
        else:
            model_output_ids = self.model.generate(
                **inputs,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens
            )

        model_output = self.tokenizer.batch_decode(model_output_ids)[0]

        response = {"response": model_output}

        # get output text
        response_text = self.processor.decode(model_output[0][2:], skip_special_tokens=True)

        # cull prompt from output:
        response_text = response_text.replace(prompt_text, "").strip()
        # remove EOS token at the end of output:
        if response_text[-4:len(response_text)] == "</s>":
            response_text = response_text[:-4]


        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
