from typing import List, Dict, Tuple, Any
from retry import retry
from PIL import Image
from io import BytesIO

import json
import openai
import backends
import requests
import base64


logger = backends.get_logger(__name__)

MODEL_GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
MODEL_GPT_4_0613 = "gpt-4-0613"
MODEL_GPT_4_0314 = "gpt-4-0314"
MODEL_GPT_35_1106 = "gpt-3.5-turbo-1106"
MODEL_GPT_35_0613 = "gpt-3.5-turbo-0613"
MODEL_GPT_3 = "text-davinci-003"
MODEL_GPT_4_VISION_PREVIEW = "gpt-4-vision-preview"
SUPPORTED_MODELS = [MODEL_GPT_4_0314, MODEL_GPT_4_0613, MODEL_GPT_4_1106_PREVIEW, MODEL_GPT_35_1106, MODEL_GPT_35_0613, MODEL_GPT_3, MODEL_GPT_4_VISION_PREVIEW]

NAME = "openai"

MAX_TOKENS = 100   # 2024-01-10, das: Should this be hardcoded???

class OpenAI(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        if "organisation" in creds[NAME]:
            self.client = openai.OpenAI(
                api_key=creds[NAME]["api_key"],
                organization=creds[NAME]["organisation"]
                )
        else:
            self.client = openai.OpenAI(
                api_key=creds[NAME]["api_key"]
                )
        self.chat_models: List = ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0314", "gpt-4-0613", "gpt-4-1106-preview"]
        self.vision_models: List = ["gpt-4-vision-preview"]
        self.temperature: float = -1.
        self.vision_header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {creds[NAME]['api_key']}"
        }
        
        
    def encode_image(self, image_path):
        if image_path.startswith('http'):
            return True, image_path
        with open(image_path, "rb") as image_file:
            return False, base64.b64encode(image_file.read()).decode('utf-8')

    def list_models(self):
        models = self.client.models.list()
        names = [item.id for item in models.data]
        names = sorted(names)
        return names
        # [print(n) for n in names]   # 2024-01-10: what was this? a side effect-only method?

    @retry(tries=3, delay=0, logger=logger)
    def generate_response(self, messages: List[Dict], model: str) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: chat-gpt for chat-completion, otherwise text completion
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"
        if model in self.chat_models:
            # chat completion
            prompt = messages
            api_response = self.client.chat.completions.create(model=model,
                                                          messages=prompt,
                                                          temperature=self.temperature,
                                                          max_tokens=MAX_TOKENS)
            message = api_response.choices[0].message
            if message.role != "assistant":  # safety check
                raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
            response_text = message.content.strip()
            response = json.loads(api_response.json())
        
        elif model in self.vision_models:
            prompt = messages
            vision_messages = []
            for message in messages:
                this = {"role": message["role"], 
                        "content": [
                            {
                                "type": "text",
                                "text": message["content"]
                            }
                    ]}
                if "image" in message.keys():
                    url, loaded = self.encode_image(message["image"])
                    if url:
                        this["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": loaded
                            }
                        })
                    else:
                        this["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{loaded}"
                            }
                        })
                vision_messages.append(this)
#             payload = {
#                 "model": model,
#                 "messages": vision_messages,
#                 "max_tokens": MAX_TOKENS
#             }
#             response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.vision_header, json=payload)
#             logged_response = response.json()
#             print(logged_response)
            api_response = self.client.chat.completions.create(model=model,
                                                          messages=vision_messages,
                                                          temperature=self.temperature,
                                                          max_tokens=MAX_TOKENS)
            message = api_response.choices[0].message
            if message.role != "assistant":  # safety check
                raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
            response_text = message.content.strip()
            response = json.loads(api_response.json())

        else:  # default (text completion)
            prompt = "\n".join([message["content"] for message in messages])
            api_response = self.client.completions.create(model=model, prompt=prompt,
                                                     temperature=self.temperature, max_tokens=100)
            response = json.loads(api_response.json())
            response_text = api_response.choices[0].text.strip()
        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS
