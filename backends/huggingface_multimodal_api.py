"""
Backend using HuggingFace transformers for ungated multimodal models.
"""
from typing import List, Dict, Tuple, Any
import torch
import backends
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaNextForConditionalGeneration, LlavaNextProcessor
from jinja2 import Template

logger = backends.get_logger(__name__)

def load_processor(model_spec: backends.ModelSpec) -> AutoProcessor:
    '''
    Load processor for a specific model (Example - LlavaProcessor, Kosmos2Processor) 

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return processor: Processor for the specific model
    '''
    logger.info(f'Loading huggingface model Processor: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id'] # Get the model name

    # Load the processor 
    # NOTE - Further models may contain Tokenizer instead of Processor
    
    if hf_model_str in ["llava-hf/llava-v1.6-34b-hf"]:
        processor = LlavaNextProcessor.from_pretrained(hf_model_str, device_map="auto", verbose=False) 
    else:
        processor = AutoProcessor.from_pretrained(hf_model_str, device_map="auto", verbose=False)

    return processor


def load_model(model_spec: backends.ModelSpec) -> AutoModelForVision2Seq:
    '''
    Load a specific model 

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return model: The specific model
    '''

    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id'] # Get the model name

    if hf_model_str in ["llava-hf/llava-v1.6-34b-hf"]:
        model = LlavaNextForConditionalGeneration.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto")
    else:
        model = AutoModelForVision2Seq.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto")
    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    logger.info(f"Model device map: {model.hf_device_map}")
    
    return model

def load_image(image_file: str) -> Image:
    '''
    Load an image from a given link/directory

    :param image_file: A string that defines the link/directory of the image 
    :return image: Loaded image
    '''
    if image_file.startswith('http') or image_file.startswith('https'):
        image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def clean_messages(current_messages: List[Dict]) -> List[Dict]:
    '''
    Flatten double user messages

    :param current_messages: A list of messages passed to the model
    :return current_messages: Merged double user messages into one, keeping the 'image' key

    Example (Cloudgame)- 
    {'role': 'user', 'content': 'This seems correct.'} 
    {'role': 'user', 'content': 'Are there any chickens in the image? Answer with only "Yes" or "No".', 'image': 'games/cloudgame/resources/images/3.jpg'}
    
    {'role': 'user', 'content': 'This seems correct. Are there any chickens in the image? Answer with only "Yes" or "No".', 'image': 'games/cloudgame/resources/images/3.jpg'}
    '''

    for msg_idx, message in enumerate(current_messages):
        if msg_idx < len(current_messages)-1 and message['role'] == "user" and current_messages[msg_idx+1]['role'] == "user":
            # Merge into next message, ensuring 'image' key is not deleted
            current_messages[msg_idx+1]['content'] = f"{message['content']} " + current_messages[msg_idx+1]['content']
            del current_messages[msg_idx] 

    return current_messages

def get_images(prompt_text: str, messages: List[Dict], image_placeholder: str) -> List:
    '''
    Return loaded images from messages

    :param prompt_text: A string that goes into the input of the Processor
    :param messages: A list of messages passed to the model
    :return images: A list of loaded images, that can be directly passed as input to the Processor.
    '''

    # Count number of image placeholders (<image>, <img>, ...) in the cleaned prompt
    num_images = prompt_text.count(image_placeholder) 
    
    # Collect image links/file locations mentioned in messages
    imgs = []
    for _, message in enumerate(messages):
        if 'image' in message:
            imgs.append(message['image'])

    # Check if number of image placeholders and number of images passed are valid
    if len(imgs) != num_images:
        if len(imgs) == 1:
            # If only one image is available, copy it for num_images times. 
            # For games that have single image for all turns, but the game passes only one image in the message
            imgs *= num_images
        else:
            # If the number of images doesn't match. 
            # For games that have different image at each turn, number of image in message = number of image passed.  
            raise ValueError(f"Number of images ({len(imgs)}) does not match expected count ({num_images}).\nPlease check the messages and ensure there is an image link/location for each turn")

    # Load Images
    loaded_images = [load_image(m) for m in imgs]

    return loaded_images


class HuggingfaceMultimodal(backends.Backend):
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return HuggingfaceMultimodalModel(model_spec)


class HuggingfaceMultimodalModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = load_processor(model_spec)
        self.multimodal_model = load_model(model_spec).to(self.device)
        self.template = model_spec["custom_chat_template"]
        self.assistant_tag = model_spec["assistant"]
        self.image_placeholder = model_spec["placeholder"]

    def generate_response(self, messages: List[Dict],
                          log_messages: bool = False) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "user", "content": "Are there any clouds in the image? Answer with only "Yes" or "No"."},
                    {"role": "assistant", "content": "Yes"},
                    {"role": "user", "content": "This seems correct."},
                    {'role': 'user', 'content': 'Are there any chickens in the image? Answer with only "Yes" or "No".', 'image': 'games/cloudgame/resources/images/3.jpg'}
                ]
        :param model: model name
        :param log_messages: If True, raw and cleaned messages passed will be logged.
        :return: the continuation
        """

        # log current given messages list:
        if log_messages:
            logger.info(f"Raw messages passed: {messages}")

        # Cleanup double user messages
        cleaned_messages = clean_messages(messages)

        # Get prompt by applying jinja template
        template_str = self.template
        template = Template(template_str)
        prompt_text = template.render(messages=cleaned_messages)
        
        # Replace image placeholder with the correct one for the model
        prompt_text = prompt_text.replace('<image>', self.image_placeholder)

        # Get a list of images that will be passed to the Processor
        images = get_images(prompt_text, messages, self.image_placeholder)

        # Store prompt_text
        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(), "temprature": self.get_temperature()}
        
        # Generate the output
        if not images:
            inputs = self.processor(prompt_text, images=[Image.new("RGB", (300,300))], padding=True, return_tensors="pt").to("cuda")
            inputs["pixel_values"] = None
        else:
            inputs = self.processor(prompt_text, images=images, padding=True, return_tensors="pt").to("cuda")
        model_output = self.multimodal_model.generate(**inputs, max_new_tokens=self.get_max_tokens())
        generated_text = self.processor.batch_decode(model_output, skip_special_tokens=True)

        # Store generated text
        response = {'response': generated_text}

        for text in generated_text:
            response_text = text.split(self.assistant_tag + ":")[-1] # Get the last assistant response

        return prompt, response, response_text