import copy
import gc
import requests
import torch

from io import BytesIO
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from ..inference import InferenceEngine

class LLavaOneVisionForInferBasic(InferenceEngine):
    def __init__(self, model_path = None, model_name = "llava_qwen", load_bits = 8, max_new_tokens = 512, top_p = 1.0, top_k = 1, temperature = 0.8, repetition_penalty = 1.0):
        assert model_path is not None, "Please provide model path"
        self.model_path = model_path
        if load_bits == 4:
            self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(model_path, None, model_name, load_4bit=True, device_map="auto")
        elif load_bits == 8:
            self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(model_path, None, model_name, load_8bit=True, device_map="auto")
        elif load_bits == 16:
            self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(model_path, None, model_name, device_map="auto")
        else:
            raise ValueError("Please provide valid load_bits.")
    
        self.model.eval()
        self.gen_config = {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }

    def load_image(self, img_url):
        if img_url.startswith("http://") or img_url.startswith("https://"):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36',
            }
            response = requests.get(img_url, headers=headers).content
            image_io = BytesIO(response)
            image = Image.open(image_io)
        else:
            image = Image.open(img_url)
        return image
    
    def parse_input(self, query = None, imgs = None):
        inputs = []

        if isinstance(imgs, str):
            imgs = [imgs]
        
        imgs = [self.load_image(img) for img in imgs]
        image_tensor = process_images(imgs, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]
        image_sizes = [image.size for image in imgs]

        question = f"\n{query}"
        for _ in imgs:
            question = DEFAULT_IMAGE_TOKEN + question
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        inputs['input_ids'] = input_ids
        inputs['images'] = image_tensor
        inputs['image_sizes'] = image_sizes
        return inputs


    def infer(self, query=None, imgs=None):
        inputs = self.parse_input(query, imgs)

        conversation = self.model.generate(
            **inputs,
            **self.gen_config
        )
        response = self.tokenizer.batch_decode(conversation, skip_special_tokens=True)
        return response[0]