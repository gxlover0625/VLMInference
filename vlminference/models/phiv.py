import gc
import requests
import torch

from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from ..inference import InferenceEngine

class PhiVForInferBasic(InferenceEngine):
    def __init__(self, model_path = None, max_new_tokens = 512, top_p = 1.0, top_k = 1, temperature = 0.8, repetition_penalty = 1.0):
        assert model_path is not None, "Please provide model path"
        self.model_path = model_path

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='flash_attention_2'    
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            num_crops=4
        )

        self.gen_config = {
            "do_sample": True,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty
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

    def parse_input(self, query=None, imgs=None):
        placeholder = ""
        images = []

        if imgs is None:
            imgs = []
        
        if isinstance(imgs, str):
            imgs = [imgs]
        
        for i, img_url in enumerate(imgs):
            images.append(self.load_image(img_url))
            placeholder += f"<|image_{i+1}|>\n"
        
        messages = [
            {"role": "user", "content": placeholder + query},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        if isinstance(images,list) and len(images)==0:
            images = None
        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda")
        return inputs

    def infer(self, query=None, imgs=None):
        inputs = self.parse_input(query, imgs)
        generate_ids = self.model.generate(
            **inputs, 
            **self.gen_config
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
    
    def batch_infer(self, query_list = None, imgs_list = None):
        raise NotImplementedError("Batch inference is not supported for this model")