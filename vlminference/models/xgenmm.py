import gc
import requests
import torch

from io import BytesIO
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor

from ..inference import InferenceEngine

class XGenMMForInferBasic(InferenceEngine):
    def __init__(self, model_path = None, max_new_tokens = 512, top_p = None, top_k = None, temperature = 0.05, repetition_penalty = 1.0):
        assert model_path is not None, "Please provide model path"
        
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True).eval().cuda()
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, legacy=False)
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.eos_token = '<|end|>'
    
        self.gen_config = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
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
    
    def apply_prompt_template(self, query):
        s = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n{query}<|end|>\n<|assistant|>\n'
            )
        return s

    def parse_input(self, query=None, imgs=None):
        inputs = {}
        if imgs is None:
            inputs['pixel_values'] = None
            inputs['image_sizes'] = None
        else:
            image_list = []
            image_sizes = []

            if type(imgs) is str:
                imgs = [imgs]
            
            for img_url in imgs:
                image = self.load_image(img_url)
                image_list.append(self.image_processor([image], image_aspect_ratio='anyres')["pixel_values"])
                image_sizes.append(image.size)
            inputs['pixel_values'] = [image_list]
            inputs['image_sizes'] = [image_sizes]

        prompt = self.apply_prompt_template(query)
        language_inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[name] = value.cuda()
        return inputs
    
    def infer(self, query=None, imgs=None):
        inputs = self.parse_input(query, imgs)
        image_sizes = inputs.pop('image_sizes')
        generated_text = self.model.generate(**inputs, image_size=image_sizes, **self.gen_config)
        if imgs is None:
            prompt_len = len(inputs['input_ids'][0])
            response = self.tokenizer.decode(generated_text[0][prompt_len:], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
        return response
    
    def batch_infer(self, query_list=None, imgs_list=None):
        if any(imgs is None for imgs in imgs_list):
            raise NotImplementedError("Batch inference without images is not supported yet.")

        batch_images = []
        batch_image_sizes = []
        batch_prompt = []
        for query, imgs in zip(query_list, imgs_list):
            image_list = []
            image_sizes = []

            if type(imgs) is str:
                imgs = [imgs]
            
            for img_url in imgs:
                image = self.load_image(img_url)
                image_list.append(self.image_processor([image], image_aspect_ratio='anyres')["pixel_values"].cuda())
                image_sizes.append(image.size)
            
            batch_images.append(image_list)
            batch_image_sizes.append(image_sizes)
            batch_prompt.append(query)
        
        batch_inputs = {
            "pixel_values": batch_images
        }

        query_inputs = [self.apply_prompt_template(query) for query in batch_prompt]
        language_inputs = self.tokenizer(query_inputs, return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True)
        language_inputs = {name: tensor.cuda() for name, tensor in language_inputs.items()}
        batch_inputs.update(language_inputs)

        generated_text = self.model.generate(**batch_inputs, image_size=batch_image_sizes, **self.gen_config)
        response_list = self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)
        del batch_inputs
        torch.cuda.empty_cache()
        gc.collect()
        return response_list
            
            