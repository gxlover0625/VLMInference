import math
import requests
import torch
import torchvision.transforms as T

from io import BytesIO
from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from ..inference import InferenceEngine

class InternVL2ForInfer(InferenceEngine):
    def __init__(self, model_path = None, system_prompt = None, context_max_len = 8192, cache_max_entry_count = 0.8, max_new_tokens = 512, top_p = 1.0, top_k = 1, temperature = 0.8, repetition_penalty = 1.0):
        assert model_path is not None, "Please provide model path"
        if system_prompt is None:
            system_prompt = '我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'
        chat_template_config = ChatTemplateConfig('internvl-internlm2')
        chat_template_config.meta_instruction = system_prompt
        self.gen_config = GenerationConfig(max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, temperature=temperature, repetition_penalty=repetition_penalty)
        self.model = pipeline(model_path, chat_template_config=chat_template_config, backend_config=TurbomindEngineConfig(tp=torch.cuda.device_count(), session_len=context_max_len, cache_max_entry_count=cache_max_entry_count))
    
    def load_image(self, img_url):
        return load_image(img_url)
    
    def parse_input(self, query = None, imgs = None):
        if imgs is None:
            prompt = query
            return prompt
        else:
            if isinstance(imgs, list) and len(imgs) == 1:
                imgs = imgs[0]

            if isinstance(imgs, list):
                images = [self.load_image(img_url) for img_url in imgs]
                prompt_prefix = ""
                for i in range(len(images)):
                    prompt_prefix += f'Image-{i+1}: {IMAGE_TOKEN}\n'
                prompt = prompt_prefix + query
            else:
                images = self.load_image(imgs)
                prompt = query
            return (prompt, imgs)

    def infer(self, query = None, imgs = None):
        inputs = self.parse_input(query, imgs)
        response = self.model(inputs, gen_config = self.gen_config).text
        return response

    def batch_infer(self, query_list = None, imgs_list = None):
        inputs_list = [self.parse_input(query, imgs) for query, imgs in zip(query_list, imgs_list)]
        response_list = self.model(inputs_list, gen_config = self.gen_config)
        response_list = [response.text for response in response_list]
        return response_list


class InternVL2ForInferBasic(InferenceEngine):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def __init__(self, model_path = None, model_name = "InternVL2-8B", max_new_tokens = 512, top_p = 1.0, top_k = 1, temperature = 0.8, repetition_penalty = 1.0):
        assert model_path is not None, "Please provide model path"

        device_map = self.split_model(model_name)

        major, minor = torch.cuda.get_device_capability()
        compute_capability = major * 10 + minor
        if compute_capability < 80:
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.bfloat16

        self.gen_config = {
            "do_sample": True,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty
        }

        self.model = AutoModel.from_pretrained(model_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, trust_remote_code=True, device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def build_transform(self, input_size):
        MEAN, STD = InternVL2ForInferBasic.IMAGENET_MEAN, InternVL2ForInferBasic.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
            'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80
        }[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    def load_image(self, img_url, input_size=448, max_num=12):
        if img_url.startswith("http://") or img_url.startswith("https://"):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36',
            }
            response = requests.get(img_url, headers=headers).content
            image_io = BytesIO(response)
            image = Image.open(image_io)
        else:
            image = Image.open(img_url)
        image = image.convert('RGB')
        
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def parse_input(self, query = None, imgs = None):
        if imgs is None:
            prompt = query
            return prompt
        
        if isinstance(imgs, list) and len(imgs) == 1:
            imgs = imgs[0]
        
        if isinstance(imgs, str):
            images = self.load_image(imgs, max_num=12).to(self.torch_dtype).cuda()
            prompt_prefix = f'<image>\n'
            prompt = prompt_prefix + query
            return (prompt, images)
        else:
            images = [self.load_image(img_url, max_num=12).to(self.torch_dtype).cuda() for img_url in imgs]
            num_patches_list = [img.shape[0] for img in images]
            images = torch.cat(images, dim=0)
            
            prompt_prefix = ""
            for i in range(len(images)):
                prompt_prefix += f'Image-{i+1}: <image>\n'
            prompt = prompt_prefix + query
            return (prompt, images, num_patches_list)
    
    def infer(self, query = None, imgs = None):
        inputs = self.parse_input(query, imgs)
        if imgs is None:
            response, history = self.model.chat(self.tokenizer, None, inputs, self.gen_config, history=None, return_history=True)
        else:
            if len(inputs) == 3:
                response, history = self.model.chat(self.tokenizer, inputs[1], inputs[0], self.gen_config, num_patches_list=inputs[2], history=None, return_history=True)
            else:
                response, history = self.model.chat(self.tokenizer, inputs[1], inputs[0], self.gen_config, history=None, return_history=True)
        del inputs
        torch.cuda.empty_cache()
        return response
    
    def batch_infer(self, query_list = None, imgs_list = None):
        if all(imgs is None for imgs in imgs_list):
            raise NotImplementedError("batch_infer for query only is not implemented yet.")
        
        new_imgs_list = []
        for lst in imgs_list:
            if isinstance(lst, list) and len(lst) == 1:
                new_imgs_list.append(str(lst[0]))
            else:
                new_imgs_list.append(lst)
        imgs_list = new_imgs_list

        if all(isinstance(imgs, str) for imgs in imgs_list):
            inputs_list = [self.parse_input(query, imgs) for query, imgs in zip(query_list, imgs_list)]
            prompts = [prompt for (prompt, images) in inputs_list]
            images = [inputs[1] for inputs in inputs_list]
            num_patches_list = [img.size(0) for img in images]
            images = torch.cat(images, dim=0)
            response_list = self.model.batch_chat(self.tokenizer, images, prompts, num_patches_list=num_patches_list, generation_config=self.gen_config)
            
            del inputs_list, prompts, num_patches_list, images
            torch.cuda.empty_cache()
            return response_list
        else:
            raise NotImplementedError("batch_infer for multiple images is not implemented yet.")