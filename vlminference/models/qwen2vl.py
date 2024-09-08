import gc
import torch

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from ..inference import InferenceEngine

class Qwen2VLForInferBasic(InferenceEngine):
    def __init__(self, model_path = None, max_new_tokens = 512, top_p = 1.0, top_k = 1, temperature = 0.8, repetition_penalty = 1.0):
        assert model_path is not None, "Please provide model path"
        
        major, minor = torch.cuda.get_device_capability()
        compute_capability = major * 10 + minor
        if compute_capability < 80:
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.bfloat16
        
        # try:
        #     import flash_attn
        #     print('use flash attention 2 to accelerate inference.')
        #     self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #         model_path,
        #         torch_dtype=torch.bfloat16,
        #         attn_implementation="flash_attention_2",
        #         device_map="auto",
        #     )
        # except Exception as e:
        #     print("flash attention 2 is not installed, use the default mode.")
        #     self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #         model_path, torch_dtype="auto", device_map="auto"
        #     )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.gen_config = {
            "do_sample": True,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty
        }
    
    def parse_input(self, query=None, imgs=None):
        if imgs is None:
            messages = [
                {"role": "user", "content": query},
            ]
            return messages
        
        if isinstance(imgs, str):
            imgs = [imgs]
        
        content = []
        for img in imgs:
            content.append({
                "type": "image",
                "image": img
            })
        content.append({
            "type": "text",
            "text": query
        })

        messages = [
            {"role": "user", "content": content},
        ]
        return messages

    def infer(self, query=None, imgs=None):
        messages = self.parse_input(query, imgs)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        return response[0]

    def batch_infer(self, query_list=None, imgs_list=None):
        messages_list = [self.parse_input(query, imgs) for query, imgs in zip(query_list, imgs_list)]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True)
            for msg in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.torch_dtype).to("cuda")

        generated_ids = self.model.generate(**inputs, **self.gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_list = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        return response_list
