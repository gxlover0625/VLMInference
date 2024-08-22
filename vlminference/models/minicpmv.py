from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..inference import InferenceEngine

class MiniCPMVForInfer(InferenceEngine):
    def __init__(self, model_path = None, context_max_len = 2048, max_new_tokens = 512, top_p = 1.0, top_k = -1, temperature = 0, repetition_penalty = 1.0):
        assert model_path is not None, "Please provide model path"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=1, max_model_len=context_max_len)
        
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

        self.sampling_params = SamplingParams(
            stop_token_ids=stop_token_ids, 
            use_beam_search=True,
            temperature=temperature, 
            best_of=3,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens
        )
    
    def parse_input(self, query=None, imgs=None):
        if imgs is None:
            messages = [{
                "role": "user",
                "content": query
            }]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = {
                "prompt": prompt,
            }
        else:
            if isinstance(imgs, list):
                images = [self.load_image(img_url) for img_url in imgs]
                prompt_prefix = ""
                for i in range(len(images)):
                    prompt_prefix += f'(<image>./</image>)\n'
                messages = [{
                    "role": "user",
                    "content": prompt_prefix + query
                }]
            else:
                images = self.load_image(imgs)
                prompt_prefix = f'(<image>./</image>)\n'
                messages = [{
                    "role": "user",
                    "content": prompt_prefix + query
                }]

            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = {
                "prompt": prompt,
                "multi_modal_data":{
                    "image": images
                }
            }

        return inputs
    
    def infer(self, query = None, imgs=None):
        inputs = self.parse_input(query, imgs)
        outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
        response = outputs[0].outputs[0].text
        return response

    def batch_infer(self, query_list = None, imgs_list = None):
        inputs_list = [self.parse_input(query, imgs) for query, imgs in zip(query_list, imgs_list)]
        response_list = self.model.generate(inputs_list, sampling_params=self.sampling_params)
        response_list = [response.outputs[0].text for response in response_list]
        return response_list


