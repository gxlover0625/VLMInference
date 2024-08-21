from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

from ..inference import InferenceEngine

class InternVL2ForInfer(InferenceEngine):
    def __init__(self, model_path = None, system_prompt = None, context_max_len = 8192, max_new_tokens = 512, top_p = 1.0, top_k = 1, temperature = 0.8, repetition_penalty = 1.0):
        assert model_path is not None, "Please provide model path"
        if system_prompt is None:
            system_prompt = '我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'
        chat_template_config = ChatTemplateConfig('internvl-internlm2')
        chat_template_config.meta_instruction = system_prompt
        self.gen_config = GenerationConfig(max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, temperature=temperature, repetition_penalty=repetition_penalty)
        self.model = pipeline(model_path, chat_template_config=chat_template_config, backend_config=TurbomindEngineConfig(session_len=context_max_len))
    
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