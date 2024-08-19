from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

from ..evaluator import EvalInterface

class InternVL2ForEval(EvalInterface):
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

    def eval(self, query = None, imgs = None):
        # 无图片
        if imgs is None:
            response = self.model.chat(query, gen_config=self.gen_config).response.text

        # 有图片
        else:
            if isinstance(imgs, list):
                images = [self.load_image(img_url) for img_url in imgs]
                prompt_prefix = ""
                for i in range(len(images)):
                    prompt_prefix += f'Image-{i+1}: {IMAGE_TOKEN}\n'
                prompt = prompt_prefix + query
            else:
                images = self.load_image(imgs)
                prompt = query

            response = self.model.chat((prompt, images), gen_config=self.gen_config).response.text

        if len(response) == 0:
            raise Exception("Out of context max tokens, please try to shorten the context")
        return response