from zhipuai import ZhipuAI

from ..inference import InferenceEngine

class GLM4ForInferAK(InferenceEngine):
    def __init__(self, access_key = None):
        assert access_key is not None, "access_key is required"
        self.model = ZhipuAI(api_key=access_key)
        self.model_name = "glm-4v"

    def parse_input(self, query = None, imgs = None):
        if imgs is None:
            inputs = [
                {
                    "role": "user",
                    "content": query
                }
            ]
        else:
            if isinstance(imgs, list) and len(imgs) !=0:
                imgs = imgs[0]
            inputs = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": imgs
                            }
                        }
                    ]
                },
            ]
        return inputs
    
    def infer(self, query = None, imgs = None):
        inputs = self.parse_input(query, imgs)
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=inputs
        )
        return response.choices[0].message.content

    def batch_infer(self, query_list = None, imgs_list = None):
        raise NotImplementedError("batch inference is not implemented")
