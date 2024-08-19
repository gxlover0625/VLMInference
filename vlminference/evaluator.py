import requests

from abc import ABC, abstractmethod
from io import BytesIO
from PIL import Image

class EvalInterface(ABC):
    @abstractmethod
    def eval(self, query = None, imgs = None):
        pass

    def load_image(self, img_url, mode = "RGB"):
        if img_url.startswith("http://") or img_url.startswith("https://"):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36',
            }
            response = requests.get(img_url, headers=headers).content
            image_io = BytesIO(response)
            image = Image.open(image_io)
        else:
            image = Image.open(img_url)
            
        if mode == "RGB":
            image = image.convert('RGB')
        return image