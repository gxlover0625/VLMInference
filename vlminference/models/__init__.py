import os

# 获取环境变量
model_env = os.getenv('MODEL_ENV')

# 根据环境变量导入相应的模型
if model_env == 'internvl2':
    from .internvl2 import InternVL2ForInfer, InternVL2ForInferBasic
    __all__ = [
        "InternVL2ForInfer",
        "InternVL2ForInferBasic",
    ]
elif model_env == 'minicpmv':
    from .minicpmv import MiniCPMVForInfer
    __all__ = [
        "MiniCPMVForInfer",
    ]
elif model_env == "glm4v":
    from .glm4v import GLM4ForInferAK
    __all__ = [
        "GLM4ForInferAK",
    ]
elif model_env == "qwen2vl":
    from .qwen2vl import Qwen2VLForInferBasic
    __all__ = [
        "Qwen2VLForInferBasic",
    ]
elif model_env == "xgenmm":
    from .xgenmm import XGenMMForInferBasic
    __all__ = [
        "XGenMMForInferBasic",
    ]
elif model_env == "phiv":
    from .phiv import PhiVForInferBasic
    __all__ = [
        "PhiVForInferBasic",
    ]
elif model_env == "llava_onevision":
    from .llava_onevision import LLavaOneVisionForInferBasic
    __all__ = [
        "LLavaOneVisionForInferBasic",
    ]
else:
    raise ValueError(f"Unknown MODEL_ENV: {model_env}")
