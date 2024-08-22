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
else:
    raise ValueError(f"Unknown MODEL_ENV: {model_env}")
