# VLMInference
Unified Vision-Language Model Inference APIs

## 写在前面
为什么要写这个项目？很大程度上是因为目前主流的视觉语言模型提供的推理API很不一样，哪怕是同一个模型，纯文本、单张图片、多张图片的推理API也存在差异。在我实际工作中，常常要对不同模型进行测试以构建benchmark，不同API难以记住，来回查阅文档也是非常麻烦。因此，我决定创建本项目，将各家模型的推理API进行进一步封装，提供统一、最为便于使用的接口。

### 适用情景
- 纯文本、单图片、多图片推理
- 单样本（batch size = 1）
  
### 局限性
- 未统一实现batch inference接口，只能自行通过for循环+单样本推理实现，部分模型存在浪费显存的情况，推理速度慢
- 只适用于技术选型，在少样本上进行测试（1K），不能部署于生成环境
- 不同模型依赖不同，需要为不同模型创建不同的conda环境
  
### 硬件要求
在不考虑量化的情况下，torch.float16和torch.bfloat16是最为主流的推理精度。在模型大小为7B或8B的情况下，模型本身就需要占用14G或16G显存，建议使用16G以上的消费级显卡如RTX 3090、RTX 4090，在条件支持的情况下可以选择32G的V100，40G的A100或80G的A100。

本项目开发环境中主要有80G的A100和32G的V100，因此会优先支持这两块显卡。

## 模型支持
优先支持[司南大模型排行榜](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME)中开源、4-10B、位次前列的模型。

服务器无法直接连接huggingface的话，建议使用[huggingface镜像网站](https://hf-mirror.com/)或者[modelscope](https://www.modelscope.cn/home)下载权重。这里以linux系统为例，展示两种下载方式的脚本
```bash
# huggingface镜像下载，请自行修改model_name和local_dir
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download model_name --local-dir local_dir

# modelscope下载，请自行修改model_name和local_dir
pip install -U modelscope
modelscope download --model=model_name --local_dir local_dir
```
| 模型系列 | 模型大小 | 推理框架 | 显存要求 | 模型权重 | 环境配置 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| InternVL2 | 8B | LMDeploy | 16G+ | [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B) | [InternVL](https://internvl.readthedocs.io/en/latest/get_started/installation.html) + [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/installation.html) |
| MiniCPM-V-2.6 | 8B | vLLM | 16G+ | [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) | [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V?tab=readme-ov-file#install) + [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html) |

## 设计思路
1. 采取策略模式，设计抽象接口EvalInterface，包含抽象方法eval和其他常用方法比如加载图片。eval方法可以完成单样本情况下的纯文本、单张图片、多张图片的统一推理。其他常用方法由于继承关系，可以在子类中进行重写。
2. 为各家模型提供具体实现类，如将InternVL2模型封装为InternVL2ForEval，实现EvalInterface接口。
```python
from abc import ABC, abstractmethod

class EvalInterface(ABC):
    @abstractmethod
    def eval(self, query = None, imgs = None):
        pass
    
    def load_img(self, img_url):
        pass
```
