# VLMInference
Unified Vision-Language Model Inference APIs

## 写在前面
为什么要写这个项目？很大程度上是因为目前主流的视觉语言模型提供的推理API很不一样，哪怕是同一个模型，纯文本、单张图片、多张图片的推理API也存在差异。在我实际工作中，常常要测试不同模型的效果以构建benchmark，众多API难以记住，来回查阅文档也是非常麻烦。因此，我决定创建本项目，将各家模型的推理API进行进一步封装，提供统一、最为方便使用的接口。

### 适用情景
- 纯文本、单图片、多图片+单轮对话推理
- 所有模型单样本推理（batch size = 1）
- 部分模型批量推理（batch size = N）
  
### 局限性
- 未统一实现批量推理接口，部分模型只能自行通过for循环+单样本推理实现全样本推理，推理速度较慢。通过LMDeploy、vLLM等部署框架实现的单样本推理接口存在浪费显存的情况
- 只适用于技术选型，在少样本上进行测试（1K），不能部署于生产环境
- 不同模型依赖版本不同，需要为不同模型创建不同的conda环境
  
### 硬件要求
在不考虑量化的情况下，torch.float16和torch.bfloat16是最为主流的推理精度。在模型大小为7B或8B的情况下，模型本身就需要占用14G或16G显存，建议使用显存16G以上的消费级显卡如RTX 3090、RTX 4090，在条件支持的情况下可以选择显存32G以上的V100、A100及以上。本项目开发环境中主要有80G的A100、32G的V100、24G的3090，因此会优先支持这三种显卡。

建议CUDA版本更新到12及以上，有些库如vLLM、LMDeploy要求CUDA版本最小为11.8。部分模型在安装Flash-Attention能大大提升推理速度，但对显卡架构有要求，V100无法安装Flash-Attention。

## :dart: 更新日志
[24/08/19] 更新了InternVL2-8B的单样本推理API。

[24/08/20] 更新了InternVL2-8B的批量推理API，支持混合输入，大大提升速度。

[24/08/21] 更新了不依赖LMDeploy的InternVL-8B的单样本以及批量推理API，但批量推理API只支持单张图片推理，也不支持混合输入。

[24/08/22] 更新了MiniCPM-V-2.6的单样本以及批量推理API。

[24/08/23] 更新了GLM-4V的单样本推理API，此版本推理需要去[智谱AI官网](https://open.bigmodel.cn/console/overview)获取access key，不需要下载模型，不需要显卡支持，直接通过http请求推理，官方并不支持多图片推理。

[24/09/01] InternVL2-8B更新了多GPU推理功能，解决单卡推理超出显存的问题，但会显著降低推理速度。解决数据类型bug、单次推理一直占用显存bug。

## 模型支持
优先支持[司南排行榜](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME)中开源、4-10B、位次前列的多模大模型。

如果服务器无法直接连接huggingface的话，建议使用[huggingface镜像网站](https://hf-mirror.com/)或者[modelscope](https://www.modelscope.cn/home)下载权重。这里以linux系统为例，展示两种下载方式的脚本
```bash
# huggingface镜像下载，请自行修改model_name和local_dir
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download model_name --local-dir local_dir

# modelscope下载，请自行修改model_name和local_dir
pip install -U modelscope
modelscope download --model=model_name --local_dir local_dir
```
| 模型系列 | 模型大小 | 推理框架 | 显存要求 | 模型权重 | 环境配置 | 单样本 | 批量 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| InternVL2 | 8B | LMDeploy<br/>Transformers | 16G+ | [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B) | [InternVL](https://internvl.readthedocs.io/en/latest/get_started/installation.html) + [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/installation.html) | 纯文本/单图片/多图片 | 纯文本/单图片/多图片/混合 |
| MiniCPM-V-2.6 | 8B | vLLM | 16G+ | [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) | [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V?tab=readme-ov-file#install) + [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html) | 纯文本/单图片/多图片 | 纯文本/单图片/多图片/混合 |
|GLM-4V | 9B | AccessKey | 28G+ | [GLM-4V](https://huggingface.co/THUDM/glm-4v-9b) | [GLM-4V](https://github.com/THUDM/GLM-4/blob/main/basic_demo/README.md) | 纯文本/单图片 | / |

## 快速开始
以InternVL2-8B为例，请根据`模型支持`这一小节配置好环境以及下载好权重
```bash
# 安装InternVL、LMDeploy环境
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVL
conda create -n internvl python=3.9 -y
conda activate internvl
pip install -r requirements.txt
pip install lmdeploy

# 下载模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download OpenGVLab/InternVL2-8B --local-dir ./weights/OpenGVLab/InternVL2-8B

# 克隆本项目
cd ../
git clone https://github.com/gxlover0625/VLMInference.git
cd VLMInference
```
下面是推理的代码
```python
from vlminference.models import InternVL2ForInfer

# 权重路径
model_path = ""
infer_engine = InternVL2ForInfer(model_path)

# 纯文本推理
print(infer_engine.infer(query = "你好"))
# >>>你好！请问有什么我可以帮助你的吗？

# 单张图片推理，传递url或者本地路径
print(infer_engine.infer(query = "请问图片描述了什么？", imgs = "url/path"))
print(infer_engine.infer(query = "请问图片描述了什么？", imgs = ["url/path"]))

# 多张图片推理
print(infer_engine.infer(query = "请问图片描述了什么？", imgs = ["url1/path1", "url2/path2"]))

# 批处理推理
print(infer_engine.infer(query_list = ["你好", "请问图片描述了什么？"], imgs_list = [None,"url2/path2"]))
```

## 设计思路
1. 采取策略模式，设计抽象接口InferenceEngine，包含infer等抽象方法和其他常用方法比如加载图片。infer方法可以完成单样本情况下的纯文本、单张图片、多张图片的统一推理。其他常用方法由于继承关系，可以在子类中进行重写实现自定义功能。
2. 为各家模型提供具体实现类，如将InternVL2模型封装为InternVL2ForInfer，实现InferenceEngine接口。
```python
from abc import ABC, abstractmethod

class InferenceEngine(ABC):
    @abstractmethod
    def parse_input(self, query = None, imgs = None):
        pass

    @abstractmethod
    def infer(self, query = None, imgs = None):
        pass
    
    def load_img(self, img_url):
        pass
```