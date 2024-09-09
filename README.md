# VLMInference
Unified Vision-Language Model Inference APIs

## 写在前面
为什么要写这个项目？
1. 不同视觉语言模型的推理API很不一样，来回查阅文档非常麻烦
2. 对于新手或者是组内合作，统一且简单的API可以大幅提高效率，减少重复劳动

因此，我决定创建本项目，将各家模型的推理API进行进一步封装，提供统一、最为方便使用的接口。

### 适用情景
- 纯文本、单图片、多图片+单轮对话推理
- 所有模型单样本推理（batch size = 1）
- 部分模型批量推理（batch size = N），可以支持json格式的输入输出
  
### 局限性
- 不适用于多轮对话推理，不适合部署于生产环境
- 部分模型的依赖较难安装，如vllm、flash-attention
  
## :dart: 更新日志
括号内表示推理框架，transformers代表官网原生代码

[24/09/09] 更新了`Qwen2-VL`（transformers）的单样本以及批量推理API。

[24/09/01] 更新了InternVL2-8B（原生代码以及lmdeploy）的多GPU推理功能，解决单卡推理超出显存的问题，但会显著降低推理速度。解决数据类型bug、单次推理一直占用显存bug。

[24/08/23] 更新了GLM-4V（api）的单样本推理API，此版本推理需要去[智谱AI官网](https://open.bigmodel.cn/console/overview)获取access key，不需要下载模型，不需要显卡支持，直接通过http请求推理，官方并不支持多图片推理。

[24/08/22] 更新了MiniCPM-V-2.6（vllm）的单样本以及批量推理API，支持混合输入。

[24/08/21] 更新了InternVL2-8B（transformers）的单样本以及批量推理API，但批量推理API只支持单张图片推理，也不支持混合输入。

[24/08/20] 更新了InternVL2-8B（lmdeploy）的批量推理API，支持混合输入。

[24/08/19] 更新了InternVL2-8B（lmdeploy）的单样本推理API。

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

## 硬件要求
本项目优先支持24G以上的显卡，如3090、V100、A100等。

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