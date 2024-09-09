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

[24/09/09] 更新了`Qwen2-VL`（transformers）的单样本、批量推理API，支持混合输入。

[24/09/01] 更新了`InternVL2`（transformes、lmdeploy）的多GPU推理功能，解决单卡推理超出显存bug、数据类型bug、单次推理一直占用显存bug。

[24/08/23] 更新了`GLM-4V`（api）的单样本推理API，需要去[智谱AI官网](https://open.bigmodel.cn/console/overview)获取access key，**不支持多图片推理**。

[24/08/22] 更新了`MiniCPM-V-2.6`（vllm）的单样本、批量推理API，支持混合输入。

[24/08/21] 更新了`InternVL2`（transformers）的单样本、批量推理API，但批量推理API只支持单张图片推理，也不支持混合输入。

[24/08/20] 更新了`InternVL2`（lmdeploy）的批量推理API，支持混合输入。

[24/08/19] 更新了`InternVL2`（lmdeploy）的单样本推理API。

## 模型支持
> [!IMPORTANT]
> 所有实现的模型在本项目的vlminference/models目录中，优先支持[司南排行榜](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME)中开源、10B以内、位次前列的多模大模型

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
| 模型系列 | 推理框架 | 具体实现类 | 环境配置 | 单样本 | 批量 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-8B) | LMDeploy | InternVL2ForInfer | [InternVL](https://internvl.readthedocs.io/en/latest/get_started/installation.html)<br> [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/installation.html) | :white_check_mark:纯文本<br> :white_check_mark:单图片<br> :white_check_mark:多图片 | :white_check_mark:纯文本<br> :white_check_mark:单图片<br> :white_check_mark:多图片<br> :white_check_mark:混合输 |
| | Transformers | InternVL2ForInferBasic | [InternVL](https://internvl.readthedocs.io/en/latest/get_started/installation.html) | :white_check_mark:纯文本<br> :white_check_mark:单图片<br> :white_check_mark:多图片 | :x:纯文本<br> :white_check_mark:单图片<br> :x:多图片<br> :x:混合输 |
| [MiniCPM-V-2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6) | vLLM | MiniCPMVForInfer | [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V?tab=readme-ov-file#install) <br> [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html) | :white_check_mark:纯文本<br> :white_check_mark:单图片<br> :white_check_mark:多图片 | :white_check_mark:纯文本<br> :white_check_mark:单图片<br> :white_check_mark:多图片<br> :white_check_mark:混合输 |
| [GLM-4V](https://open.bigmodel.cn/console/trialcenter?modelCode=glm-4v) | API | GLM4ForInferAK | [zhipuai](https://open.bigmodel.cn/dev/api#glm-4v) | :white_check_mark:纯文本<br> :white_check_mark:单图片<br> :x:多图片 | / |
| [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | Transformers | Qwen2VLForInferBasic | [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL?tab=readme-ov-file#quickstart) | :white_check_mark:纯文本<br> :white_check_mark:单图片<br> :white_check_mark:多图片 | :white_check_mark:纯文本<br> :white_check_mark:单图片<br> :white_check_mark:多图片<br> :white_check_mark:混合输 |

## 硬件要求
本项目优先支持24G以上的显卡，如3090、V100、A100等，其中V100不支持flash-attention 2，建议CUDA版本大于等于11.8。
> [!TIP]
> 如果运行代码时出现内核级错误，请查看是否正确安装pytorch版本。通过`nvcc --version`查看cuda版本，在pytorch官网选择对应的CUDA版本进行安装。

## 快速开始
以InternVL2-8B为例
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
```
下面是推理的代码
```python
import os
import sys
# 将本项目添加倒工作环境中！
sys.path.append("本项目路径")
# 根据vlminference/models/__init__.py中找到对应模型
os.environ['MODEL_ENV'] = 'internvl2'
from vlminference.models import InternVL2ForInfer

# 权重路径
model_path = "权重路径"
infer_engine = InternVL2ForInfer(model_path)

# 纯文本推理
print(infer_engine.infer(query = "你好"))
# >>>你好！请问有什么我可以帮助你的吗？

# 单张图片推理，传递url或者本地路径，可以用列表包围
print(infer_engine.infer(query = "请问图片描述了什么？", imgs = "url/path"))
print(infer_engine.infer(query = "请问图片描述了什么？", imgs = ["url/path"]))

# 多张图片推理，传递url或者本地路径，但不能嵌套列表
print(infer_engine.infer(query = "请问图片描述了什么？", imgs = ["url1/path1", "url2/path2"]))

# 批处理推理，如果是纯文本推理，imgs_list对应位置填上None，保持每一对query和imgs的格式满足单样本推理格式。
print(infer_engine.batch_infer(query_list = ["你好", "请问图片描述了什么？"], imgs_list = [None, "url2/path2"]))
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