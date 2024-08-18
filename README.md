# VLMInference
Unified Vision-Language Model Inference APIs

## 写在前面
为什么要写这个项目？很大程度上是因为目前主流的视觉语言模型提供的推理API很不一样，哪怕是同一个模型，纯文本、单张图片、多张图片的推理API也存在差异。在我实际工作中，常常要对不同模型进行测试以构建benchmark，不同API难以记住，来回查阅文档也是非常麻烦。因此，我决定创建本项目，将各家模型的推理API进行进一步封装，提供统一、最为便于使用的接口。

### 适用情景
- 纯文本、单图片、多图片推理
- 单样本（batch size = 1）
  
### 局限性
- 未统一实现batch inference接口，只能通过for循环+单样本推理，部分模型存在浪费显存的情况
- 少样本测试（1K），主要用于技术选型，不能部署于生成环境
- 不同模型依赖不同，需要为不同模型创建不同的conda环境
  
### 硬件要求
在不考虑量化的情况下，torch.float16和torch.bfloat16是最为主流的推理精度。在模型大小为7B或8B的情况下，模型本身就需要占用14G或16G显存，建议使用16G以上的消费级显卡如RTX 3090、RTX 4090，在条件支持的情况下可以选择32G的V100，40G的A100或80G的A100。

本项目开发环境中主要有80G的A100和V100，因此会优先支持这两块显卡。



## 设计思路
采取策略模式，设计抽象接口EvalInterface，包含eval方法。eval方法可以完成纯文本、单张图片、多张图片统一推理。为各家模型提供具体实现类，如InternVL2模型封装为InternVL2ForEval，实现EvalInterface接口。
```python
from abc import ABC, abstractmethod

class EvalInterface(ABC):
    @abstractmethod
    def eval(self, query = None, imgs = None):
        pass
```
