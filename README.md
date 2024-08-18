# VLMInference
Unified Vision-Language Model Inference APIs

## 写在前面
为什么要写这个项目？很大程度上是因为目前主流的视觉语言模型提供的推理API很不一样，哪怕是同一个模型，纯文本、单张图片、多张图片的推理API也存在差异。在我实际工作中，常常要对不同模型进行测试以构建benchmark，不同API难以记住，来回查阅文档也是非常麻烦。因此，我决定创建本项目，将各家模型的推理API进行进一步封装，提供统一、最为便于使用的接口。

适用情景：
- 纯文本、单图片、多图片推理
- 单样本的简单测试

## 设计思路
采取策略模式，设计抽象接口EvalInterface，包含eval方法。eval方法可以完成纯文本、单张图片、多张图片统一推理。为各家模型提供具体实现类，如InternVL2模型封装为InternVL2ForEval，实现EvalInterface接口。
```python
from abc import ABC, abstractmethod

class EvalInterface(ABC):
    @abstractmethod
    def eval(self, query = None, imgs = None):
        pass
```
