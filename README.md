# VLMInference
Unified Vision-Language Model Inference APIs

## 写在前面
为什么要写这个项目？很大程度上是因为目前主流的视觉语言模型提供的推理API很不一样，哪怕是同一个模型，纯文本、单张图片、多张图片的推理API也存在差异。在我实际工作中，常常要对不同模型进行测试以构建benchmark，不同API难以记住，来回查阅文档也是非常麻烦。因此，我决定创建本项目，将各家模型的推理API进行进一步封装，提供统一、便于使用的接口。
