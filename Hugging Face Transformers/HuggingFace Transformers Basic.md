# 1. Tansformers基础知识

**常见自然语言处理任务：**

- 情感分析
- 文本生成
- 命名实体识别
- 阅读理解
- 掩码填充
- 文本摘要
- 机器翻译
- 特征提取
- 对话机器人



> **Transformers及相关库：**
>
> - Transformers：核心库，模型加载、模型训练、流水线等。
> - Tokenizer：分词器，对数据进行预处理，文本到token序列的互相转换
> - Datasets：数据集库，提供了数据集的加载、处理等方法
> - Evaluate：评估函数，提供各种评价指标的计算函数
> - PEFT：高效微调模型的库，提供了几种高效微调的方法，小参数量撬动大模型
> - Accelerate：分布式训练，提供了分布式训练解决方案，包括大模型的加载与推理解决方案
> - Optimum：优化加速库，支持各种后端，如Onnxruntime、OpenVino等
> - Gradio：可视化部署库，几行代码快速实现基于Web交互的算法演示系统

## 1.1 Transformers环境安装

### 1.1.1 Pytorch安装

- PyTorch：[官方网址](https://pytorch.org/)

  ![image-20231128212038273](./HuggingFace%20Transformers%20Basic.assets/image-20231128212038273.png)

- Cuda：

  - 如果只需要训练、简单推理，则无需单独安装cuda，直接安装pytorch
  - 如果有部署需求，例如导出TensorRT模型，则需要进行cuda安装

### 1.1.2 VS Code安装



## 1.2 代码示例

### 1.2.1 代码示例1

```python
import gradio as gr
from transformers import *

# 通过Interface加载pipeline并启动文本分类服务
gr.Interface.from_pipeline(pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")).launch()
```

> ```asciiarmor
> d:\work_tools\Anaconda\program\envs\my_transformers\lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
> from .autonotebook import tqdm as notebook_tqdm
> d:\work_tools\Anaconda\program\envs\my_transformers\lib\site-packages\transformers\deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
> warnings.warn(
> d:\work_tools\Anaconda\program\envs\my_transformers\lib\site-packages\transformers\generation_utils.py:24: FutureWarning: Importing `GenerationMixin` from `src/transformers/generation_utils.py` is deprecated and will be removed in Transformers v5. Import as `from transformers import GenerationMixin` instead.
> warnings.warn(
> d:\work_tools\Anaconda\program\envs\my_transformers\lib\site-packages\torchaudio\backend\utils.py:62: UserWarning: No audio backend is available.
> warnings.warn("No audio backend is available.")
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-finetuned-dianping-chinese",
> "architectures": [
>  "BertForSequenceClassification"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "id2label": {
>  "0": "negative (stars 1, 2 and 3)",
>  "1": "positive (stars 4 and 5)"
> },
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "label2id": {
>  "negative (stars 1, 2 and 3)": 0,
>  "positive (stars 4 and 5)": 1
> },
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> 
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-finetuned-dianping-chinese",
> "architectures": [
>  "BertForSequenceClassification"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "id2label": {
>  "0": "negative (stars 1, 2 and 3)",
>  "1": "positive (stars 4 and 5)"
> },
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "label2id": {
>  "negative (stars 1, 2 and 3)": 0,
>  "positive (stars 4 and 5)": 1
> },
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> 
> loading weights file pytorch_model.bin from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071\pytorch_model.bin
> All model checkpoint weights were used when initializing BertForSequenceClassification.
> 
> All the weights of BertForSequenceClassification were initialized from the model checkpoint at uer/roberta-base-finetuned-dianping-chinese.
> If your task is similar to the task the model of the checkpoint was trained on, you can already use BertForSequenceClassification for predictions without further training.
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-finetuned-dianping-chinese",
> "architectures": [
>  "BertForSequenceClassification"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "id2label": {
>  "0": "negative (stars 1, 2 and 3)",
>  "1": "positive (stars 4 and 5)"
> },
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "label2id": {
>  "negative (stars 1, 2 and 3)": 0,
>  "positive (stars 4 and 5)": 1
> },
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> 
> loading file vocab.txt from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071\vocab.txt
> loading file tokenizer.json from cache at None
> loading file added_tokens.json from cache at None
> loading file special_tokens_map.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071\special_tokens_map.json
> loading file tokenizer_config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071\tokenizer_config.json
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-finetuned-dianping-chinese",
> "architectures": [
>  "BertForSequenceClassification"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "id2label": {
>  "0": "negative (stars 1, 2 and 3)",
>  "1": "positive (stars 4 and 5)"
> },
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "label2id": {
>  "negative (stars 1, 2 and 3)": 0,
>  "positive (stars 4 and 5)": 1
> },
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> 
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-finetuned-dianping-chinese\snapshots\25faf1874b21e76db31ea9c396ccf2a0322e0071\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-finetuned-dianping-chinese",
> "architectures": [
>  "BertForSequenceClassification"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "id2label": {
>  "0": "negative (stars 1, 2 and 3)",
>  "1": "positive (stars 4 and 5)"
> },
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "label2id": {
>  "negative (stars 1, 2 and 3)": 0,
>  "positive (stars 4 and 5)": 1
> },
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> ```
>
> ```
> Running on local URL:  http://127.0.0.1:7860
> 
> To create a public link, set `share=True` in `launch()`.
> ```

![image-20231129095704400](./HuggingFace%20Transformers%20Basic.assets/image-20231129095704400.png)

### 1.2.2 代码示例2

```python
import gradio as gr
from transformers import *

# 通过Interface加载pipeline并启动阅读理解服务
gr.Interface.from_pipeline(pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")).launch()
```

> ```asciiarmor
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-chinese-extractive-qa\snapshots\9b02143727b9c4655d18b43a69fc39d5eb3ddd53\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-chinese-extractive-qa",
> "architectures": [
>  "BertForQuestionAnswering"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "gradient_checkpointing": false,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> 
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-chinese-extractive-qa\snapshots\9b02143727b9c4655d18b43a69fc39d5eb3ddd53\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-chinese-extractive-qa",
> "architectures": [
>  "BertForQuestionAnswering"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "gradient_checkpointing": false,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> 
> loading weights file pytorch_model.bin from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-chinese-extractive-qa\snapshots\9b02143727b9c4655d18b43a69fc39d5eb3ddd53\pytorch_model.bin
> All model checkpoint weights were used when initializing BertForQuestionAnswering.
> 
> All the weights of BertForQuestionAnswering were initialized from the model checkpoint at uer/roberta-base-chinese-extractive-qa.
> If your task is similar to the task the model of the checkpoint was trained on, you can already use BertForQuestionAnswering for predictions without further training.
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-chinese-extractive-qa\snapshots\9b02143727b9c4655d18b43a69fc39d5eb3ddd53\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-chinese-extractive-qa",
> "architectures": [
>  "BertForQuestionAnswering"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "gradient_checkpointing": false,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> 
> loading file vocab.txt from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-chinese-extractive-qa\snapshots\9b02143727b9c4655d18b43a69fc39d5eb3ddd53\vocab.txt
> loading file tokenizer.json from cache at None
> loading file added_tokens.json from cache at None
> loading file special_tokens_map.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-chinese-extractive-qa\snapshots\9b02143727b9c4655d18b43a69fc39d5eb3ddd53\special_tokens_map.json
> loading file tokenizer_config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-chinese-extractive-qa\snapshots\9b02143727b9c4655d18b43a69fc39d5eb3ddd53\tokenizer_config.json
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-chinese-extractive-qa\snapshots\9b02143727b9c4655d18b43a69fc39d5eb3ddd53\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-chinese-extractive-qa",
> "architectures": [
>  "BertForQuestionAnswering"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "gradient_checkpointing": false,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> 
> loading configuration file config.json from cache at C:\Users\DueFireTop-NUC/.cache\huggingface\hub\models--uer--roberta-base-chinese-extractive-qa\snapshots\9b02143727b9c4655d18b43a69fc39d5eb3ddd53\config.json
> Model config BertConfig {
> "_name_or_path": "uer/roberta-base-chinese-extractive-qa",
> "architectures": [
>  "BertForQuestionAnswering"
> ],
> "attention_probs_dropout_prob": 0.1,
> "classifier_dropout": null,
> "gradient_checkpointing": false,
> "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
> "hidden_size": 768,
> "initializer_range": 0.02,
> "intermediate_size": 3072,
> "layer_norm_eps": 1e-12,
> "max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
> "num_hidden_layers": 12,
> "pad_token_id": 0,
> "position_embedding_type": "absolute",
> "transformers_version": "4.35.2",
> "type_vocab_size": 2,
> "use_cache": true,
> "vocab_size": 21128
> }
> ```
>
> ```
> Running on local URL:  http://127.0.0.1:7862
> 
> To create a public link, set `share=True` in `launch()`.
> ```

![image-20231129095847096](./HuggingFace%20Transformers%20Basic.assets/image-20231129095847096.png)



# 2. 基础组件之Pipeline

## 2.1 什么是Pipeline

- 将数据预处理、模型调用、结果后处理 三部分组装成的流水线
- 使我们能够直接输入文本便得到最终的答案

![image-20231129100947507](./HuggingFace%20Transformers%20Basic.assets/image-20231129100947507.png)



## 2.2 Pipeline支持的任务类型

| 名称                                     | 任务类型   |
| ---------------------------------------- | ---------- |
| text-classification (sentiment-analysis) | text       |
| token-classification (ner)               | text       |
| question-answering                       | text       |
| fill-mask                                | text       |
| summarization                            | text       |
| translation                              | text       |
| text2text-generation                     | text       |
| text-generation                          | text       |
| conversational                           | text       |
| table-question-answering                 | text       |
| zero-shot-classification                 | text       |
| automatic-speech-recognition             | multimodal |
| feature-extraction                       | multimodal |
| audio-classification                     | audio      |
| visual-question-answering                | multimodal |
| document-question-answering              | multimodal |
| zero-shot-image-classification           | multimodal |
| zero-shot-audio-classification           | multimodal |
| image-classification                     | image      |
| zero-shot-object-detection               | multimodal |
| video-classification                     | video      |



## 2.3 Pipeline的创建与使用

- 根据任务类型直接创建Pipeline
- 指定任务类型，再指定模型，创建基于指定模型的Pipeline
- 预先加载模型，再创建Pipeline
- 使用GPU进行推理加速

## 2.4 Pipeline的背后实现

 [pipeline.ipynb](..\1-Started\1-pipeline\pipeline.ipynb) 



# 3. 基础组件之Tokenizer

## 3.1 Tokenizer简介

数据预处理
- Step1 分词：使用分词器对文本数据进行分词（字、字词）
- Step2 构建词典：根据数据集分词的结果，构建词典映射（这一步并不绝对，如果采用预训练词向量，词典映射要根据词向量文件进行处理）
- Step3 数据转换：根据构建好的词典，将分词处理后的数据做映射，将文本序列转换为数字序列
- Step4 数据填充与截断：在以batch输入到模型的方式中，需要对过短的数据进行填充，过长的数据进行截断，保证数据长度符合模型能够接受的范围，同时batch内的数据维度大小一致

## 3.2 Tokenizer实现

 [tokenizer.ipynb](..\1-Started\2-tokenizer\tokenizer.ipynb) 



# 4. 基础组件之Model

## 4.1 Model简介

![image-20231205083942140](./HuggingFace%20Transformers%20Basic.assets/image-20231205083942140.png)

- Transformer

  - 原始的Transformer为编码器、解码器模型
  - Encoder部分接收输入并构建完整特征表示，Decoder部分使用Encoder的编码结果以及其他的输入生成目标序列
  - 无论是编码器还是解码器，均由多个TransformerBlock堆叠而成
  - TransformerBlock由注意力机制（Attention）和FFN组成

- 注意力机制

  - 注意力机制的使用是Transformer的一个核心特征，在计算当前词的特征表示时，可以通过注意力机制有选择性的告诉模型要使用哪些上下文

- 模型类型

  <img src="./HuggingFace%20Transformers%20Basic.assets/image-20231205084441477.png" alt="image-20231205084441477" style="zoom:80%;" />

  - 编码器模型：自编码模型，使用Encoder，拥有双向的注意力机制，即计算每一个词的特征时都能看到完整上下文
  - 解码器模型：自回归模型，使用Decoder，拥有单向的注意力机制，即计算每一个词的特征时都只能看到上文，无法看到下文
  - 编码器解码器模型：序列到序列模型，使用Encoder+Decoder，Encoder部分使用双向的注意力，Decoder部分使用单向注意力

  | 模型类型                         | 常用预训练模型                     | 适用任务                         |
  | -------------------------------- | ---------------------------------- | -------------------------------- |
  | 编码器模型：自编码模型           | ALBERT, BERT, DistillBERT, RoBERTa | 文本分类、命名实体识别、阅读理解 |
  | 解码器模型：自回归模型           | GPT, GPT-2, Bloom, LLaMA           | 文本生成                         |
  | 编码器解码器模型：序列到序列模型 | BART, T5, Marian, mBART, GLM       | 文本摘要、机器翻译               |



## 4.2 Model Head

ModelHead时连接在模型后的层，通常为1个或多个全连接层。

ModelHead将模型的编码的表示结果进行映射，以解决不同类型的任务。

![image-20231205085129699](./HuggingFace%20Transformers%20Basic.assets/image-20231205085129699.png)



## 4.3 Model基本使用方法

 [model.ipynb](..\1-Started\3-model\model.ipynb) 



## 4.4 模型微调代码实例

任务类型：文本分类

使用模型：hfl/rbt3

数据集地址：https://github.com/SophonPlus/ChineseNlpCorpus

代码实例： [model.ipynb](..\1-Started\3-model\model.ipynb) 















