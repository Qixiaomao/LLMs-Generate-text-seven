#AI #python #LLM 

![[image-29.png]]

# 第一阶段
## 数据准备与采样
**嵌入(embedding)** 将数据->向量格式的过程
深度学习无法直接处理视频、音频、文本等原始格式的数据。因此我们使用嵌入模型将这些原始数据转换为深度学习架构容易理解和处理的密集向量表示。
![[image-30.png|402x230]]
图示显示的是不同数据格式需要使用不同的嵌入模型。
嵌入的本质是将离散对象（如单词、图像甚至整个文档）映射到连续向量空间中的点，其主要目的是将非数值的数据转换为神经网络可以处理的格式。

词嵌入的维度(dimension)可以从一维到数千维不等。更高的维度有助于捕捉到更细微的关系，但这通常以牺牲计算效率为代价。

### 嵌入向量
嵌入向量分为将文本分割为单词、将单词转换为词元，以及将词元转化为嵌入向量。
#### 文本分词
![[image-31.png|445x368]]
词元既可以是单个单词，也可以是包括标点符号在内的特殊字符。如上图所示。

获取文本，并构造简易分词器,将文本分词。
```python:
import urllib.request

import re

'''

获取文本样本，并进行预处理，通过构造简易分词器

打印前30个字符

'''

  

url = ("https://raw.githubusercontent.com/rasbt/"

       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"

       "the-verdict.txt")

file_path = "the-verdict.txt"

urllib.request.urlretrieve(url, file_path)

  

# 通过python读取文本样本

with open(file_path,"r",encoding="utf-8") as f:

    raw_text = f.read()

       # 构造简易分词器

    preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)

    preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]

    print("Total number of characters:",len(preprocessed_text))

    print(preprocessed_text[:30])  # 打印前30个字符
```

之后，就将词元转换为词元ID，如下所示：
![[image-32.png|423x307]]


**词元ID（token ID)** 将词元从Python字符转换为整数表示，这一过程是将词元ID转换为嵌入向量前的必经步骤。不仅需要编码将文本分词后的词元ID汇集成一个表，也要通过逆向词汇表将词元ID转换成对应的唯一词元。

**实现简单的文本分词器**
```python:
# 构造简易分词器

class SimpleTokenizerV1:

       """_summary_ 将词汇表作为类属性存储，以便于在encode方法

       和decode方法中访问，第二行的代码是创建逆向词汇表，将词元ID映射

       回原始文本词文

       """

       def __init__(self, vocab):

              self.str_to_int = vocab

              self.int_to_str = {i:s for s,i in vocab.items()}

       # 处理输入文本，将其转换为词元ID

       def encode(self, text):

              preprocessed = re.split(r'([,.?_!"()\']|--|\s)',text)

              preprocessed = [item.strip() for item in preprocessed if item.strip()]

              ids = [self.str_to_int[s] for s in preprocessed]

              return ids

       # 将词元ID转换回原始文本词汇

       def decode(self, ids):

              text = " ".join([self.int_to_str[i] for i in ids])

              text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)

              return text  # 将空格和标点符号连接起来
```

输出结果：
```python:
前50个词汇表条目：
!:0
":1
':2
(:3
):4
,:5
--:6
.:7
::8
;:9
?:10
A:11
Ah:12
Among:13
And:14
Are:15
Arrt:16
As:17
At:18
Be:19
Begin:20
Burlington:21
But:22
By:23
Carlo:24
Carlo;:25
Chicago:26
Claude:27
Come:28
Croft:29
Destroyed:30
Devonshire:31
Don:32
Dubarry:33
Emperors:34
Florence:35
For:36
Gallery:37
Gideon:38
Gisburn:39
Gisburns:40
Grafton:41
Greek:42
Grindle:43
Grindle::44
Grindles:45
HAD:46
Had:47
Hang:48
Has:49

编码结果：
[1, 58, 2, 872, 1013, 615, 541, 763, 5, 1155, 608, 5, 1, 69, 7, 39, 873, 1136, 773, 812, 7]    
解码结果：
" It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.
```

### 滑动窗口：批次数量，最大长度，以及步长对于张量输入的影响

下边的结果分别显示了 batch_size, max_length, stride 大小对于最后张量显示的影响结果：
```python:
PS D:\Mylearn\INTI courses\毕业论文相关\LLMs> python .\sliding_window.py
编码情况: 5145
[tensor([[ 40, 367]]), tensor([[ 367, 2885]])]
输入形状: torch.Size([1, 2])
目标形状: torch.Size([1, 2])
PS D:\Mylearn\INTI courses\毕业论文相关\LLMs> python .\sliding_window.py
编码情况: 5145
[tensor([[  40,  367, 2885, 1464, 1807, 3619,  402,  271]]), tensor([[  367,  2885,  1464,  1807,  3619,   402,   271, 10899]])]
输入形状: torch.Size([1, 8])
目标形状: torch.Size([1, 8])
PS D:\Mylearn\INTI courses\毕业论文相关\LLMs> python .\sliding_window.py
编码情况: 5145
[tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]]), tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])]
输入形状: torch.Size([8, 4])
目标形状: torch.Size([8, 4])
```
**结论：**
- 避免不同批次之间的数据重叠，因为过多的重叠可能会增加模型过拟合的风险。

### 创建词元嵌入
为大语言模型训练准备输入文本的最后一步是将词元ID转换为嵌入向量。初始阶段，必须用随机值初始化这些嵌入权重，这是大预言模型学习的起点。
![[image-33.png|534x457]]
如上图所示：大模型的输入文本准备工作包括文本分词、将词元转换为词元ID，以及将词元ID转换为嵌入向量。本小结，利用词元ID来创建词元嵌入向量。由于类GPT大模型是使用反向传播算法(backpropagation algorithm)训练的深度神经网络，因此需要连续的向量表示或嵌入。
根据文本词汇量，然后嵌入的词元权重如下：
```python:
编码情况: 5145
嵌入形状： torch.Size([160, 768])
嵌入权重： Parameter containing:
tensor([[-0.5591,  1.3307,  0.4330,  ..., -0.4343,  1.0051, -0.9435],
        [ 0.3773,  1.1433,  0.8273,  ..., -0.4734,  0.3262,  0.1165],
        [-1.1760, -0.4040,  0.2519,  ..., -0.3626, -0.7352,  0.0739],
        ...,
        [ 0.5261, -0.4806,  0.2917,  ..., -0.2439, -0.0767, -2.4854],
        [ 0.0762,  0.8324, -0.8247,  ...,  0.3003,  0.2424,  0.1773],
        [ 1.7645,  0.7379,  0.6943,  ...,  0.5567, -0.2391,  1.7904]],
       requires_grad=True)
```

#### 编码单词位置信息
理论上，词元嵌入非常适合作为大模型的输入。然而，大模型存在一个小缺陷——它们的自注意力机制
无法感知词元在序列中的位置或顺序。嵌入层的工作机制是，无论词元ID在输入序列中的位置如何，相同的词元ID始终被映射到相同的向量表示。
嵌入位置的方式：绝对位置嵌入方式，相对位置嵌入方式。OpenAI 的GPT模型使用的是绝对位置嵌入，这些嵌入会在训练过程中被优化，有别于原始Transformer模型中的固定或预定义位置编码。
过程代码如下：
```python:
from importlib.metadata import version

import tiktoken

import torch

from torch.utils.data import Dataset, DataLoader

  

def read_text(file_path):

    """读取文本文件内容"""

    with open(file_path, "r", encoding="utf-8") as f:

        raw_text = f.read()

        if not raw_text:

            raise ValueError("文件不存在或者路径不对，请检查文件路径和内容。")

    return raw_text

  

# 批处理输入和目标的数据集

class GPTDataseV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):

        self.input_ids = []

        self.target_ids = []

        token_ids = tokenizer.encode(txt)  # 对全部文本进行分词

        for i in range(0, len(token_ids) - max_length, stride):

            # 使用滑动窗口将文本划分为长度

            # 为max_length的重叠序列

            input_chunk = token_ids[i:i + max_length]

            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))

            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):

        return len(self.input_ids) # 返回数据集的长度

    def __getitem__(self, index):

        return self.input_ids[index], self.target_ids[index]

  

# 批量生成输入-目标对的数据加载器

def create_data_loader_v1(txt, batch_size=4, max_length=256,

                       stride=128, shuffle=True, drop_last=True,

                       num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2") # 初始化分词器

    dataset = GPTDataseV1(txt, tokenizer, max_length, stride) # 创建数据集

    dataloader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        drop_last=drop_last, # 是否丢弃最后一个不完整的批次

        num_workers=num_workers # 多线程加载数据

    )

    return dataloader

  

if __name__ == "__main__":

    file_path = r"D:\Mylearn\INTI courses\毕业论文相关\LLMs\the-verdict.txt"

    raw_text = read_text(file_path)

    tokenizer = tiktoken.get_encoding("gpt2")

    enc_text = tokenizer.encode(raw_text)

    print("编码情况:",len(enc_text)) # 词元总数

    # stride 决定了滑动窗口的步长

    # max_length 决定了每个批次的最大长度

    # batch_size 决定了每个批次的样本数

    dataloader = create_data_loader_v1(

        raw_text,batch_size=8, max_length=4,stride=4,

        shuffle=False

    )

    # 将dataloader转换为可迭代对象

    data_iter = iter(dataloader)

    inputs, targets = next(data_iter)

    print("Token IDs:",inputs) # 打印输入的Token IDs

    print("\nInput shape:\n",inputs.shape) # 打印输入的形状

    # 创建词元嵌入

    # 这里使用GPT2的词元嵌入，可以用其他模型的词元嵌入

    vocab_size = 50257  # GPT-2的词表大小

    output_dim = 256  # 假设嵌入维度为256

    max_length = 4  # 假设每个批次的最大长度为4

    # 如果设定批次大小为8，每个批次包含4个词元，256维度，结果是(8, 4, 256)

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # 创建嵌入层

    # 将输入的Token IDs转换为嵌入向量

    token_embedding = token_embedding_layer(inputs)

    print("\n嵌入向量形状:\n", token_embedding.shape) # 打印嵌入向量的形状

    # 为了获取GPT模型采用的绝对位置嵌入，只需要创建一个维度与token_embedding_layer相同的位置嵌入层

    context_length = max_length  # 假设每个批次的最大长度为4

    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) # 创建位置嵌入层

    pos_embeddings = pos_embedding_layer(torch.arange(context_length)) # 得到位置嵌入

    print("\n位置嵌入形状:\n",pos_embeddings.shape) # 打印位置嵌入的形状

    input_embeddings = token_embedding + pos_embeddings # 合并位置嵌入和词元嵌入

    print("\n输入嵌入形状：\n",input_embeddings.shape) # 打印输入嵌入的形状
```
最后嵌入位置信息的结果如下：
```python:
编码情况: 5145
Token IDs: tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Input shape:
 torch.Size([8, 4])

嵌入向量形状:
 torch.Size([8, 4, 256])

位置嵌入形状:
 torch.Size([4, 256])
PS D:\Mylearn\INTI courses\毕业论文相关\LLMs> python .\sliding_window.py
编码情况: 5145
Token IDs: tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Input shape:
 torch.Size([8, 4])

嵌入向量形状:
 torch.Size([8, 4, 256])

位置嵌入形状:
 torch.Size([4, 256])

输入嵌入形状：
 torch.Size([8, 4, 256])
```

![[image-34.png|409x526]]
如上图所示，在输入处理流水线中，输入文本首先被分隔为独立的词元，随后，这些词元通过词汇表转换为词元ID。这些词元ID继而被转换为嵌入向量，并添加与之大小相同位置嵌入，最终形成用于大模型核心层的输入嵌入。

### 自注意力机制
自注意力机制是所有基于transformer架构的大模型基石，一旦掌握了其基本原理，你就能攻克本书以及大模型实现过程中最具挑战的部分之一。
**自注意力机制中的”自“** 指的是通过关联单个输入序列中的不同位置来计算注意力权重的能力。可以评估并学习输入本身各个部分之间的关系和依赖，比如句子中的单词或者图像中的像素。

**点积** 本质上是将两个向量逐个元素相乘然后对乘积求和的简洁方法。
点积不仅仅被视为一种将两个向量转化为标量值的数学工具，而且也是度量相似度的一种方式，因为它可以量化两个向量之间的对齐程度：
==点积越大，向量之间的对齐程度或相似度就越高。在自注意力机制中，点积决定了序列中每个元素对其他元素的关注程度：点积越大，两个元素之间的相似度和注意力分数就越高。==
如下所示代码，构建了一个不带权重的注意力点积计算：
```python:
import torch

import torch.nn as nn

  

# 假设该句已经按照讨论的方式嵌入为三维向量，

# 选择较小的嵌入维度进行说明

inputs = torch.tensor(

  

  [[0.43, 0.15, 0.89], # Your     (x^1)

   [0.55, 0.87, 0.66], # journey  (x^2)

   [0.57, 0.85, 0.64], # starts   (x^3)

   [0.22, 0.58, 0.33], # with     (x^4)

   [0.77, 0.25, 0.10], # one      (x^5)

   [0.05, 0.80, 0.55]] # step     (x^6)

  

)

  

query = inputs[1] # 选择第二个词元作为查询向量

attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):

    # 计算每个词元与查询向量的点积

    attn_scores_2[i] = torch.dot(x_i, query)

print("注意力分数（查询向量为第二个词元）:\n", attn_scores_2)

  

## 理解点积

# 点积是两个向量的内积，即两个向量的长度乘以夹角的余弦值。

res = 0

for idx, element in enumerate(inputs[0]):

    res += element * inputs[1][idx]

print("\n点积计算结果（第一个词元与第二个词元）:\n", res)

print("点积计算结果（第一个词元与第二个词元）与torch.dot的结果相同:", torch.dot(inputs[0], query))
```
最后结果如下：
```python:
注意力分数（查询向量为第二个词元）:
 tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

点积计算结果（第一个词元与第二个词元）:
 tensor(0.9544)
点积计算结果（第一个词元与第二个词元）与torch.dot的结果相同: tensor(0.9544)
```

下一步就是对先前计算的每个注意力分数进行归一化处理。归一化处理的目的：获得总和权重为1的注意力权重。
这种归一化是一个惯例，有助于解释结果，并能维持大模型的训练稳定性。
实际应用来说，使用softmax函数进行归一化更为常见，而且是一种更可取的做法。这种做法更好地处理了极值，并在训练期间提供了更有利地梯度特性。
```python:

atten_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("\n注意力权重：\n",atten_weights_2_tmp)

print("权重总和：\n", atten_weights_2_tmp.sum())
'''
结果:
注意力权重：
 tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
权重总和：
 tensor(1.0000)
 '''
# 一般归一化处理， softmax自制函数处理， 使用PyTorch的softmax函数处理
# 归一化处理

atten_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("\n注意力权重：\n",atten_weights_2_tmp)

print("权重总和：\n", atten_weights_2_tmp.sum())

  

# softmax 函数实现归一化

def softmax_native(x):

    # 计算softmax函数

    # return torch.exp(x) / torch.sum(torch.exp(x), dim=0)

    # 这里与上一版本的区别是将dim改位-1，以适应不同版本的pytorch

    # 同时将keepdim = True, 以保持维度不变

    return torch.exp(x) / torch.exp(x).sum(dim=-1,keepdim=True)

atten_weights_2_native = softmax_native(attn_scores_2)

print("注意力权重：\n",atten_weights_2_native)

print("权重总和: \n",atten_weights_2_native.sum())

  

# 实践中建议使用PyTorch 的 softmax实现，该实现经过了大量性能优化

attn_weights_2 = torch.softmax(attn_scores_2,dim=-1)

print("Attention weights:",attn_weights_2)

print("Attention weights sum:",attn_weights_2.sum())
```

#### 带权重的注意力机制
![[image-35.png]]

通过引入3个可训练的权重矩阵 $W_{q}、W_{k}、W_{v'}$ 一步步的实现自注意力机制。
这3个矩阵用于将嵌入的输入词元$x^{(i)}$ 分别为查询向量、键向量和值向量。如下图所示：
![[image-36.png]]

