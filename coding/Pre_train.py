import torch
import torch.nn as nn
import tiktoken
from DummyGPTModel import generate_text_simple
from DummyGPTModel import GPTModel
from DummyGPTModel import GPT_CONFIG_124M

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 使用.unsqueeze(0)增加batch维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # 去掉batch维度
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# Initialize the model
model = GPTModel(GPT_CONFIG_124M)
start_context = "Every effort moves you"
text1 = "every effort moves"
text2 = "I really like"
tokenizer = tiktoken.get_encoding("gpt2")

inputs = torch.tensor(
    [[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]]
)

targets = torch.tensor(
    [[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]
)

# 屏蔽模型参数的梯度跟踪
with torch.no_grad():
    logits = model(inputs)

# 概率分数(probas)张量的最终张量维度
probas = torch.softmax(logits,dim=-1)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n",token_ids)

# 将词元ID转换回文本
print(f"Targets batch 1: {token_ids_to_text(targets[0],tokenizer)}")
print(f"Outputs batch 1:{token_ids_to_text(token_ids[0].flatten(),tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx,[0,1,2],targets[text_idx]]
print("Text 1:",target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx,[0,1,2],targets[text_idx]]
print("Text 2:",target_probas_2)

# 对概率分数应用对数
log_probas = torch.log(torch.cat((target_probas_1,target_probas_2)))
print("Step 4 Log:",log_probas)

# 平均对数概率
avg_log_probas = torch.mean(log_probas)
print("Step 5 for avg log:",avg_log_probas)

# 负平均对数概率就是平均对数概率乘以-1,
# 深度学习中，将上一步负值-11.0813转换为11.0813 称为交叉熵损失
neg_avg_log_probas = avg_log_probas * -1
print("Step 6 neg avg:",neg_avg_log_probas)

# 上边的好几个步骤，pytorch 可以使用cross_entropy函数完成，
# 先简要回顾一下logits
print("Logits shape:",logits.shape)
print("Targets shape:",targets.shape)

logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()
print("Flattened logits:",logits_flat.shape)
print("Flattened targets:",targets_flat.shape)


# PyTorch 的 cross_entropy 函数将为我们处理所有这些步骤,
# 最后得到的数据与第六步的数据相同
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

# 困惑度通常与交叉熵损失一起用来评估模型在诸如，
# 语言建模等任务中的性能
# 困惑度可以通过perplexity = torch.exp(loss)计算得到
