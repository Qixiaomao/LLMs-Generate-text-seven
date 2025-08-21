import torch
import torch.nn as nn
import tiktoken
from MultiHeadAttn_demo import MultiHeadAttentionV1

'''
层归一化、GELU激活函数、前馈神经网络和残差连接模块
Transformer模块

'''
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False
        )
        
    def forward(self, in_idx):
        # device 的设置允许我们在CPU或GPU上运行模型
        # 具体取决于输入数据所在设备
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        
        
        
# class DummyGPTModel(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         # 使用占位符替换TransformerBlock
#         self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
#         self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
#         self.drop_emb = nn.Dropout(cfg["drop_rate"])
#         self.trf_blocks = nn.Sequential(
#             *[DummyTransformerBlock(cfg)
#               for _ in range(cfg["n_layers"])]
#         )
#         # 使用占位符替换层归一化
#         self.final_norm = DummyLayerNorm(cfg["emb_dim"])
#         self.out_head = nn.Linear(
#             cfg["emb_dim"], cfg["vocab_size"], bias=False
#         )
        
#     def forward(self, in_idx):
#         batch_size, seq_len = in_idx.shape
#         tok_embeds = self.tok_emb(in_idx)
#         pos_embeds = self.pos_emb(
#             torch.arange(seq_len, device=in_idx.device)
#         )
#         x = tok_embeds + pos_embeds
#         x = self.drop_emb(x)
#         x = self.trf_blocks(x)
#         x = self.final_norm(x)
#         logits = self.out_head(x)
#         return logits
    

# 一个简单的占位符类，稍后将被真正的TransformerBlock替换
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    # 这块代码不执行任何操作，只返回其输入
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    # 这里的参数只是为了模仿层归一化的接口
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        
    def forward(self, x):
        return x

# 层归一化类
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        # 关闭科学及算法
        torch.set_printoptions(sci_mode=False)
        norm_x = (x-mean) / torch.sqrt(var+self.eps)
        
        return norm_x * self.scale + self.shift

# GELU激活函数的实现
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715 * torch.pow(x,3))))

# FeedForward模块 : 一个小型的神经网络
# 由两个线性层和一个GELU激活函数组成
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )  
        
    def forward(self, x):
        return self.layers(x)

# 残差连接
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        # 5层网络实现
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1],layer_sizes[2]),GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2],layer_sizes[3]),GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3],layer_sizes[4]),GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),GELU())
        ])
        
    # 计算当前层的输出，并检查是否可以使用快捷连接
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x+layer_output
            else:
                x = layer_output
        return x

def print_gradients(model, x):
    # 定义损失函数，打印所有损失参数
    output = model(x)
    # 前向传播
    target = torch.tensor([[0.]])
    
    # 计算损失 : 基于目标和输出之间的差距计算损失
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    loss.backward() # 反向传播
    
    # 打印所有参数的梯度
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# GPT 的Transformer模块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionV1(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self,x):
        # 注意力块中添加快捷链接
        shortcut = x 
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # 将原始输入添加回来
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# 当前文本截断至支持的长度，如果大语言模型仅仅支持
# 5个词元，但此时文本长度为10，
# 则只有最后5个词元被保留
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 定义一个空列表来保存生成的文本
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            
        logits = logits[:,-1,:]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1,keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == "__main__": 
    GPT_CONFIG_124M={
        "vocab_size":50257, # vocabulary size
        "context_length":1024, # context length
        "emb_dim":768, # embedding dimension
        "n_heads":12, # number of heads
        "n_layers":12, # number of layers
        "drop_rate":0.1, # dropout rate
        "qkv_bias": False # query-key-value bias
    }   
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    # 这里的tokenizer.encode()函数会将文本转换为token id的列表
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    # 将batch转换为tensor
    batch = torch.stack(batch)
    # print(batch)
    
    # 初始化参数量为1.24亿的DummyGPTModel实例
    # torch.manual_seed(123)
    # model = DummyGPTModel(GPT_CONFIG_124M)
    # logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits)
    
    # 均值表示数据中的位置，方差表示数据在位置上的分布
    # 层归一化的作用是将数据标准化，使得数据在各个位置上分布相似
    # 这里的LayerNorm类模仿了层归一化的接口
    # 测试LayerNorm模块
    torch.manual_seed(123)
    batch_example = torch.randn(2,5)

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1,keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True)
    # print("mean:\n",mean)
    # print("var:\n",var)
    x = torch.rand(2,4,768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print("Input shape:\n", x.shape)
    print("Output shape:\n",output.shape)
    # out = ffn(x)
    # # print(out.shape)
    # layer_size = [3,3,3,3,3,1]
    # sample_input = torch.tensor([[1.,0.,-1.]])
    # torch.manual_seed(123)
    # # 实例化不带跳跃连接的模型
    # # model_without_shortcut = ExampleDeepNeuralNetwork(
    # #     layer_size, use_shortcut=False
    # # )
    # # print_gradients(model_without_shortcut, sample_input)
    # # 实例化包含跳跃连接的模型
    # model_with_shortcut = ExampleDeepNeuralNetwork(
    #     layer_size, use_shortcut=True
    # )
    #print_gradients(model_with_shortcut, sample_input)
    
    # 测试GPTModel模块
    model = GPTModel(GPT_CONFIG_124M)
    
    out = model(batch)
    #print("Input batch:\n",batch)
    #print("\nOutput logits:",out.shape)
    #print(out)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    # 因为权重共享的原因，所以需要减去输出层的权重数
    total_params_gpt2 = (
        total_params - sum(p.numel()
        for p in model.out_head.parameters())
    )
    # print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
    
    # 计算前馈神经网络和注意力模块的参数数量
    Trans_block = TransformerBlock(GPT_CONFIG_124M)
    # print(Trans_block)
    total_params_ff = sum(p.numel() for p in Trans_block.ff.parameters())
    # print(f"Total number of parameters in feed forward module:{total_params_ff:,}")
    
    total_params_attn = sum(p.numel() for p in Trans_block.att.parameters())
    # print(f"Total number of parameters in attention module:{total_params_attn:,}")
    # 计算GPTModel的1.63亿个参数的内存需求
    # 计算总的字节大小（假设每个参数占用4字节的32位浮点数）
    total_size_bytes = total_params * 4
    # 转换为兆字节(MB)
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    # print(f"Total size of the model :{total_size_mb:.2f} MB")
    
    # 测试生成文本:提供一个上下文的大小，并编码为词元ID
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:\n",encoded)
    # 添加batch 维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:\n",encoded_tensor.shape)
    # 将模式设置为eval模式，禁用dropout等只在训练间使用的随机组件
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:",out)
    print("Output length:",len(out[0]))
    # 使用分词decode方法可以将ID转换为文本
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)