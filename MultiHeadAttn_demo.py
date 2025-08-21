import torch
import torch.nn as nn

# 单头注意力机制
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # 添加了dropout层
        self.register_buffer(
            'mask', # 添加了掩码，因果注意力机制
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec




# 编写一个多头注意力机制的封装类
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out,context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(
            d_in, d_out, context_length, dropout, qkv_bias
        )
                                    for _ in range(num_heads)])
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
  
# 合并多头
class MultiHeadAttentionV1(nn.Module):
    def __init__(self, d_in,
                 d_out, context_length,dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
            
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 减少投影维度以匹配所需的输出维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # 使用一个线性层来组合头的输出
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
        
    def forward(self, x):
        b , num_tokens, d_in = x.shape
        # 张量形状
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens,self.num_heads, self.head_dim)
        
        # 从形状(b, num_tokens, num_heads, head_dim)
        # 变换为(b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        # 计算每个头的点积
        attn_scores = queries @ keys.transpose(2,3)
        # 被截断为词元数量的掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1,2)
        
        # 组合头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        # 添加一个可选的线性投影
        context_vec = self.out_proj(context_vec)
        return context_vec
                  
    

if __name__ == '__main__':
    # 测试一下多头注意力机制的封装类
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) # torch.Size([2, 6, 3]) 

    # 实例化一个多头注意力机制的封装类
    torch.manual_seed(123)
    context_length = batch.shape[1] # 词元的数量
    d_in, d_out =  3,1 # 输入和输出的维度
    # 上下文向量 ： d_out * num_heads
    # 如果要将思维改变为二维嵌入向量，只需要将d_out 改为1即可
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vec = mha(batch)
    print(context_vec)
    print("context_vecs.shape:\n", context_vec.shape) # torch.Size([2,6,2])
    # torch.Size([2, 6, 2]) 这个输出的含义：两个输入向量，六个词元，每个词元需要两个维度嵌入
    # 所以输出的维度是2，即每个词元的两个维度嵌入
    
    # 测试一下多头注意力机制的实现类
    #torch.manual_seed(123)
    # ha = MultiHeadAttentionV1(d_in, d_out, context_length, 0.0, num_heads=2)
    
    a = torch.tensor(
        [[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]]
    )
    print(a @ a.transpose(2,3))
    