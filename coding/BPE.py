from importlib.metadata import version
import tiktoken

# version_info : 0.8.0
# print("tiktoken version:", version("tiktoken"))
file_path = "./LLMs/the-verdict.txt"
# 读取文本
with open(file_path,"r",encoding="utf-8") as f:
    raw_text = f.read()
    if not raw_text:
        raise ValueError("文件不存在或者路径不对，请检查文件路径和内容。")


if __name__ == "__main__":
   tokenizer = tiktoken.get_encoding("gpt2")
   text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
   text_test = "Akwirw ier"
   integers = tokenizer.encode(text_test, allowed_special={"<|endoftext|>"})
   # print(integers)
   """
   结果：[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]
   发现一：<|endoftext|> 被分配了最大的词元ID 50256
   发现二：BPE分词器可以正确地编码和解码未知单词，比如'someunknownPlace',不使用<|unk|>词元的前提下是怎么
   处理任何未知词汇的呢？
   BEP算法的原理：将不在预定词汇表中的词元分解为已知的词元组合，
   例如'someunknownPlace'被分解为'some', 'unknown', 'Place'等已知词元。
   其实将单词分解为更小的子词单元甚至是单个字符，从而能够处理词汇表之外的单词。
   """ 
#    print(integers)
#    strings = tokenizer.decode(integers)
#    print("解码结果:",strings) # 解码结果: Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.
   print("编码结果：", integers)
   # 编码结果： [33901, 86, 343, 86, 220, 959]
   strings = tokenizer.decode(integers)
   print("解码结果:",strings)
   # 解码结果: Akwirw ier
   
   
