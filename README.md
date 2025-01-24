## 概述
这是一个基于 **GPT** (Generative Pretrained Transformer) 架构的模型实现。该实现使用了 **自定义分词器**、**多头自注意力机制**、**位置编码** 等关键技术。它包括了训练和生成文本的功能，可以用于文本生成、语言建模等任务。

---

## 功能
- **自定义分词器**：使用 `tiktoken` 库加载和处理文本数据。
- **多头自注意力机制**：用于处理文本中的词间关系。
- **位置编码**：通过正弦和余弦函数实现的位置编码，增强模型对序列顺序的理解。
- **文本生成**：支持文本生成，输入一段文本，模型会生成下一个词。

---

## 依赖项
- **Python 3.7+**
- **PyTorch 1.8+**
- **tiktoken**（用于分词处理）

可以通过以下命令安装依赖：

```bash
pip install torch tiktoken
```

---

## 安装
1. 克隆仓库到本地：

```bash
git clone https://github.com/yourusername/gpt-model.git
cd gpt-model
```

2. 安装所需的依赖项：

```bash
pip install -r requirements.txt
```

---

## 使用方法

### 初始化模型
要初始化模型，创建 `GPT` 类的实例并设置必要的参数：

```python
from gpt_model import GPT, Positional_Encoding

# 设置模型参数
d_model = 512
n_layers = 2
num_heads = 4
vocab_size = 50257  # 假设词汇表大小

# 初始化模型
model = GPT(vocab_size, d_model, num_heads, n_layers).to(device)
```

### 训练模型
模型的训练过程使用的是 **交叉熵损失函数** 和 **AdamW 优化器**：

```python
epochs = 5000
for ep in range(epochs):
    xb, yb = train_loader.get_batch()  # 获取批次数据
    logits, loss = model(xb, yb)  # 前向传播和损失计算
    
    # 梯度更新
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if ep % eval_step == 0 or ep == epochs - 1:
        model.eval()
        with torch.no_grad():
            xvb, yvb = eval_loader.get_batch()  # 获取验证数据
            _, e_loss = model(xvb, yvb)  # 验证损失

            print(f'Epoch: {ep}, train_loss: {loss}, eval_loss: {e_loss}')
        model.train()
```

### 文本生成
通过 `generate` 方法，模型可以根据给定的输入生成文本：

```python
inputs = torch.tensor([[1]], device=device)  # 输入一个token
generated_text = model.generate(inputs, max_new_tokens=50)  # 生成50个新token
print(generated_text)
```

---

## 配置项
- `d_model`：模型的隐藏层维度（默认：`512`）
- `n_layers`：模型的层数（默认：`2`）
- `num_heads`：每个自注意力层的注意力头数（默认：`4`）
- `vocab_size`：词汇表大小
- `context_length`：每个批次的最大序列长度

---

## 示例
以下是完整的示例，包括模型初始化、训练和文本生成：

```python
from gpt_model import GPT, Positional_Encoding

# 设置模型参数
d_model = 512
n_layers = 2
num_heads = 4
vocab_size = 50257
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化模型
model = GPT(vocab_size, d_model, num_heads, n_layers).to(device)

# 训练
epochs = 1000
eval_step = 500
for ep in range(epochs):
    xb, yb = train_loader.get_batch()  # 获取训练数据
    logits, loss = model(xb, yb)  # 前向传播
    optim.zero_grad()
    loss.backward()
    optim.step()

    if ep % eval_step == 0 or ep == epochs - 1:
        model.eval()
        with torch.no_grad():
            xvb, yvb = eval_loader.get_batch()  # 获取验证数据
            _, e_loss = model(xvb, yvb)
            print(f'Epoch: {ep}\tlr:{lr}\ttrain_loss: {loss}\teval_loss: {e_loss}')
        model.train()

# 文本生成
inputs = torch.tensor(tokenizer.encode('love'), dtype=torch.long, device=device).unsqueeze(0)
generated_text = m.generate(inputs, max_new_tokens=50)[0]
print(generated_text)

# 输出
Epochs: 0    lr: 0.001    train_loss: 10.9810    eval_loss: 9.3945
Epochs: 500    lr: 0.001    train_loss: 2.4500    eval_loss: 5.0497
Epochs: 999    lr: 0.001    train_loss: 0.4665    eval_loss: 3.5088

love me,
Every year me,
I got a man,
If you were both first time,
It might not supposed to be taking a parked car I'm sitting here, man is old friend by the field in love,
Come on
```

---

## 总结
这个实现提供了一个简化的 GPT 模型，支持文本的训练和生成。模型利用 **多头自注意力**、**位置编码** 和 **交叉熵损失** 进行训练，并且支持基于输入文本生成新的内容。
