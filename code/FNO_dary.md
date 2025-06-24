
## 数据预处理

```
load_darcy_flow_small(
    n_train, 
    n_tests, 
    batch_size, 
    test_batch_sizes, 
    data_root=WindowsPath('E:/Hankel-FNO/neuraloperator/neuralop/data/datasets/data'), 
    test_resolutions=[16, 32], 
    encode_input=False, 
    encode_output=True, 
    encoding='channel-wise', 
    channel_dim=1
)
```


## 模型构建(FNO函数)
![[Fourier-Neural-Operator-FNO-network-architecture.png]]


1. **Lifting Layer（升维层）**
    
    - 将输入函数 u(x,y)u(x, y)（如初始场）通过一个线性映射 PP，升维至隐藏通道空间，增强网络容量。
        
2. **傅里叶层（Fourier Layer）重复堆叠**  
    每一层操作如下：
    
    - **FFT（傅里叶变换）**：将当前表示从空间域转换到频域 v^(kx,ky)\hat{v}(k_x, k_y)；
        
    - **频域线性变换**：乘以可学习参数 RR，限制到前 `n_modes × n_modes` 低频模式；
        
    - **iFFT（逆傅里叶变换）**：将频域信号转换回空间域；
        
    - 加上一个空间域的点级变换（channel-wise linear）和非线性激活，形成残差连接。
        
3. **Projection Layer（投影层）**
    
    - 最后一层将隐藏通道再次 **降维** 到目标输出通道数 QQ，生成预测结果。

### 💡 功能简介

`FNO` 是 N 维 **Fourier Neural Operator**，用于在规则网格上通过**傅里叶卷积（Fourier Convolutions）**学习函数到函数的映射关系，适用于偏微分方程（PDE）求解等科学建模任务。

---

### 🔧 关键参数解释

|参数名|类型|说明|
|---|---|---|
|`n_modes`|`Tuple[int]`|在每个空间维度上保留的频率模式数量（如 `(16, 16)` 表示保留低频 16×16）|
|`in_channels`|`int`|输入的通道数，如标量场为 1，矢量场可为 2 或 3|
|`out_channels`|`int`|输出通道数，通常与预测目标数量相等|
|`hidden_channels`|`int`|隐藏层通道数（类似于网络宽度）|
|`n_layers`|`int`|使用的谱卷积层数，默认为 4|
|`lifting_channel_ratio`|`float/int`|输入升维比例（决定 lifting 层的宽度）|
|`projection_channel_ratio`|`float/int`|输出降维比例（决定 projection 层的通道数）|
|`positional_embedding`|`str or nn.Module`|坐标编码方式，如 `'grid'`、`'fourier'`，或自定义模块|
|`non_linearity`|`nn.Module`|激活函数，默认为 `gelu`|
|`norm`|`str`|可选归一化方法：`'ada_in'`, `'group_norm'`, `'instance_norm'`|
|`complex_data`|`bool`|是否处理复数输入（如复值声场）|
|`use_channel_mlp`|`bool`|是否在通道维度使用 MLP（提升表达能力）|
|`channel_mlp_dropout`|`float`|MLP 的 Dropout 比例|
|`channel_mlp_expansion`|`float`|MLP 扩张比例|
|`channel_mlp_skip`|`str`|MLP 的 skip connection 类型：`'linear'`, `'identity'`, `'soft-gating'`|
|`fno_skip`|`str`|谱卷积层的 skip 类型|
|`domain_padding`|`int/float/list`|是否进行空间维度的补边，常用于边界处理|
|`domain_padding_mode`|`str`|补边方式，如 `'symmetric'`, `'one-sided'`|
|`fno_block_precision`|`str`|计算精度，如 `'full'`、`'mixed'` 等|
|`factorization`, `rank`, `fixed_rank_modes`, `implementation`|`str/float/bool`|控制谱卷积的低秩分解策略与实现方式，通常用于加速|
|`conv_module`|`nn.Module`|指定卷积模块，默认使用 `SpectralConv`（频域卷积）|

---

### 📌 模型核心组成

- `Lifting layer`：将低维输入升维到 `hidden_channels`
    
- `SpectralConv blocks` × `n_layers`：在频域中进行卷积操作
    
- `Channel MLP`：提高非线性拟合能力
    
- `Projection layer`：将特征映射回输出维度
    

---

### 🔁 工作机制概述

```text
输入函数 u(x, y)
 ↓
升维映射 → 坐标嵌入
 ↓
谱卷积层（FFT → 权重乘法 → iFFT）
 ↓
非线性 + 残差连接 + 归一化
 ↓
通道 MLP
 ↓
降维输出
```

---

### 🧾 总结

`FNO` 是一种高效、结构清晰的函数逼近架构，适合解决空间维度固定、边界已知、具有全局结构的物理建模问题，如：

- 地下流体（Darcy flow）
    
- 热传导与扩散
    
- 声场传播与波动问题
    

如果你有具体应用场景，比如声压场建模，我可以帮你配置一套合适的 FNO 参数。需要吗？

## 参数个数打印

# `sys.stdout.flush()` 的作用解释

`sys.stdout.flush()` 是一个用于强制刷新标准输出缓冲区的命令。

## 详细解释

1. **输出缓冲机制**：
   - 在 Python 中，标准输出（通常是控制台）是带缓冲的
   - 这意味着当你使用 `print()` 语句时，输出内容可能不会立即显示在控制台上
   - 输出内容通常会被存储在缓冲区中，直到以下情况发生才会实际显示：
     - 缓冲区已满
     - 遇到换行符
     - 程序结束
     - 或显式调用 `flush()`

2. **`flush()` 的作用**：
   - 强制将缓冲区中的所有内容立即写入到标准输出
   - 确保到目前为止的所有打印内容立即可见

3. **在代码中的用途**：
   - 确保即使在长时间运行的训练循环中，也能看到实时输出
   - 特别是在训练神经网络时，可以确保进度日志立即显示，而不是等待某个缓冲条件满足

## 实际应用场景

在 plot_FNO_darcy.py 文件中，使用 `sys.stdout.flush()` 的位置是在打印模型参数信息后：

```python
n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()  # 确保上面的打印立即显示
```

后续在打印训练设置信息后也有类似用法：

```python
print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()  # 确保所有信息立即显示
```
