
## A .Physical Background

for a medium with constant density, a three dimensional (3D) Helmholtz equation in cylindrical coordinates (r, φ, z) can be written as:

$$
\frac{1}{r} \frac{\partial}{\partial r}\left( r \frac{\partial p}{\partial r} \right) + \frac{1}{r^{2}} \frac{\partial^2p}{\partial \varphi^{2}}
+ \frac{\partial^{2}p}{\partial z^{2}}+ \frac{\omega^2}{c^{2}}p = 0
\tag{2} $$

Assuming azimuthal symmetry, (2) can be reformulated as:

$$\frac{\partial^2 p}{\partial r^2} + \frac{1}{r} \frac{\partial p}{\partial r} + \frac{\partial^2 p}{\partial z^2} + \frac{\omega^2}{c^2} p = 0\tag{3}$$

the acoustic pressure can be factorized as
$$
p(r, z) = \psi(r, z) H_0^{(1)}(k_0 r)\tag{4}$$

Hankel 函数满足的贝塞尔微分方程（Bessel Equation）

$$\frac{\partial^2 H_0^{(1)}(k_0 r)}{\partial r^2} + \frac{1}{r} \frac{\partial H_0^{(1)}(k_0 r)}{\partial r} + k_0^2 H_0^{(1)}(k_0 r) = 0\tag{5}$$

Hankel 函数远场近似（Far-field approximation）

$$H_0^{(1)}(k_0 r) \approx \sqrt{\frac{2}{\pi k_0 r}} \, e^{i(k_0 r - \frac{\pi}{4})}\tag{6}$$

对ψ的主控制方程（Generalized wave equation for envelope function）
$$\frac{\partial^2 \psi}{\partial r^2} + 2 i k_0 \frac{\partial \psi}{\partial r} + \frac{\partial^2 \psi}{\partial z^2} + k_0^2 (n^2 - 1) \psi = 0\tag{7}$$

小角度近似（Small-angle approximation）

$$\frac{\partial^2 \psi}{\partial r^2} \ll 2 i k_0 \frac{\partial \psi}{\partial r}\tag{8}$$
标准抛物线方程（Standard parabolic equation）

$$2 i k_0 \frac{\partial \psi}{\partial r} + \frac{\partial^2 \psi}{\partial z^2} + k_0^2 (n^2 - 1) \psi = 0\tag{9}$$


## B.1 Input Encoding method


### Hankel Function Encoding

only encode the amplitude of Hankel function,


### Bathymetry Encoding

**SSP** = **Sound Speed Profile**  
即：**声速剖面**

- **声速剖面**是指：**在水体（如海洋、湖泊）中，不同深度处的声速分布情况。**
    
- 由于温度、盐度、压力随深度变化，水下的声速不是一个常数，而是一个随深度z变化的函数 c(z)。

combining it with the SSP data directly is more convenient.

### Conclusion

the input of Hankel-FNO consists of three channels: **SSP data with Hankel function encoding**, **bathymetry information**, and **positional encoding**.

## B.2 Output Design Method

directly set TL as the output


## Training


基于公式（10），给定一个映射 $G^\dagger : A \rightarrow U$ 以及 $N$ 对数据 $\{a^{(i)}, u^{(i)}\}_{i=1}^N$，其中 $a^{(i)} \in U$ 且 $u^{(i)} \in V$，我们的目标是为参数化映射 $G_\theta$ 寻找一个最优的参数 $\theta^\dagger$，使得

$$
G_\theta(a^{(i)}) \approx G^\dagger(a^{(i)}) = u^{(i)}.
$$
然而，尽管理论上神经算子可以学习无限维空间之间的映射，但实际实现中由于计算资源限制，必须在有限的离散数据点集上进行训练。

在我们的实现中，我们假设 $D = D'$，并且设 $D^{(i)} = \{x_k^{(i)}\}_{k=1}^K$ 是区域 D 的一个包含 KK 个点的离散化表示。则离散化后的输入-输出对可以表示为 $\{a^{(i)}|_{D^{(i)}}, u^{(i)}|_{D^{(i)}}\}$。

D 表示**定义域**，也就是“函数的输入空间”或者“物理场分布的空间区域”。在实现时，**D** 会被离散化为一系列**空间点**（比如用网格采样）

优化问题可以写为：
$$\theta^\dagger = \arg\min_\theta \sum_{i=1}^N L\left(G_\theta(a^{(i)}|_{D^{(i)}}), u^{(i)}|_{D^{(i)}}\right),
$$
其中 $L(\cdot, \cdot)$ 表示损失函数。模型参数 $\theta$ 通过基于反向传播的梯度下降法进行有效优化。
- $D^{(i)}$ 是第 i 组样本的离散化网格点集合。
    
- $a^{(i)}|_{D^{(i)}}$​ 就是在这些点上的输入特征。
    
- $u^{(i)}|_{D^{(i)}}$​ 是这些点上的目标输出（比如声压、温度、速度等）。


## Inference


## Transfer Learning(迁移学习)

  
为了将物理知识和传统微分方程求解器的约束条件融入傅里叶神经算子中，我们可以对输入做以下3方面的设计。

考虑到训练一个模型需要大量数据，尤其对于海洋声学有关的场景是格外棘手的，因此，研究采取了一种微调的方式，模型最初使用**具有相似水深地形信息、单一震源频率和固定震源位置**的数据进行训练。

- In practice, training such a model needs a large amount of training data, which is often intractable in ocean acoustic-related scenarios.

- The model is initially trained using data with **similar bathymetry information, a single source frequency, and a fixed source position.**

- Subsequently, it is **finetuned(微调)** with a limited dataset encompassing different environmental and source conditions.

微调的目标是通过最小化损失函数L来调整模型参数$\theta$。


## NUMERICAL RESULTS 部分分析

---

###  一、实验设计与数据集

#### 1. **三维环境数据（3D Environmental Field Data）**

- **来源与内容**：实验数据取自南海，包含温度、盐度、地形（bathymetry）等信息。
    
- **采集方式**：2020年6月27日，每小时采集一次，选择了4个具有类似地形的圆形区域作为实验对象。
    
- **数据处理**：每个区域沿36个不同方位（bearings）采集环境场数据。每组数据通过温度、盐度计算SSP（声速剖面），再用RAM数值模型计算TL（传输损失）。
    
- **输入输出**：模型输入为SSP和地形信息，输出为TL。数据总量3456组。对于超出地形深度的区域，声速被设为1700 m/s（泥沙层声速）。
    

---

#### 2. **性能评估指标**

- **主要指标**：RMSE（均方根误差），公式如下：
    
    $$\mathrm{RMSE} = \sqrt{\frac{1}{Q} \| \hat{TL} - TL \|_F^2 }$$
    - $^\hat{TL}$：模型预测结果，TL：RAM结果，Q：TL数据总数。
        
- **硬件环境**：Intel i9-5.8GHz + NVIDIA 4090 GPU。
    

---

### 二、与基线方法对比（B. Comparison with Baselines）

#### 1. **比较对象**

- **OFormer**：基于Transformer的运算符学习方法。
    
- **Vanilla FNO**：原始Fourier Neural Operator。
    
- **RAM**：传统数值解法，作为基准。
    

#### 2. **推理速度**

- RAM推理所有测试数据需115.33秒，OFormer需11.72秒，Hankel-FNO需1.97秒，Vanilla FNO需1.38秒。
    
- **结论**：FNO类方法（尤其Hankel-FNO）推理速度远快于传统方法。Hankel-FNO比Vanilla FNO略慢，但准确率更高。
    

#### 3. **精度与视觉效果**

- 图4展示了各方法预测的TL分布与RMSE值。
    
    - **OFormer** 只能捕捉主要模式，细节欠缺。
        
    - **FNO类方法** 细节与真实结果接近。
        
- **消融实验（ablation study）**表明，Hankel函数编码提升了长距离推理的精度，尤其在中长距离范围内（见图5、图6、图7）。
    

---

### 三、迁移学习能力（C. Transfer Learning）

#### 1. **不同源频率**

- 比较了100Hz（低频）和300Hz（高频）两种源频率的迁移性能。
    
- **结论**：不微调时误差大（RMSE高），仅用少量新数据微调后，模型能快速适应新频率。
    
- 低频场景模型适应更快（变化更平滑）。
    

#### 2. **不同源深度**

- 对比了100米和200米源深度的适应性。
    
- **结论**：模型能快速迁移到新深度，源位置变化越大，迁移难度略增。
    

#### 3. **不同地形**

- 考察地形更深且变化更剧烈的场景。
    
- **结论**：地形变化时模型适应性最好，训练样本数越多，迁移误差越低。
    

#### 4. **微调时间**

- 微调50、100、200、300组数据，所需时间分别为48s、87s、160s、244s，**迁移效率高且可控**。
    

---

### 四、图表与可视化亮点

- **推理速度柱状图（Fig.3）**直观对比各法速度差异。
    
- **TL分布与误差图（Fig.4-7）**展现模型在空间和距离上的预测精度及物理一致性。
    
- **迁移学习表格**，量化样本量与精度提升关系。
    

---

###  总体评价与物理意义

- **Hankel-FNO** 兼具高效推理和高精度，尤其在长距离声传播场景下表现优异，**融合物理先验显著提升泛化能力**。
    
- 与传统方法相比，速度提升数十倍以上，且能适应任意分辨率输入，支持实时大规模声场图谱应用。
    
- 迁移学习策略使得模型在新环境下用极少样本即可快速适应，实用性强。
    
