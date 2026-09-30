# SigLIP 论文深度解读

📄 [Sigmoid Loss for Language Image Pre-Training](https://www.aminer.cn/pub/64225b7e90e50fcafde150b8)（ICCV 2023，Google DeepMind，Zhai, Mustafa, Kolesnikov, Beyer，被引 3100+）

---

## 一、解决了什么问题

CLIP 式语言-图像预训练长期被一个看似不起眼的计算瓶颈卡住：**softmax 对比损失的“全局视角”依赖**。

1. **Softmax 必须看全批**：CLIP 的 InfoNCE 损失要对 N×N 相似度矩阵的每一行做 softmax 归一化——**每个设备的输出必须聚合到全局，再广播回来**，通信开销巨大；
2. **batch size 与硬件强绑定**：这导致训练必须用超大规模 TPU 集群（CLIP/OpenCLIP 都用了数百至上千个加速器），中小实验室根本玩不起；而且 batch size 成了和损失函数纠缠的超参；
3. **小 batch 下表现差**：softmax 损失在 batch 小时负样本少、性能明显下降。

一句话：**能不能设计一个只依赖 image-text 对本身、不需要全局归一化的对比损失？**

## 二、提出的方案：成对 Sigmoid 损失

把每对 (图像 i, 文本 j) 的对比变成了**独立的二分类问题**：

```
L = -1/|B| Σ_{i,j∈B} log σ(z · logit(x_i, t_j))

其中：
  z = -1  若 i = j（正对，要 logit 高）
  z = +1  若 i ≠ j（负对，要 logit 低）
  另加可学习的偏置 bias（补偿正负对 1:(N-1) 的不平衡）
```

关键性质：

- **只看单对**：不需要该行其他对的值做归一化，彻底解除了对全局批的依赖；
- **可学习偏置**：负对远多于正对（1:16k），偏置自动校准不平衡；
- **通信开销骤降**：只需交换相似度矩阵数值，不必全局 softmax 同步；
- **附带红利**：正负对比例成为可独立研究的变量（论文借此做了负样本比例的系统实验）。

## 三、网络结构

**架构与 CLIP 完全一致——这篇论文的贡献几乎全部在损失函数上**：

```
图像 ──► ViT 视觉编码器 ──► 全局嵌入 ─┐
                                      ├─► logit 矩阵 (N×N) ──► 成对 sigmoid 损失
文本 ──► 文本编码器 ──────► 文本嵌入 ─┘
                              （学到的温度 τ + 可学习偏置 b）
```

| 组件 | 说明 |
|------|------|
| 视觉编码器 | ViT-B/L 等标准配置（也可用 LiT 方案：冻结的预训练 ViT + 可学文本塔） |
| 文本编码器 | 标准 Transformer |
| 相似度计算 | 全局 CLS 嵌入内积 × 温度 + 偏置 |
| **损失（核心）** | **逐对 sigmoid 二分类**，替代 softmax 行归一化 |
| 训练技巧 | AdaFactor、降低 β₂ 稳定大 batch、预训练视觉塔冻结权重不施加 weight decay |

配套实验设置的三条线：
- **SigLIP**：从零联合训练双塔；
- **SigLiT**：LiT 配置（冻结视觉塔 + 训练文本塔），用于极限 batch 规模实验；
- **mSigLIP**：100+ 语言的多语言版本。

## 四、实验结果

![batch size 的影响：sigmoid 损失在小 batch 下显著优于 softmax，大 batch 下两者趋同且都在 ~32k 饱和，极限可训到 1M batch](https://open.aminer.cn/hub/amglm_noauth/rag/paper/image/69a1e47fe4a34d7fc064cb83)

### 1. 主结果：效率碾压

- 仅用 **4 块 TPUv4**，两天训出 **SigLiT-L：ImageNet 零样本 84.5%**——同等结果此前需要数百块加速器；
- Base 级 CLIP 4k batch、Large 级 LiT 20k batch 都能在 4 块芯片上训练；
- **batch size 解耦**：sigmoid 在小 batch（如 1k–4k）下明显优于 softmax；大 batch 下两者持平。

### 2. 反直觉发现：batch 不是越大越好

- 把 batch 推到极限 **100 万**：收益迅速递减，**32k 左右即饱和**，继续增大反而有害；
- 结论：CLIP 时代“batch 越大越好”的信仰被纠正——sigmoid 损失 + 32k batch 是性价比最优解。

### 3. 零样本性能

- **同等 batch 与训练量下，SigLIP 在 ImageNet 零样本上一致优于 CLIP**（如 ViT-L/16 +40k batch：SigLIP 80.5% vs CLIP 78.7%，训练量仅为 CLIP 论文的 1/5 左右就达到持平）；
- 多语言 mSigLIP 在 36 语种跨模态检索（Crossmodal-3600）上表现出色；
- 附加发现：sigmoid 训练对**数据噪声更鲁棒**（文本错配等破坏下性能衰减更慢）。

### 4. 消融要点

- 预训练视觉塔上禁用 weight decay 是 LiT 稳定训练的关键；
- 大 batch 下的梯度尖峰通过降低优化器 β₂ 动量解决；
- 负样本 1:16k 到近 1:1 的不平衡**并不致命**，可学习偏置足以补偿。

## 五、意义与后续

**意义**：
1. 把语言-图像预训练的**算力门槛降了一个数量级**，中小团队也能训 CLIP 级模型；
2. “逐对 sigmoid 替代 softmax 对比”的思想影响深远，此后大量图文模型改用 sigmoid 损失；
3. 确立了 **32k batch 饱和点**的经验结论，成为后续大模型图文训练的默认配置参考。

**后续演进与生态**：

| 方向 | 代表工作 |
|------|---------|
| **官方续作** | SigLIP 2（2025）：加入语义、定位、全局-局部等辅助目标，密集任务性能大幅提升 |
| **VLA / 机器人** | 你之前看的 [OpenVLA](https://www.aminer.cn/pub/666ba41e01d2a3fbfc11b57b) 正是用 **SigLIP 做语义视觉编码器**（与 DINOv2 空间特征互补）；RT-2 的继任者、众多 VLM（PaliGemma 系）也都以 SigLIP 为视觉塔 |
| **多语言** | mSigLIP 直接影响了多语言图文检索与多语种 VLM |

**一句话总结**：SigLIP 用一个“把 softmax 对比换成逐对 sigmoid 二分类”的极简改动，解除了图文预训练对全局 batch 的依赖——4 块 TPU 达到此前数百块芯片的效果，性能还略胜 CLIP。它是继 CLIP 之后图文对比学习最实用的改进，也因被 OpenVLA、PaliGemma 等采用而成为具身智能视觉栈的关键组件。

至此你已经集齐了 OpenVLA 视觉编码器的完整技术谱系：**DINO → DINOv2/DINOv3（空间敏感特征）+ SigLIP（语言对齐特征）→ 双编码器融合 → OpenVLA**。需要的话我可以把这条“视觉特征如何进入机器人大脑”的线索整理成一张对比表。