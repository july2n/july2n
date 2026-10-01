# July2n

### 聚焦大模型推理系统与 GPU 算子优化

**AI Infra · LLM Inference · GPU Kernels**

电子科技大学硕士 · 2027 届 · 成都<br />
[GitHub](https://github.com/july2n) · [知乎](https://www.zhihu.com/people/lang-zi-13-92-35/posts) · [Email](mailto:1344242662@qq.com)

## 关于我

围绕大模型推理与 GPU 计算做工程实践，关注从框架执行路径到算子实现的性能优化，也在知乎记录算法原理、源码阅读与实验心得。

- **推理系统 · Inference Systems** — 请求调度、连续批处理、分块预填充、KV / Prefix Cache 与 CUDA Graph。
- **混合模型 · Hybrid Models** — Full Attention 与 Gated DeltaNet 的状态管理、投机解码验证与状态恢复。
- **算子优化 · GPU Optimization** — GEMM / FlashAttention、Tensor Core、共享内存布局、访存与计算流水线。

## 项目实践

### [HybridInfer ↗](https://github.com/july2n/HybridInfer)

**面向混合注意力模型的轻量级 LLM 推理引擎**<br />
`Python` `PyTorch` `Triton` `CUDA Graph`

基于 nano-vllm 扩展，支持 Qwen3.5 Dense 纯文本推理，协调 Full Attention 的分页 KV Cache 与 Gated DeltaNet 的递归状态。

- **调度与执行**：连续批处理、分块预填充、预填充 / 解码混合执行与异步输出。
- **历史状态复用**：请求级状态池、块对齐前缀快照、解码图与分段预填充图。
- **推理加速探索**：Triton 算子优化，接入 n-gram、MTP、EAGLE-3、DFlash / DSpark 投机路径。

### [FlashAttention Ampere Lab ↗](https://github.com/july2n/flash-attention-ampere-lab)

**面向 RTX 3060 Ti 的 FlashAttention 内核实践**<br />
`C++` `CUDA` `Tensor Core` `Benchmark`

基于 Flash Attention from Scratch 改编，围绕 Ampere SM 8.6 实践七阶段优化，并与 PyTorch Flash-SDPA 对照测量。

- **融合计算**：QK、Online Softmax 与 PV 融合，支持 FP16 / BF16 forward。
- **内核优化**：共享内存 Swizzle、异步搬运、寄存器双缓冲与 Tensor Core 流水线。
- **实验验证**：设备自动调优、数值正确性对照、原始性能报告与可复现图表。

#### 更多实践

| 项目 | 内容 |
| :--- | :--- |
| [**CUDA Notes**](https://github.com/july2n/CUDA_Notes) | 从 Elementwise、Reduce、GEMM 到 WMMA、CUTLASS / CuTe，实践分块、向量化访存与多阶段流水线。 |
| [**Mini Nanotron**](https://github.com/july2n/Mini_nanotron) | 分布式并行机制的教学实现：DP 梯度同步、TP 行列并行、PP 层切分与组合式 Wrapper。 |

## 技术栈

**编程与模型实现**<br />
`C++` `Python` `PyTorch`

**GPU 内核与推理实践**<br />
`CUDA` `Triton` `Tensor Core` `CUTLASS` `CuTe` `CUDA Graph`

**框架源码与性能分析**<br />
学习 vLLM、SGLang、DeepSpeed 的执行路径与缓存 / 并行机制；结合算子 Benchmark、数值对照和设备调优验证实现。

## 技术写作

在知乎整理从原理到实现的学习笔记，与项目实践相互印证。

**推理系统与模型适配**

- [浅析推理框架如何适配 Hybrid Model](https://zhuanlan.zhihu.com/p/2088054724482417459)
- [vLLM ModelRunner V2：从请求状态到 GPU 执行](https://zhuanlan.zhihu.com/p/2087297945079230966)
- [AI Infra 面试常考—投机采样系列（EAGLE）](https://zhuanlan.zhihu.com/p/2016823705112716509)

**GPU 算子与训练机制**

- [AI Infra 面试常考—FlashAttention 系列](https://zhuanlan.zhihu.com/p/2015196808893192187)
- [沉淀篇—FlashAttention-2 MMA 优化](https://zhuanlan.zhihu.com/p/2030802121566761319)
- [从算法到源码：DeepSpeed ZeRO-3 的参数切分、同步与通信](https://zhuanlan.zhihu.com/p/2078058952940639868)

[更多知乎文章 ↗](https://www.zhihu.com/people/lang-zi-13-92-35/posts)

---

欢迎交流推理系统与 GPU 优化实践，也关注 **2027 届 AI Infra 开发机会**。<br />
联系我：[1344242662@qq.com](mailto:1344242662@qq.com)
