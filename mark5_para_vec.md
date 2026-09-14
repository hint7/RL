**1.** 输入：初始策略网络参数 $\theta_0$，初始价值网络参数 $\omega_0$；并行环境数量 $N=\text{num\_envs}$，每次 rollout（固定交互采样） 的时间步长 $M=\text{n\_steps}$，故每轮采样总样本量（batch size）$B=N\times M$。

**2.** 创建向量化并行环境 `vec_env`，一次 `step` 同时推进 $N$ 个子环境，得到按环境维度堆叠的 $s_t^{(i)},a_t^{(i)},r_{t+1}^{(i)},\text{done}_t^{(i)}$（$i=1,\ldots,N$）。

**3.** `for` $k = 0, 1, 2, \ldots$（第 $k$ 次策略更新）`do`

**4.** 用当前策略 $\pi(\theta_k)$ 在 `vec_env` 中连续采样 $M$ 步。每一步都把 $N$ 个环境的数据写入 rollout buffer，缓冲区张量形状常为：

- `obs`: $[M,N,\text{obs\_dim}]$
- `actions`: $[M,N,\text{act\_dim}]$
- `rewards`: $[M,N]$
- `dones`: $[M,N]$
- `log_probs_old`: $[M,N]$
- `values_old`: $[M,N]$

注意：若某个子环境 `done=True`，`vec_env` 会只重置该子环境并继续并行采样，不影响其他环境。

**5.** 基于第 4 步缓存的 `values_old`，对每个环境轨迹分别做 GAE，得到优势 $\hat{A}_{t}^{(i)}$：

$$
\hat{A}_{t}^{(i)}=\delta_{t}^{(i)}+\gamma\lambda(1-\text{done}_t^{(i)})\hat{A}_{t+1}^{(i)},
\quad
\delta_t^{(i)}=r_{t+1}^{(i)}+\gamma(1-\text{done}_t^{(i)})v(s_{t+1}^{(i)};\omega_k)-v(s_t^{(i)};\omega_k).
$$

若 `done=True`，下一状态价值项被 mask 成 0（即 bootstrap 截断）。

**6.** 计算目标价值（return / value target）：

$$
\hat{R}_{t}^{(i)}=\hat{A}_{t}^{(i)}+v(s_t^{(i)};\omega_k).
$$

于是得到 `advantage` 与 `value_target` 两个 $[M,N]$ 张量。


**7.** 一旦上一步完成，接下来的更新就完全不依赖时序关系了，所以数据【obs，action，reward，log_prob，old_value】可以打乱，即使来自不同的环境的数据也能进入一个 minibatch。具体来说，将 rollout buffer 从 $[M,N,\cdots]$ 展平为一个大 batch：$[B,\cdots]$（其中 $B=M\times N$）。  
例如 `obs_flat:[B,obs_dim]`、`actions_flat:[B,act_dim]`、`adv_flat:[B]`、`logp_old_flat:[B]`。


**8.** `for` `epoch` `in` `epochs` `do`

- 对 index $[0,\ldots,B-1]$ 打乱。
- 按 `minibatch_size` 切分成若干 **minibatch**（每个小批大小为 $b$，通常 $b \ll B$）。
- 每个 minibatch 单独前向、反向、更新参数；遍历完所有 minibatch 记为 1 个 epoch。

**9.** 在每个 minibatch 上最大化 PPO-Clip 策略目标：

$$
L_{\pi}(\theta)=
\frac{1}{b}\sum_{j=1}^{b}
\min\left(
\rho_j(\theta)\hat{A}_j,\;
(\;1 + sgn(\hat{A}_j) \cdot \epsilon\;)\hat{A}_j
\right),
$$


**10.** 在每个 minibatch 上最小化价值损失（可配 value clipping）：

$$
L_V(\omega)=\frac{1}{b}\sum_{j=1}^{b}\left(v(s_j;\omega)-\hat{R}_j\right)^2.
$$

最小化通常用 Adam。实践中常把策略损失、价值损失和熵正则合并为总损失一起优化。

**11.** `end for` `epoch`

**12.** `end for` $k$

---

补充理解（并行 PPO 的关键）：

- `vec_env` 提高的是“采样吞吐”，不是单样本学习规则本身。
- `batch = M\times N`：由“时间维长度 $\times$ 并行环境数”共同决定。
- `minibatch`：是优化器每次更新时从 batch 中抽取的小块数据。
- 常见关系：固定 `minibatch_size` 时，增大 `num_envs` 会增大 batch，总的每轮更新步数（minibatch 个数）也会变化。

---

### 注：向量化并行环境

「并行」和「向量化」是两件事，只是常被放在一起说。

**并行**：$N$ 份环境实例一起参与采样。是否在**同一物理时刻**真正同时执行，取决于底层实现，见下文。

**向量化**：数据和控制流按带 batch 维的张量组织。这里的「向量化」不是线性代数里的向量，也不完全是 CPU 的 SIMD，而是：**把 $N$ 个环境的数据堆成带 batch 维的张量，用一次 API、一次前向传播处理整批。**

#### 对比：非向量化 vs 向量化

非向量化（一次只和一个环境打交道）：

```python
state = env.reset()           # 一个 state，shape 比如 (4,)
action = policy(state)        # 一个 action
next_state, r, done, _ = env.step(action)  # 标量 reward、一个 bool
```

向量化（一次和 $N$ 个环境打交道，接口和数据形状都变了）：

```python
obs = vec_env.reset()         # shape: [N, obs_dim]
actions = policy(obs)         # shape: [N, act_dim]，一次前向算 N 个动作
obs, rewards, dones, _ = vec_env.step(actions)
# rewards: [N], dones: [N]
```

| | 单环境 | 向量化 |
| --- | --- | --- |
| obs | 一条样本 | N 条样本堆成矩阵 [N, obs_dim] |
| step 调用 | 每个 env 各调一次 | 一次 step 推进 N 个 env |
| 策略网络 | 输入 [obs_dim] | 输入 [N, obs_dim]，batch 推理 |
| buffer | 一条条 append | 直接写 [M, N, ...] 张量 |

#### 为什么要叫「向量化」

名字来自 NumPy/PyTorch 的习惯：把 $N$ 个标量/小数组 stack 成一行或一列，用矩阵运算代替 `for i in range(N)` 循环。

```python
# 非向量化：循环 N 次
for i in range(N):
    a[i] = policy(obs[i])

# 向量化：一次算完
actions = policy(obs_batch)   # obs_batch: [N, obs_dim]
```

GPU 上 batch 越大，利用率通常越高——这是 PPO 采样快的重要原因。

#### 「并行」和「向量化」的关系

- **并行** = $N$ 份环境一起采数据
- **向量化** = 这 $N$ 份数据不拆成 $N$ 次标量调用，而是堆成 [N, ...]（再和 $M$ 步合成 [M, N, ...]），一次 step、一次网络前向处理整批

二者可以独立存在：

| | 无并行 | CPU 多进程 | GPU 仿真并行 |
| --- | --- | --- | --- |
| 无向量化 | 动手学 RL（单环境） | —（A3C 等分布式 RL，非 PPO vec_env 路线） | —（GPU 仿真天然返回 batch 张量） |
| 向量化 | **环境侧**：Gymnasium SyncVectorEnv / SB3 DummyVecEnv / tianshou DummyVectorEnv（**单线程串行** step；API 批量，可以有 num_envs>1）<br/>**算法侧**：SB3 / tianshou / skrl 消费 batch 数据 | **环境侧**：Gymnasium AsyncVectorEnv / SB3 SubprocVecEnv / tianshou SubprocVectorEnv（**多核：真并行**；**单核：时间片并发，非真并行**）<br/>**算法侧**：SB3 / tianshou / skrl 消费 batch 数据 | **环境侧**：Isaac Gym / Isaac Lab / mjlab 等（GPU **真并行** + batch 张量由仿真器提供）<br/>**算法侧**：rsl_rl / skrl 消费 batch 数据

**小结**：向量化强调「batch 形状 + batch 接口 + batch 计算」；并行强调多个环境一起在跑。

#### 并行底层原理

从「串行」、「并发」到「真并行」：

| 程度 | 含义 | 典型例子 |
| --- | --- | --- |
| **串行** | 同一时刻只推进 1 个 env | SB3 DummyVecEnv 内部 for 循环 |
| **并发** | 逻辑上「同时在跑」，物理上交替执行 | 单核 CPU + 多进程（OS 时间片轮转） |
| **真并行** | 同一时刻多份计算单元各算各的 | 多核 CPU + 多进程；GPU SIMT batch 仿真 |

| | CPU 多进程 | CPU 多线程 | GPU 并行（SIMT 模型，CUDA 编程） |
| --- | --- | --- | --- |
| 并行单位 | N 个 **进程**（各自独立程序） | N 个 **线程**（同一进程内） | N 个 **GPU 线程**（kernel 内，按 warp 编组） |
| 规模 | 几个～几十个 | 几个～几十个 | 几千～几万个 |
| 是否真并行 | **单核**：否，OS 时间片**并发**（同一时刻只跑 1 个进程）<br/>**多核**：是，最多同时跑「核数」个进程 | **单核**：通常并发；CPU 密集还受 GIL 等限制<br/>**多核**：可部分真并行 | **是**，硬件 SIMT 大规模数据并行 |
| 执行独立性 | 完全独立，各自地址空间，可跑不同代码 | 较独立，可分支；共享地址空间 | 同一 warp 内必须走同一指令（分支 diverge 会变慢） |
| 内存与数据 | 进程间内存隔离；靠 IPC / pickle / 共享内存传数据，再 stack 成 batch | 共享堆；各自栈；同步（锁等）较复杂 | 状态排成连续 **显存** 张量 `[num_envs, ...]`（SoA），访存合并（coalesced） |
| 调度 | **OS** 调度进程（时间片轮转或绑核） | **OS** 调度线程 | **GPU 硬件** 调度 warp，不经 OS |
| 适合 | 多核 CPU 上跑多个 gym 环境；进程隔离（一个 env 崩不影响别的） | 任务杂、分支多、I/O 多；RL vec_env 中较少作为主方案 | **同一公式重复算很多遍**（物理积分、碰撞、矩阵乘） |
| RL 中的例子 | SB3 SubprocVecEnv / Gymnasium AsyncVectorEnv / tianshou SubprocVectorEnv | — | Isaac Gym / Isaac Lab / mjlab；底层 PhysX GPU + CUDA kernel |

**SIMT**（Single Instruction, Multiple Threads）是 GPU 的线程执行模型；**CUDA** 是 NVIDIA 提供的、在该模型上编写/launch kernel 的编程平台。二者是「模型 + 工具」关系。
