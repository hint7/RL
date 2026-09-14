**1.** 输入：初始策略网络参数 $\theta_0$，初始价值网络参数 $\omega_0$。

**2.** `for` $k = 0, 1, 2, \ldots$（第 $k$ 轮数据rollout[交互采样] = 第 $k$ 局 episode）`do`

**3.** 通过在环境中运行策略 $\pi(\theta_k)$，收集一条轨迹 $\tau=(s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T)$。把每一时间步 $(s_t, a_t, r_{t+1})$ 对应的 $[s_t, a_t, r_{t+1}, s_{t+1}, \log\pi_{\theta_k}(a_t\mid s_t), v(s_t;\omega_k), \text{done}]$ 写进 rollout buffer（轨迹缓冲区）。

**4.** 基于当前价值网络 $v(s_t;\omega_k)$，计算优势估计 $\hat{A}_t$（使用 GAE 估计方法）：

$$
\hat{A}_t^{\text{GAE}}=\delta_t^V+\gamma\lambda\hat{A}_{t+1}^{\text{GAE}}=
 r_{t+1}+\gamma v(s_{t+1};\omega_k)-v(s_t;\omega_k)+\gamma \lambda \hat{A}_{t+1}^{\text{GAE}}.
$$

其中令 $v(s_{t+1};\omega_k) = 0$（若 $\text{done} = \text{True}$）。并令 $\hat{A}_{T}^{\text{GAE}}=0$，反向遍历，得到优势估计列表 $\text{advantage-list}:[\hat{A}_0^{\text{GAE}},\hat{A}_1^{\text{GAE}},\ldots,\hat{A}_{T-1}^{\text{GAE}}]$。

**5.** 计算回报 $\hat{R}_t$。注意这里的 $\hat{R}_t$ 是用于价值网络回归的目标；不是理论定义的随机回报 $R_t = r_t+\gamma R_{t+1}$ 。回想A2C使用单步TD target构造价值目标：$\hat{R}_t=r_{t+1}+\gamma v(s_{t+1};\omega_k)$，偏差太大；现在有了GAE方法估算优势A，又优势定义为A=Q-V，沿着采样轨迹，把 Q对应成“从 (t) 开始的回报目标” $\hat R_t$ ，就可以通过第 4 步得到的 $\hat{A}_t^{\text{GAE}}$ 计算 $\hat{R}_t=\hat{A}_t^{\text{GAE}}+v(s_t;\omega_k)$，得到 $\text{value-list}:[\hat{R}_0,\hat{R}_1,\ldots,\hat{R}_{T-1}]$。

**6.** `for` `epoch` `in` `epochs` `do`

- 注意：下面 7、8 两步通常对同一组 rollout buffer 里的数据重复 `epochs` 次梯度更新。
- 下面的 $\arg\max$ / $\arg\min$ 只是优化目标，一次梯度步无法完全达到该极值。

**7.** 通过最大化 PPO-Clip 目标更新策略：

$
\begin{aligned}
&\theta_{k+1} = \arg\max_\theta \frac{1}{T} \sum_{t=0}^{T-1} \min \left(
  \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_k}(a_t\mid s_t)} \hat{A}_t^{\text{GAE}},\ 
  (1\pm\epsilon) \hat{A}_t^{\text{GAE}}\right) 
  = \arg\max_\theta \frac{1}{T} \Big\{\\[0.5em]
& 
  \min \left( \frac{\pi_\theta(a_0\mid s_0)}{\pi_{\theta_k}(a_0\mid s_0)} \hat{A}_0^{\text{GAE}},\ 
  (1\pm\epsilon) \hat{A}_0^{\text{GAE}}\right) + \min \left( \frac{\pi_\theta(a_1\mid s_1)}{\pi_{\theta_k}(a_1\mid s_1)} \hat{A}_1^{\text{GAE}},\ 
  (1\pm\epsilon) \hat{A}_1^{\text{GAE}}\right) + \cdots \Big\}
\end{aligned}
$

式中 $(1\pm\epsilon)$ 的正负号与 $\hat{A}_t^{\text{GAE}}$ 同号。通常把 $\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_k}(a_t\mid s_t)}$ 恒等变换为 $\exp\bigl(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_k}(a_t\mid s_t)\bigr)$。log 能保证数值稳定，避免太接近 0 的概率被当做 0 处理。

行为策略（采样时的策略）即 rollout 时的 $\pi_{\theta_k}$，其 $\log\pi_{\theta_k}(a_t\mid s_t)$ 缓存在 rollout buffer 里，在整个多 epoch 更新中不变（ “old log prob”）。当前要优化的 $\pi_\theta$ 在每次参数更新后都会变，比值 $\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_k}(a_t\mid s_t)}$ 在 epoch 之间会变化。

最大化用带 Adam 的随机梯度上升实现；算一次梯度要对 $T$ 个 $\min$ 式各自选起作用的分支再反向传播求梯度。若选到了被clip的分支（A为正 $\rho$ 太大 或 A为负 $\rho$ 太小），对 $\theta$ 梯度为0；若选到了没被clip的分支，梯度为$\dfrac{\hat{A}_t^{\text{GAE}}}{\pi_{\theta_k}(a_t\mid s_t)} \nabla_{\theta} \pi_\theta(a_t\mid s_t)$ ，即：

$
\nabla_{\theta} min()=
\begin{cases} 
    &\qquad\qquad\qquad\qquad\qquad\qquad0,  \quad&if \; clip \\
    &\hat{A}_t^{\text{GAE}} \cdot exp\bigl(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_k}(a_t\mid s_t)\bigr) \cdot \nabla_{\theta} \log\pi_\theta(a_t\mid s_t),  \quad &else
\end{cases} 
$

**8.** 通过对均方误差回归拟合价值网络：

$$
\omega_{k+1} = \arg\min_\omega \frac{1}{T} \sum_{t=0}^{T-1} \left( v(s_t;\omega) - \hat{R}_t \right)^2
$$

最小化用梯度下降实现。

**9.** `end for` `epoch`

**10.** `end for` $k$


注：
回报的严格定义 与 实现中的lazy命名