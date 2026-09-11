**前言**

没有大模型的逐步指导，我应该永远学不会强化学习；然而大模型的发明部分依赖强化学习。这就引申出一个猜想，如果我的研究时间是无限的，但我不能借助外部工具，我就发明不了大模型，也发明不了强化学习，这殊为遗憾。就像在实数轴上抽到有理数的概率为 0 ， 猴子胡乱敲键盘敲出莎士比亚集的概率比我研究明白强化学习的概率高。

**前言 2**

目前用的最多的强化学习算法是 PPO，然而 PPO 提出九年了，市面上能讲清楚 PPO 的教材和博客实在不多。本篇笔记综合多本教材和博客 [ 见文末 ]，以最短路径直入 PPO，即便如此，因为前置知识的需要，也只能将 PPO 排到 2.2.5 节叙述。

# 1、基本概念
## 1.1、 学习过程
在典型的强化学习框架$`^{(注1)}`$中，智能体通过持续与环境交互来学习，它首先观察当前状态$`s_t`$，依据其策略$`\pi`$选择动作$`a_t`$，从环境中接收奖励$`r_{t+1}`$，随后转移到下一状态，直到该局(episode)结束T， 有序列：
$$
s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T
$$
智能体从一次次的交互中学习到关于环境的知识并更新策略$`\pi`$，如图所示，其中<font style="color:#ECAA04;">黄色线条</font>表示<font style="color:#ECAA04;">一次学习步骤</font>的过程。<font style="color:#DF2A3F;">强化学习的目标即，找到使得奖励</font>$`func(r_1,r_2,...,r_T)`$<font style="color:#DF2A3F;"> 最大的策略, </font>这里的函数 $`func`$不是简单的加和，不同的方法可能会设定不同的目标函数 $`func`$，但都离不开<font style="color:#DF2A3F;">加权求和再求期望</font>的形式，见 1.2。

<!-- 这是一张图片，ocr 内容为：动作A 动作AT 策略元 奖励R 价值9 环境 智能体 T(ALS;0) A(S, A;W) 价值网络(CRITIC) 环境 策略网络(ACTOR) 奖励R TT+1 状态8T ST+1 状态 基础框架 (A) (B)ACTOR-CRITIC算法 -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773143518222-39955bf1-d2a5-4d4d-b4dc-225ea7be6432.png)

 这种交互可以表述为一个马尔可夫决策过程$`<S,A,P,R>`$，包含状态转移模型$`P`$、状态空间$`S`$、动作空间$`A`$、奖励$`R`$。若智能体对环境进行建模，估计一个环境模型$`<\hat P,\hat R>`$，该学习称为有模型学习；而通常在现实情境中，智能体很难对$`P`$或$`R`$完全建模，智能体依靠采样的轨迹 通过蒙特卡洛法$`^{(注1-2)}`$学习，称为无模型学习。目前大多数 RL 算法都是无模型的，本文要介绍的算法也都是无模型的$`^{(注1-3)}`$。

当采用随机策略，定义策略函数$`\pi: S \times A \mapsto[0,1]`$为概率密度函数$`\pi(a \mid s)=\mathbb{P}(A=a \mid S=s)`$。当采用确定策略，定义策略函数$`\pi: S \mapsto A`$。接下来除了 DDPG 一小节，其余都采用随机策略。

## 1.2、 学习目标
### 1.2.1 回报
**<font style="color:#DF2A3F;">强化学习的目标是使总奖励尽可能大</font>**，如 2024 年图灵奖得主、强化学习之父 Sutton 在书中所言，强化学习就是学习“**做什么(即如何把当前的情境映射成动作)才能使得数值化的收益信号最大化**”，**<font style="color:#DF2A3F;">但这个总奖励怎么定义是有讲究的</font>**。

:::info
具体要优化的目标函数应该满足以下条件：

a） 能递归分解：递归结构能让长期优化问题变成可学习的局部更新问题。

b） 保证收敛性：目标函数不应随时间步增加发散至无穷，应保证学习过程能够收敛。

c） 有时间偏好：智能体更重视近期奖励更符合实际决策过程，但不能忽视远程延迟奖励。

:::

通过对未来奖励进行折扣，定义<font style="color:#DF2A3F;">回报 </font>$`G_t`$为随时间加权的奖励之和：

$$
\begin{aligned} G_t &= \sum_{i=t}^T \gamma^{(i-t)} R_{i+1} \\&= R_{t+1}+\gamma\;R_{t+2}+\gamma^2\;R_{t+3}+...+\gamma^{T-t}\;R_T\end{aligned}
$$
  
其中，$`\gamma`$是一个介于 0 和 1 之间的折扣因子，旨在平衡当前奖励与未来奖励的相对权重。$`\gamma`$越接近 1，越强调长期奖励，但设得过于接近 1 可能会影响训练性能。

强化学习一般不会把回报作为目标函数，而是下一节的价值函数。

### 1.2.2 平均回报: 三个价值函数
<font style="color:#DF2A3F;">在</font>$`t`$<font style="color:#DF2A3F;">时刻，回报 </font>$`G_t`$<font style="color:#DF2A3F;"> 是一个随机变量</font>$`^{(注2)}`$（下文大写的$`G,S,A`$都表示随机变量）。但智能体需要在 $`t`$时刻 预测 $`G_t`$ 的一个确定值，不然只有等一局结束才能更新策略。<font style="color:#DF2A3F;">此时， </font>$`G_t`$<font style="color:#DF2A3F;"> 取期望一方面可以消除其随机性，另一方面可以考虑到未来的延迟奖励。</font>

**（笔者觉得期望这个词不容易记忆，其实期望可以记作平均，一种按概率加权的平均）**

**————————————————————————————————————****—————————**

**（1）定义 ****<font style="color:#DF2A3F;">动作</font>****价值函数 **$`Q^\pi`$** 为**，

在观测到$`S_t=s_t, A_t=a_t`$的条件下，$`G_t`$相对于$`S_{t+1}, A_{t+1}, \cdots, S_T, A_T`$的期望，
$$
\begin{aligned}
Q^\pi(s_t, a_t)
&= \mathbb{E}\!\left[ G_t \mid S_t = s_t,\; A_t = a_t \right] \\
&= \mathbb{E}_{S_{t+1}, A_{t+1}, \cdots, S_T, A_T}\!\left[ G_t \mid S_t = s_t,\; A_t = a_t \right] \\
&= \mathbb{E}_{\substack{
S_{i} \sim P(\cdot \mid S_{i-1}, A_{i-1}),\\
A_i \sim \pi(\cdot \mid S_i),\; i \ge t+1
}}
\!\left[ G_t \mid S_t = s_t,\; A_t = a_t \right].
\end{aligned}
$$
$`Q^\pi`$依赖于$`s_t`$、$`a_t`$以及$`\pi`$，但不依赖于时间$`t+1`$及之后的状态和动作。

**<font style="color:#DF2A3F;">通俗来说，</font>**$`Q^{\pi}(s,a)`$**<font style="color:#DF2A3F;">就是 （某个策略）在状态 s下执行动作 a 的价值</font>**

**【**$`Q^\pi`$** 可用于估计：给定 策略 **$`\pi`$**和当前状态 **$`s_t`$**，每个动作 **$`a_t`$**的好坏。****<font style="color:#DF2A3F;">（“好”，指的是 使回报 </font>**$`G_t`$**<font style="color:#DF2A3F;">高。下同）</font>****】**

**————————————————————————————————————****—————————**

**（2）定义****<font style="color:#DF2A3F;"> 最优动作</font>****价值函数**$`Q^{\ast}`$**为**，

当有多种策略可供选择时，最大的动作价值函数，
$$
Q^{\ast}\left(s_t, a_t\right)=\max _\pi Q^\pi\left(s_t, a_t\right)
$$
$`Q^*`$依赖于$`s_t`$、$`a_t`$，它与$`\pi`$无关。

**通俗来说，**$`Q^{*}(s,a)`$**就是 任意策略在状态 s下执行动作 a 的最大价值**

** ****【**$`Q^*`$**可用于估计： 给定状态**$`s_t`$**，每个动作**$`a_t`$**的 好坏。】**

**<font style="color:#DF2A3F;">在强化学习中可指定目标函数为 </font>**$`Q^*`$**<font style="color:#DF2A3F;">，这称为价值学习。</font>**常见的价值学习算法有Q-learning、DQN等。

**—————————————————————————————————————————————**

**（3）定义 ****<font style="color:#DF2A3F;">状态</font>****价值函数**$`V^{\pi}`$**为**，

动作价值函数$`Q^\pi(s_t, a_t)`$相对于动作变量的期望，
$$
V^\pi\left(s_t\right)=\mathbb{E}_{A_t \sim \pi(\cdot|s_t)}\left[Q^\pi\left(s_t, A_t\right)\right]
$$
$`V^\pi`$依赖于$`s_t`$、$`\pi`$，与$`a_t`$无关。

**<font style="color:#DF2A3F;">通俗来说，</font>**$`V^\pi`$**<font style="color:#DF2A3F;">就是 状态s 的平均价值</font>**

**【**$`V^\pi`$** 可用于估计：给定策略 **$`\pi`$**，当前状态 **$`s_t`$**的好坏。】**

$`V^\pi`$也能写成：
$$
\begin{aligned}
V^\pi\left(s_t\right)

&= \mathbb{E}_{ A_{t},S_{t+1}, A_{t+1}, \cdots, S_T, A_T}\!\left[ G_t \mid S_t = s_t\; \right] \\

\end{aligned}
$$
进一步地，$`\mathbb{E}_S\left[V^\pi(S)\right]`$依赖于$`\pi`$，与$`s_t`$、$`a_t`$无关。

**【**$`\mathbb{E}_S\left[V^\pi(S)\right]`$**可用于估计：策略**$`\pi`$**的好坏。】**

**<font style="color:#DF2A3F;">在强化学习中可以指定目标函数为</font>**$`\mathbb{E}_S\left[V^\pi(S)\right]`$**<font style="color:#DF2A3F;">，这称为策略学习。</font>**常见的策略学习算法有原始 Actor-Critic（以及改进版 A2C,A3C），TRPO/PPO，DDPG 以及 SAC 等。

**—————————————————————————————————————————————**

强化学习其实就是要解决 的问题即  优化上述目标函数，<font style="color:#DF2A3F;">在足够理想的情况下，强该问题都能用传统优化算法解决</font>（如遗传算法、模拟退火等算法）$`^{(注3)}`$。

### 1.2.3 优势函数
Baird 在论文 _"Advantage Updating"_(1993)  中明确提出了“优势更新”的概念，这通常被认为是优势函数概念最早的显式表述之一。他指出了使用优势值【即$`Q(s,a)−V(s)`$】来更新策略比直接使用 Q 值更有效。

**优势函数**定义为：
$$
A^{\pi}(s,a)=Q^{\pi}(s,a)−V^{\pi}(s)
$$
+$`Q^{\pi}(s,a)`$：状态 s下执行动作 a 的价值
+$`V^{\pi}(s)`$：状态 s的平均价值
+ <font style="color:#DF2A3F;">注意结合上下文区分代表动作随机变量大写的 </font>$`A`$<font style="color:#DF2A3F;"> 和这里的优势函数</font>$`A^{\pi}`$

 优势 可以理解为 :                           **优势  =  这个动作的价值 − 该状态的平均价值**

****

### 1.2.4 价值函数间的关系: 贝尔曼方程
贝尔曼方程可以理解为:

**当前价值  =  即时奖励 + 未来折扣价值的期望**

根据回报的定义，不难验证贝尔曼递推关系：
$$
G_t = R_t+\gamma G_{t+1}
$$
从这个递推关系出发，结合价值函数定义，不难证明下面的几个贝尔曼方程：

**<font style="color:#DF2A3F;">贝尔曼期望方程</font>**（也可看做**<font style="color:#DF2A3F;">价值函数的递归定义</font>**）：

用$`Q^\pi`$表示$`Q^\pi`$

$`Q^\pi\left(s_t, a_t\right)=\mathbb{E}_{S_{t+1}, A_{t+1}}\left[R_t+\gamma \cdot Q^\pi\left(S_{t+1}, A_{t+1}\right) \mid S_t=s_t, A_t=a_t\right]`$.

用$`V^\pi`$表示$`Q^\pi`$
$$
Q^\pi\left(s_t, a_t\right)=\mathbb{E}_{S_{t+1}}\left[R_t+\gamma \cdot V^\pi\left(S_{t+1}\right) \mid S_t=s_t, A_t=a_t\right].
$$
用$`V^\pi`$表示$`V^\pi`$
$$
V^{\pi}(s_t) = \mathbb{E}_{A_t, S_{t+1}} \left[ R_t + \gamma \, V^{\pi}(S_{t+1}) \mid S_t = s_t \right]
$$
**<font style="color:#DF2A3F;">贝尔曼残差</font>**（在数学里，残差即：模型应该满足的关系 − 实际预测之间的差）   

 Bellman residual =$`R_t+γV^\pi(s_{t+1})−V^\pi(s_t)`$

**<font style="color:#DF2A3F;">贝尔曼期望方程推论：</font>**

用$`V^\pi`$表示$`A^\pi`$
$$
 A^\pi(s_t,a_t)
=
\mathbb{E}_{S_{t+1}}
\left[
R_t+\gamma V^\pi(S_{t+1})-V^\pi(s_t)
\mid S_t=s_t,A_t=a_t
\right] 
$$
用$`Q^\pi`$表示$`Q^\pi`$（多步版本）
$$
Q^\pi\left(s_t, a_t\right)=\mathbb{E}_{S_{t+1}, A_{t+1},...,S_{t+m}, A_{t+m}}\left[R_t+\gamma R_{t+1}+...+\gamma^{m-1}R_{t+m-1}+\gamma^{m} \cdot Q^\pi\left(S_{t+m}, A_{t+m}\right) \mid S_t=s_t, A_t=a_t\right]
$$
或者$`Q^\pi\left(s_t, a_t\right)=\mathbb{E}_{S_{t+1}, A_{t+1},...,S_{t+m}, A_{t+m}}\left[R_t+\gamma R_{t+1}+...+\gamma^{m-1}R_{t+m-1}+\gamma^{m} \cdot V^\pi\left(S_{t+m}\right) \mid S_t=s_t, A_t=a_t\right]`$

**<font style="color:#DF2A3F;">贝尔曼最优方程</font>**：
$$
Q^{*}\left(s_t, a_t\right)=\mathbb{E}_{S_{t+1} \sim p\left(\cdot \mid s_t, a_t\right)}\left[R_t+\gamma \cdot \max _{A \in \mathcal{A}} Q^{*}\left(S_{t+1}, A\right) \mid S_t=s_t, A_t=a_t\right]
$$
表示**最优策略的每一步之后仍然是最优策略。**



# 2、策略学习
策略学习的意思是通过求解一个优化问题，学出最优策略函数或它的近似函数（比如策略网络）。

强化学习中智能体观察状态$`s_t`$，根据策略$`\pi`$选择动作$`a_t`$。上面提到，<font style="color:#DF2A3F;">若采用随机策略，则策略是概率密度函数</font>$`\pi(a|s;\theta): S \times A \mapsto[0,1]`$，其中$`\theta`$是函数中的可优化参数。举个例子，把超级玛丽游戏当前屏幕上的画面作为观测$`s`$，训练时策略会输出每个动作的概率值：$`\pi(左|s)=0.5，\pi(右|s)=0.2,\pi(跳|s)=0.3`$，根据概率做抽样，得到一个动作$`a`$，让马里奥执行 ，执行后游戏环境返回$`s`$和$`r`$再输入给策略。

<font style="color:#DF2A3F;">怎么表示</font>$`\pi(a|s;\theta)`$<font style="color:#DF2A3F;">这个概率密度函数？</font>

（1）当问题足够简单，可以用数学表达式显式表示概率密度$`\pi(a|s;\theta)=function(a,s,\theta)`$;

（2） 当 a 离散,$`\pi(a|s;\theta): S \times A \mapsto[0,1]`$可以用二维表格表示;

（3）当 a 连续,  可用神经网络表示$`\pi(a|s;\theta)`$。先假设动作$`a`$是 服从 高斯分布$`\mathcal{N}(\mu, \sigma^2)`$（当然也可以是其他分布形式），只是分布具体参数未知。然后构造神经网络$`\pi_\theta`$,其中$`\theta`$是网络中待更新的参数。网络 输入是观测$`s`$,输出是 动作$`a`$的 分布参数$`\mu, \sigma`$。一般把这个神经网络称为<font style="color:#DF2A3F;">策略网络</font>。

<font style="color:#DF2A3F;">怎么优化</font>$`\pi(a|s;\theta)`$<font style="color:#DF2A3F;">这个概率密度函数？</font>

第一节提到，$`\mathbb{E}_S\left[V^\pi(S)\right]`$可用于估计策略$`\pi`$的好坏，所以可以把$`\mathbb{E}_S\left[V^\pi(S)\right]`$作为优化目标。对于用策略网络表示$`\pi(a|s;\theta)`$的情形，定义策略网络目标函数:
$$
J({\theta})=\mathbb{E}_S\left[V^\pi(S)\right]
$$
目标是使其最大化。

## 2.1、策略梯度
自然想到用**<font style="color:#DF2A3F;">梯度上升</font>**$`^{(注4-2)}`$**<font style="color:#DF2A3F;"> </font>**
$$
{\theta} \leftarrow {\theta}+\beta \cdot \nabla_\theta J(\theta)
$$
来**<font style="color:#DF2A3F;">最大化</font>**$`J({\theta})`$, 其中$`\beta`$为学习率。

### 2.1.1 策略梯度定理 
有**<font style="color:#DF2A3F;">策略梯度定理</font>**$`^{证明见(注5)}`$：
$$
\nabla_\theta J(\theta)=\mathbb{E}_S\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; {\theta})}\left[\nabla_\theta \ln \pi(A \mid S ; {\theta}) \cdot Q^\pi(S, A)\right]\right]
$$
严格来说等式右边应该乘以常数项$`(1-\gamma^n)/(1-\gamma)`$, 但这个常数项会被$`{\theta} \leftarrow {\theta}+\beta \cdot \nabla_\theta J(\theta)`$中的$`\beta`$吸收，所以这里从简不写。（这也是有些教材，如 [ 动手学 ] 将该定理中间 的 等号 写为$`\propto`$）

观察该式右边， 因为不知道$`S`$的概率密度，就算知道计算机通常也很难列出$`S`$去逐个计算，所以 需要做蒙特卡洛近似：

:::info
<!-- 这是一张图片，ocr 内容为：回忆一下,第2章介绍了期望的蒙特卡洛近似方法,可以将这种方法用于近似策略 梯度.每次从环境中观测到一个状态S,它相当于随机变量S的观测值.然后再根据当 前的策略网络(策略网络的参数必须是最新的)随机抽样得出一个动作: A ~ N( IS ; ). 计算随机梯度: :0) 三 QN(S,A)-VELNN(A|S;0). G(S,A;0) 很显然,G(S,0)是策略梯度 VES( ) ) - ES(EANN(1SSE)[9(S,A;E)]]]]]]. 于是我们得到下面的结论: 结论7.1 随机梯度G(S,A;0) 兰Q元(S,A)-VELMM(ALS:0)是策略梯度VEJ(0)的无偏估计. 应用上述结论,我们可以做随机梯度上升来更新0.使得目标函数:(0)逐渐增长: 010+8(S,A;0). 此处的B是学习率,需要手动调整.但是这种方法仍然不可行,我们计算不出9(S,0;0), 原因在于我们不知道动作价值函数Q(S,A).在后面两节中,我们用两种方法对Q, 做近似:一种方法是REINFORCE.用实际观测的回报U近似Q(S,A):另一种方法是 ACTOR-CRITIC,用神经网络Q(S,A;W)近似QN(S,A). -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773038712084-2371e783-6899-4b94-9c72-7a4049238da5.png)

:::

上述近似表明，
$$
\nabla_\theta J(\theta) \approx g(s,a;\theta)=\nabla_\theta \ln \pi(a \mid s ; {\theta}) \cdot Q^\pi(s, a)
$$
其中$`\nabla_\theta \ln \pi(a \mid s ; {\theta})`$可以通过策略网络反向传播$`^{(注5-0)}`$**<font style="color:#DF2A3F;"></font>**得到，可见<font style="color:#DF2A3F;">问题变成了如何近似</font>**<font style="color:#DF2A3F;">动作价值函数 </font>**$`Q^\pi`$

### 2.1.2 REINFORCE
上文提到$`\nabla_\theta J(\theta) \approx g(s,a;\theta)=\nabla_\theta \ln \pi(a \mid s ; {\theta}) \cdot Q^\pi(s, a)`$中需要估计$`Q^\pi`$的值。

Ronald Williams 在1992 年提出 REINFORCE 算法，用一个 episode 完成之后收集到的的<font style="color:#DF2A3F;"> T 个真实回报 的平均</font>$`\frac{1}{T}\sum_{t=1}^{T}   \tilde{G}_t`$来近似$`Q^\pi`$，其中$`\tilde{G}_t= \sum_{k=t}^{T} \gamma^{k-t} R_{k} = R_{t} + \gamma R_{t+1} + \gamma^2R_{t+2}+...+\gamma^{T}R_{T-t}，\forall \, t = 1, \cdots, T.`$

然后做一次梯度上升更新策略网络$`\pi`$的 参数。

完整流程如下：

:::info
1. 用策略网络$`\boldsymbol{\theta}_{\text{old}}`$控制智能体从头开始玩一局游戏，得到一条轨迹 (trajectory)：
$$
s_0, a_0, R_1, \quad s_2, a_2, R_3, \quad \cdots, \quad s_{T-1}, a_{T-1}, R_T.
$$
2. 计算所有 T 个回报：
$$
\tilde{G}_t = \sum_{k=t}^{n} \gamma^{k-t} \cdot R_k, \qquad \forall \, t = 1, \cdots, T.
$$
3. 用$`\{(s_t, a_t)\}_{t=1}^T`$作为数据，做反向传播计算：
$$
\nabla_{\boldsymbol{\theta}} \ln \pi(a_t \mid s_t; \boldsymbol{\theta}_{\text{old}}), \qquad \forall \, t = 1, \cdots, T.
$$
4. 做随机梯度上升更新策略网络参数：
$$
\boldsymbol{\theta}_{\text{new}} \leftarrow \boldsymbol{\theta}_{\text{old}} + \beta \cdot \frac{1}{T} \sum_{t=1}^{T}   \tilde{G}_t \cdot {\nabla_{\boldsymbol{\theta}} \ln \pi(a_t \mid s_t; \boldsymbol{\theta}_{\text{old}})}.
$$
:::

在第 4 步$`\frac{1}{T}\sum_{t=1}^{T}   \tilde{G}_t`$中，**整局游戏中所有时刻的经验累积起来**，形成一个综合的更新方向，告诉网络：“在这整局里，总体上哪些动作是好的，哪些是坏的”。$`\frac{1}{T}`$是求平均，但它能被学习率吸收掉，所以其实可以略去不写。

这种近似的 优点:无偏估计，因为$`Q^\pi`$的定义就是=$`\mathbb{E}[ \sum_{k=t}^{T-1} \gamma^{k-t} R_{k}]`$。缺点:1. 方差巨大,  涉及 T步的随机变量； 2. 必须等一段 episode 结束。

REINFORCE 算法论文第一次证明了策略梯度定理，以及带基线的策略梯度定理；并用到了Ronald以及 Hinton 等人在 1986 年提出的多层神经网络反向传播算法来计算$`\nabla_\theta \ln \pi(a \mid s ; {\theta})`$。REINFORCE算法名字直接是 Reinforcement Learning 的前几个字母大写。



### 2.1.3 时间差分 TD
**（1）概念**

REINFORCE 提供了一种$`\nabla_\theta J(\theta) \approx g(s,a;\theta)=\nabla_\theta \ln \pi(a \mid s ; {\theta}) \cdot Q^\pi(s, a)`$中估计$`Q^\pi`$值的方法，即蒙特卡洛采样估计。另一种方法即时间差分（Temporal Difference，TD）。

时间差分方法**用 （当前奖励 + 折扣因子× 未来价值估计） 去估计 当前价值**：

** 当前价值估计**$`\leftarrow`$** 当前奖励 + 折扣因子× 未来价值估计**

用右边去更新左边。折扣因子的存在是因为未来越远越不确定。这是一种**<font style="color:#DF2A3F;">自举</font>**方法$`^{(注5-1)}`$。

式子右边称为<font style="color:#DF2A3F;"> </font>**<font style="color:#DF2A3F;">TD 目标（ TD target）</font>**<font style="color:#DF2A3F;">，记为</font>$`\hat{y}`$；TD target和被估计量的原始值的差称为**<font style="color:#DF2A3F;"> TD 误差（ TD error</font>**<font style="color:#DF2A3F;">），记为</font>$`\delta`$。

假设玩游戏，要估计这一局最后能拿多少分。**蒙特卡洛法，**对这局游戏采样，等游戏结束得到一个样本（$`R_1，R_2，R_3， ... ，R_T`$）：总分 =$`R_1 + R_2 + R_3 + ... + R_T`$。 **TD 不用等结束**，比如现在到第 t 步：总分估计$`\leftarrow`$** **R_t + γ × 下一状态的总分估计。

**（2）时间差分的形式化定义**

**动作价值函数形式：**

TD 目标：$`\hat{y}_t = R_t + \gamma Q(s_{t+1}, a_{t+1})`$

TD 差分：$`\delta_t = R_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_{t},a_t)`$
$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \delta_t
$$
**状态价值函数形式：**

TD 目标：$`\hat{y}_t = R_t + \gamma V(s_{t+1})`$

TD 差分：$`\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)`$
$$
V(s_t) \leftarrow V(s_t) + \alpha \delta_t
$$
<font style="color:#DF2A3F;">注意 1、 这里的</font>$`Q、V`$<font style="color:#DF2A3F;">是对</font>$`Q^\pi、V^\pi`$<font style="color:#DF2A3F;">的估计，</font>$`Q \ne Q^\pi，V \ne V^\pi`$

<font style="color:#DF2A3F;">2、TD 目标中的 </font>$`Q(s_{t+1}, a_{t+1})`$<font style="color:#DF2A3F;">或者</font>$`V(s_{t+1})`$是通过函数逼近器（如表格、神经网络）得到的，这个逼近器的输入可能是$`t \in T`$的任一状态，所以表格会过于庞大，一般用神经网络，而神经网络的更新又借助 TD，这种自举的过程详见下一小节。

<font style="color:#DF2A3F;">3、一些本来就有学习率的场景</font>，如定义损失函数时候，往往直接用$`\hat{y}_t`$替代价值函数$`V(s_t) \leftarrow \hat{y}_t = R_t + \gamma V(s_{t+1})`$，而不是用$`V(s_t) \leftarrow V(s_t) + \alpha \delta_t`$。另外根据贝尔曼方程有$`Q^\pi\left(s_t, a_t\right)=\mathbb{E}_{S_{t+1}}\left[R_t+\gamma \cdot V^\pi\left(S_{t+1}\right) \mid S_t=s_t, A_t=a_t\right].`$所以有些时候用$`Q(s_t,a_t) \leftarrow R_t + \gamma V(s_{t+1})`$

<font style="color:#DF2A3F;">4、一些地方会把</font>$`R_t`$<font style="color:#DF2A3F;">写为</font>$`R_{t+1}`$，只是在时间步的解释上有一些区别，本质一样。需要注意：
$$
\begin{aligned}
&第一种约定在t时刻一次转移为(s_t, a_t, R_{t+1}, s_{t+1})， 则对应轨迹和时间差分\\&(s_0, a_0, R_1, s_1, a_1, R_2, \ldots, s_{T-1},a_{T-1}, R_T)\quad，\delta_t=R_{t+1}+\gamma V_{t+1}-V_t，\\
&第二种约定在t时刻一次转移为(s_t, a_t, R_{t}, s_{t+1})， 则对应轨迹和时间差分\\&(s_0, a_0, R_0, s_1, a_1, R_1, \ldots, s_{T-1}, a_{T-1}, R_{T-1})\quad，\delta_t=R_t+\gamma V_{t+1}-V_t
\end{aligned}
$$
本节**时间差分 TD** 都采用第二种约定，即$`\delta_t=R_t+\gamma V_{t+1}-V_t`$。本文其他部分则结合上下文判断。

**（3）现实例子**

现实中因为时间是连续的，不是离散时间步，所以要明确一个时间步是从哪到哪，在一个时间步 t 内能获得哪些信息：  
	例子 1。从A开车去C会路过B，约定时间步t=0为A出发之前至到达B之前，t=1为到达B之前至到达C之前，$`s_t`$为时间步t的路况，$`R_t`$为时间步t实际花费的小时数，在时间步t可知$`（s_t,R_t,s_{t+1}）`$。有模型 M 能根据$`S_t`$预估到C还要开车时间$`V_t`$。在时间步 t=0，在A出发时候模型 M 估计$`V_0`$=10 小时，实际到B之前花了$`R_0`$=6 个小时，模型 M 估计到C还需要$`V_1`$=2 个小时。设折扣因子为 1， 则TD target$`\hat{y}_0`$=6 + 1*2=8，这个值比最初预测的 10 更可靠，因为 8 中包含事实的部分。用 TD error 在时间步 t =0去更新模型，设学习率 α=0.5，更新：$`M \leftarrow M+ \alpha \, \delta`$,$`M = 10 + 0.5 \times (-2)`$=9，意思是：下次模型 M 查到$`S_0=s_0`$预测$`V_0`$=9。  
	例子 2。一辆大巴会经停始发站、A站、B站、终点站，事前不知道AB两站会上多少乘客。约定时间步t=0为S站出发后至A站出发后，t=1为A站出发后至B站出发后，$`s_t`$为时间步t的天气，$`R_t`$为时间步t实际上车的人数，在时间步t可知$`（s_t,R_t,s_{t+1}）`$。有模型 M 能根据天气$`S_t`$预测还会上$`V_t`$名乘客。 在时间步 t=0，M根据$`S_0=s_0`$预测还会上$`V_0`$=10 名乘客，大巴到达A站实际上了$`R_0`$=6 名，此时模型 M 查询天气$`S_1=s_1`$预计B还会上$`V_1`$=2名 。设折扣因子为 1，则TD target$`\hat{y}_0`$= 6 +1×2=8。这个 8 比最初预测的 10 更可靠，因为 8 中包含事实的部分。TD error$`\delta_0`$= 8-10=-2。用 TD error 在时间步 t 去更新模型，设学习率 α=0.5，更新：$`M \leftarrow M+ \alpha \, \delta`$,$`M = 10 + 0.5 \times (-2)`$=9，意思是：下次模型 M 查到$`S_0=s_0`$预测$`V_0`$=9。

**（4）TD 与贝尔曼方程**

根据贝尔曼残差的定义， Bellman residual =$`R_t+γV^\pi(s_{t+1})−V^\pi(s_t)`$。可知TD误差就是贝尔曼残差的样本形式，它刻画了当前样本没有满足贝尔曼等式的程度。贝尔曼方程$`V^{\pi}(s_t) = \mathbb{E}_{A_t, S_{t+1}} \left[ R_t + \gamma \, V^{\pi}(S_{t+1}) \mid S_t = s_t \right]`$**是“应然关系”**，**TD 是“用估计去逼近这个关系的方法”。**

回顾贝尔曼方程的多步版本：
$$
Q^\pi\left(s_t, a_t\right)=\mathbb{E}_{S_{t+1}, A_{t+1},...,S_{t+m}, A_{t+m}}\left[R_t+\gamma R_{t+1}+...+\gamma^{m-1}R_{t+m-1}+\gamma^{m} \cdot Q^\pi\left(S_{t+m}, A_{t+m}\right) \mid S_t=s_t, A_t=a_t\right]
$$
或者$`Q^\pi\left(s_t, a_t\right)=\mathbb{E}_{S_{t+1}, A_{t+1},...,S_{t+m}, A_{t+m}}\left[R_t+\gamma R_{t+1}+...+\gamma^{m-1}R_{t+m-1}+\gamma^{m} \cdot V^\pi\left(S_{t+m}\right) \mid S_t=s_t, A_t=a_t\right]`$

可以仿照该式写出<font style="color:#DF2A3F;"> n 步 TD 目标估计 </font>$`Q(s_{t}, a_{t})`$：
$$
y_t = \sum_{i=0}^{n-1} \gamma^i R_{t+i} + \gamma^n Q(s_{t+n}, a_{t+n})
$$
或$`y_t = \sum_{i=0}^{n-1} \gamma^i R_{t+i} + \gamma^n V(s_{t+n})`$

另外有<font style="color:#DF2A3F;"> λ 加权 n 步 TD 目标，称为TD(λ)：</font>
$$
y_t^{\lambda}
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}
\left(
\sum_{i=0}^{n-1} \gamma^i R_{t+i}
+
\gamma^n V(s_{t+n})
\right)
$$
### 2.1.4 演员-评论家 Actor-Critic
下面看看 TD 方法结合神经网络是怎么估计$`Q^\pi`$，并实现策略网络$`\pi(a|s;\theta)`$更新的：$`{\theta} \leftarrow {\theta}+\beta \cdot \nabla_\theta J(\theta)`$。

**<font style="color:#DF2A3F;">用另一个神经网络</font>**$`q(s_t,a_t;\omega)`$**<font style="color:#DF2A3F;">估计</font>**$`Q^\pi(s_t,a_t)`$**<font style="color:#DF2A3F;">,通常称为价值网络，怎么让它近似任意</font>**$`(s_t,a_t)`$**<font style="color:#DF2A3F;">对应的真正的</font>**$`Q^\pi_{true}(s_t,a_t)`$**<font style="color:#DF2A3F;">呢？</font>**定义损失函数：$`L_w \triangleq \frac{1}{2} \left[ q(s_t, a_t; w) - Q^\pi_{true}(s_t,a_t) \right]^2`$然而$`Q^\pi_{true} (s_t,a_t)`$是提前未知的，不像监督学习有一个标记正确值，所以只能通过TD 目标$`\hat{y}_t=R_t+\gamma \cdot q(s_{t+1},a_{t+1};\omega)`$估计，
$$
L_w \triangleq \frac{1}{2} \left[ q(s_t, a_t; w) - \hat{y}_t \right]^2
$$
那既然 TD 能估计$`Q^\pi(s_t,a_t)`$了，还要神经网络去估计$`Q^\pi(s_t,a_t)`$干嘛呢？因为 TD 估计$`Q(s_t,a_t)`$却需要$`Q(s_{t+1},a_{t+1})`$的值，这个值哪来？答，神经网络可以提供！**<font style="color:#DF2A3F;">起飞，神经网络和 TD 左脚踩右脚螺旋升天，这就是自举！</font>**

下面来看看这种使用两个神经网络的方案，称为演员-评论家**（Actor–Critic，AC）**，具体是如何操作的。

**Actor–Critic本质上是“一类算法框架”，而不是一个单一的具体算法。**基于 Q 函数的随机 Actor-Critic 方法（Stochastic Actor-Critic with Q-function critic）是这个框架中的一种**初级实现形式**，下面介绍这种实现。<!-- 这是一张图片，ocr 内容为：动作A 价值Q 价值网络 策略网络 奖励R 环境 (演员) (评委) 状态S -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773040752939-afaef874-ece0-4fae-bb27-40fbb0346e45.png)

:::info
1. 观测到当前状态$`s_t`$，根据策略网络做决策：$`a_t \sim \pi(\cdot \mid s_t; \boldsymbol{\theta}_{\text{now}})`$，并让智能体执行动作$`a_t`$。
2. 从环境中观测到奖励$`R_t`$和新的状态$`s_{t+1}`$。
3. 根据策略网络做决策：$`\tilde{a}_{t+1} \sim \pi(\cdot \mid s_{t+1}; \boldsymbol{\theta}_{\text{old}})`$，注意环境此时还未执行动作$`\tilde{a}_{t+1}`$。
4. 让价值网络打分：  
$$
\hat{q}_t = q(s_t, a_t; \boldsymbol{w}_{\text{old}})
$$
$$
\hat{q}_{t+1} = q(s_{t+1}, \tilde{a}_{t+1}; \boldsymbol{w}_{\text{old}})
$$
5. 计算 TD 目标和 TD 误差：  
$$
\hat{y}_t = R_t + \gamma \cdot \hat{q}_{t+1}
$$
$$
\delta_t = \hat{q}_t - \hat{y}_t
$$
6. <font style="color:#DF2A3F;">梯度下降，更新价值网络</font>$`q(s_t,a_t;\omega)`$<font style="color:#DF2A3F;">：</font>  
$$
\boldsymbol{w}_{\text{new}} \leftarrow \boldsymbol{w}_{\text{old}} - \alpha \cdot\nabla_{\boldsymbol{w}}L_{\omega} ，\;\;其中右边=\boldsymbol{w}_{\text{old}} - \alpha \cdot \delta_t \cdot \nabla_{\boldsymbol{w}} q(s_t, a_t; \boldsymbol{w}_{\text{old}})
$$
$`注意这里L_w = 1/2 \left[ q(s_t, a_t; w) - (\;R_t+\gamma \cdot q(s_{t+1},a_{t+1};\omega) \;) \right]^2`$对 $`\boldsymbol{w}`$求导时，不将 TD target =$`(\;R_t+\gamma \cdot q(s_{t+1},a_{t+1};\omega) \;)`$视作 $`\boldsymbol{w}`$的函数。这是“半梯度（semi-gradient）方法”**的做法：  
**有意忽略 TD target 对  $`\boldsymbol{w}`$的依赖，否则训练会变得不稳定甚至发散。

7. <font style="color:#DF2A3F;">梯度上升，更新策略网络</font>$`\pi(a|s;\theta)`$<font style="color:#DF2A3F;">：</font>  
$$
\boldsymbol{\theta}_{\text{new}} \leftarrow \boldsymbol{\theta}_{\text{old}} + \beta \cdot \nabla_\theta J(\theta)，\;\;其中右边=\boldsymbol{\theta}_{\text{old}} + \beta \cdot \hat{q}_t \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a_t \mid s_t; \boldsymbol{\theta}_{\text{old}})
$$
:::



Actor–Critic 架构在 1983年由 Sutton 和 Barto 等提出，那时候连策略梯度定理都没有，只有简单的神经网络。虽没严格推导策略梯度，但他们意识到Actor 可以根据 Critic 学到的价值函数来调整策略。后来Sutton发现，当时他们用的 td 误差和资格际的方法其实是策略梯度的一种低方差实现，对此他在1999 年的论文进行了总结。在早期价值网络使用蒙特卡洛法估计 Q 值，1988 年 Sutton 系统性提出 TD 方法后，Actor-Critic+TD 才成为强化学习标准框架。



### 2.1.5  A2C
（Advantage Actor-Critic）优势 演员-评论家 简称 A2C。

**带基线的策略梯度定理**$`^{证明见(注5)}`$**：**

:::info
<!-- 这是一张图片，ocr 内容为：定理8.1.带基线的策略梯度定理 设B是任意的函数,但是B不能依赖于A.把B作为动作价值函数QN(S,A)的基 线,对策略梯度没有影响: 8S(BARATISI) ( 0N(S, 4)- 6 ) . VO MR(ALS: (ALS: ( ' ' ' ES -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773041022213-0991ecff-ad7f-47e1-a87e-2e10836276a3.png)

<!-- 这是一张图片，ocr 内容为：之前我们推导出了带基线的策略梯度,并且对策略梯度做了蒙特卡洛近似,得到策 略梯度的一个无偏估计: [2元(S, A) - VR(5)] . VE IN R(A S; 0) . G(S,A; O) (8.2) 优势函数 因此,基于上面公式得到的 公式中的QM-V.被称作优势函数(ADVANTAGE FUNCTION). ACTOR-CRITIC 方法被称为 ACTOR-CRITIC 方法被称为  A2C. -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773041045296-c22b30a4-69bf-4f71-baa7-e19a9d0f0fca.png)

:::

<!-- 这是一张图片，ocr 内容为：( ) | ) ( ) | ' ' ' ' ) | ) | ' ' ' ( ( ) - ' ) | ) | ) | ) | ) | ) | ) | ) | ) | ) | ( ( ( ( ) | ) ) 9(ST , AT;  O) VE INN(AT ) ST; E) . ST+1; W) TD 目标 -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773043031177-0be19d10-716c-4ece-989f-3f337ace402b.png)

A2C 的训练流程和上一小节的初级 AC 方法（基于 Q 函数的随机 Actor-Critic 方法）差不多，但它的价值网络不需要输入动作，因为价值网络根据优势函数的正负就能判断动作的好坏。

<!-- 这是一张图片，ocr 内容为：动作A 策略网络 TD误差8 价值网络 奖励R 环境 (演员) (评委) 状态S -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773043265348-c0be8e96-9407-44cf-bd6f-e0de2dc90e61.png)

其中价值网络的输出是$`v(s_t;\omega)`$

A2C 由 Volodymyr等人在 2016 年提出。

## 2.2、置信域优化
本节介绍John Schulman（约翰 舒尔曼） 于 2015 年提出的 TRPO（置信域策略优化）以及 2017 年的续作 PPO（近端策略优化）。置信域方法不是TRPO的论文提出的，而是数值最优化领域中一类经典的算法，历史可以追溯到1970年。置信域方法需要构造一个函数$`L(\theta \mid \theta_{\text{old}})`$，这个函数要满足这个条件：
$$
L(\theta \mid \theta_{\text{old}}) \approx J(\theta), \quad \forall \theta \in \mathcal{N}(\theta_{\text{old}})
$$
顾名思义，在$`\theta_{\text{old}}`$的邻域上，可以信任$`L(\theta \mid \theta_{\text{old}})`$，那么集合$`\mathcal{N}(\theta_{\text{old}})`$就被称作置信域。  
可以拿$`L(\theta \mid \theta_{\text{old}})`$来替代目标函数$`J(\theta)`$

### 2.2.1 替代目标函数：重要性采样
目标函数$`J({\theta})=\mathbb{E}_S\left[V^\pi(S)\right]`$可以等价写成$`^{证明见(注5-2)}`$：
$$
J(\theta) = \mathbb{E}_{S \sim V^{\pi_{\text{old}}}} 
\Bigg[
\mathbb{E}_{A \sim \pi_{\text{old}}(\cdot \mid S)} 
\Big[
\frac{\pi_{\text{new}}(A \mid S)}{\pi_{\text{old}}(A \mid S)} \, Q^{\pi_{\text{new}}}(S, A)
\Big]
\Bigg]
$$
<font style="color:#601BDE;">实际会用</font>$`Q^{\pi_{\text{old}}}(S, A)`$<font style="color:#601BDE;">近似</font>$`Q^{\pi_{\text{new}}}(S, A)`$<font style="color:#601BDE;">。</font>

**重要性采样：用旧策略采集的数据，去估计新策略的期望回报。**

之前介绍过的 REINFORCE 算法收集完一整段轨迹才更新；而 Actor–Critic ，其初级版本 Stochastic Actor-Critic with Q-function critic是单步更新而不是收集一段轨迹再更新。但是单步更新的效率是非常低的，大部分时间都花在了使用新策略去采数据上。如果能用旧策略采样一段轨迹去更新当前策略就好了，可是在 TRPO 提出之前，这种方案不能保证策略价值单调提升。

对目标函数$`J({\theta})`$作差：
$$
\begin{aligned}
J\left(\theta\right)-J(\theta_{old}) & =\mathbb{E}_{S}\left[V^{\pi_{\theta}}\left(S\right)\right]-\mathbb{E}_{S}\left[V^{\pi_{\theta_{old}}}\left(S\right)\right] \\
& =\frac{1}{1-\gamma} \mathbb{E}_{s \sim \nu ^{\pi_{\theta}}} \mathbb{E}_{a \sim \pi_{\theta}(\cdot \mid s)}\left[A^{\pi_{\theta_{old}}}(s, a)\right]
\end{aligned}
$$
 ^{证明见(注5-3)}

只要能找到一个新策略，使得该式>0，就能保证策略性能单调递增。



$`L_{\theta_{old}}\left(\theta\right)`$和$`L(\theta)`$

+ 用$`L_{\theta_{old}}\left(\theta\right)`$表示$`J({\theta})`$的近似。它是通过 旧策略的状态分布$`V^{\pi_{\theta_{old}}}(s)`$来采样状态的，但它还是使用了 新策略的动作分布$`\pi_{\theta}(a \mid s)`$。
+ 公式为：
$$
L_{\theta_{old}}\left(\theta\right)=J(\theta_{old})+\frac{1}{1-\gamma} \mathbb{E}_{s \sim \nu ^{\pi_{\theta_{old}}}} \mathbb{E}_{a \sim \pi_{\theta}(\cdot \mid s)}\left[A^{\pi_{\theta_{old}}}(s, a)\right]
$$
考虑等式右边的关于变量$`\theta`$的 部分，且用 旧策略的采样（即旧策略的状态和动作分布）来近似新策略下的回报：
$$
L(\theta)=\mathbb{E}_{s \sim \nu^{\pi_{\theta_{\text {old }}}}, a \sim \pi_{\theta_{\text {old }}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)} A^{\pi_{\theta_{\text {old }}}}(s, a)\right]
$$
在一般的 TRPO 表述中，称$`L(\theta)`$为 替代目标函数（surrogate objective）。

  
**<font style="color:#DF2A3F;"> TRPO 目标即最大化 替代目标函数，同时约束新策略和旧策略之间的 KL 散度：</font>**
$$
\max_{\theta} \;\;\;L(\theta)=\mathbb{E}_{s \sim \nu^{\pi_{\theta_{\text {old }}}}, a \sim \pi_{\theta_{\text {old }}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)} A^{\pi_{\theta_{\text {old }}}}(s, a)\right]
$$
**KL 散度**$`^{(注5-4)}`$**的约束为：**
$$
D_{K L}\left(\pi_{\theta_{\text {old }}} \| \pi_\theta\right) \leq \delta
$$
### 2.2.2 KL 约束优化问题的求解
#### （1）自然梯度优化方法
在$`\theta \approx \theta_{\text{old}}`$时，**替代目标函数 可以一阶展开**：
$$
L(\theta) \approx L(\theta_{\text{old}}) + g^T (\theta - \theta_{\text{old}})
$$
其中：$`g=\nabla_{\theta}L(\theta)=\nabla_{\theta} \mathbb{E}_{s \sim V^{\pi_{\theta_{old}}}} \mathbb{E}_{a \sim \pi_{\theta_{old}}(\cdot \mid s)}\left[\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} A^{\pi_{\theta_{old}}}(s, a)\right]`$

为了保证新策略不偏离旧策略 太远，TRPO通过对 **KL散度 在 **$`\theta`$** 处做 二阶泰勒展开**（即求 Hessian 矩阵$`^{(注6-0)}`$） 来进行近似：
$$
D_{KL}(\pi_{\theta_{old}} \parallel \pi_\theta) \approx \frac{1}{2} (\theta - \theta_{old})^T F (\theta - \theta_{old})
$$
其中$`F`$是 Fisher信息矩阵。在以模型参数$`\theta`$为变量，计算 KL 散度关于$`\theta`$的 Hessian 矩阵时，在该点（两个分布重合时）的值恰好等于 Fisher 信息矩阵，它是策略概率分布的 对数似然函数的二阶导数，也可以理解为 KL 散度的二阶近似。$`F=\mathbb{E}_{\pi_{\theta_{\text {old }}}}\left[\nabla_\theta \log \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)^T\right]`$,更多 Fisher矩阵内容见附注$`^{(注6)}`$。

这样问题就变成了**一个带二次约束的一阶线性优化问题**：
$$
\begin{aligned}
\max_{\Delta \theta} \quad & g^T \Delta \theta \\
\text{s.t.} \quad & \frac{1}{2} \Delta \theta^T F \Delta \theta \le \delta
\end{aligned}
$$
使用卡罗需-库恩-塔克（Karush-Kuhn-Tucker，KKT）条件可以直接得到这个问题的答案（也可以用拉格朗日乘子法求解）：
$$
\Delta \theta = \frac{\sqrt{2 \delta}}{\sqrt{g^T F^{-1} g}} \, F^{-1} g
$$
参数更新方式：$`\theta_{new}=\theta+\beta F^{-1}g`$，$`\beta={\sqrt{2 \delta}}/{\sqrt{g^T F^{-1} g}}`$是缩放步长系数， 这就**自然梯度（Natural Gradient）优化方法**：自然梯度通过修改梯度$`g`$的方向为$`F^{-1}g`$，使得在分布空间中每次更新都尽量沿着Fisher信息矩阵指引的最有效的方向前进。这其实就是是 **在 KL 约束形成的椭球内找到最优上升点**：

$`\Delta \theta^T F \Delta \theta \le 2\delta`$，最优解就是：**梯度方向与椭球边界的交点。**数学上正好就是{方向：$`F^{-1}g`$,长度：$`\beta`$}。

#### （2）g 的计算
如前所述，$`g=\nabla_{\theta}L(\theta)=\nabla_{\theta} \mathbb{E}_{s \sim \nu^{\pi_{\theta_{old}}}} \mathbb{E}_{a \sim \pi_{\theta_{old}}(\cdot \mid s)}\left[\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} A^{\pi_{\theta_{old}}}(s, a)\right]`$,计算机不可能列出所有$`s \sim \nu^{\pi_{\theta_{old}} }, a \sim \pi_{\theta_{old}}`$求加权平均，所以只能用蒙特卡洛法选几个样本计算：

如果选一个样本，则：$`g=\nabla_{\theta} \left[\frac{\pi_{\theta}(a_1 \mid s_1)}{\pi_{\theta_{old}}(a_1 \mid s_1)} A^{\pi_{\theta_{old}}}(s_1, a_1)\right] = \frac{A^{\pi_{\theta_{old}}}(s_1, a_1)}{\pi_{\theta_{old}}(a_1 \mid s_1)}\nabla_{\theta}\pi_{\theta}(a_1 \mid s_1)`$【其中$`\nabla_{\theta}\pi_{\theta}`$通过策略网络反向传播得到】

如果选多个样本，则：$g \approx \frac{1}{N} \sum_{i=1}^{N}
\nabla_{\theta}
\left(
\frac{\pi_{\theta}(a_i \mid s_i)}
{\pi_{\theta_{\text{old}}}(a_i \mid s_i)}
\right)
A_i
$`。这里的 优势函数`$A$的计算见 2.2.4。

#### （3）F 逆的计算
计算$`F`$的 逆矩阵会耗费大量的内存资源和时间。TRPO 通过**共轭梯度法**$`^{(注7)}`$（conjugate gradient method）回避了这个问题，它的核心思想是直接计算$`F^{-1} g`$。具体来说，设$`x=F^{-1} g`$， 将$`\Delta \theta = \beta x`$代入约束条件$`\frac{1}{2} \Delta \theta^T F \Delta \theta = \delta`$得到$`\beta = \frac{\sqrt{2 \delta}}{\sqrt{x^T F x}}`$,于是$`\Delta \theta =\frac{\sqrt{2 \delta}}{\sqrt{x^T F x}} x`$。因此解方程$`F x =g`$解出$`x`$即可避免求逆,$`F`$是Score 函数的协方差矩阵 ，必然正定，可用**共轭梯度法**求解 方程。

#### （4）回溯线性搜索
为了确保KL约束和目标函数增加，TRPO 还会进行回溯线性搜索backtracking line search：$`\Delta \theta =\alpha^i\frac{\sqrt{2 \delta}}{\sqrt{x^T F x}} x`$,$`i`$=1,2,3，...直到满足替代目标函数增加以及满足 KL 约束。

### 2.2.3 TRPO 总结
综上， TRPO 的关键步骤包括：

1. **构造优化目标**：利用作差证明单调性与重要性采样，将策略优化问题转化为替代目标函数的最大化，同时引入 KL 散度约束以保证策略更新的稳定性。
2. **二阶近似约束**：对替代目标函数在置信域内做一阶展开，对对 KL 散度约束进行二阶泰勒展开，将原始约束优化问题简化为带二次约束的一阶线性优化问题。
3. **自然梯度优化**：通过自然梯度方法，沿 Fisher 信息矩阵引导的方向更新参数，实现最有效的策略改进。
4. **共轭梯度求解**：利用共轭梯度法计算参数更新，避免直接求解 Fisher 信息矩阵的逆，从而节省计算资源。
5. **回溯线性搜索**：通过回溯线性搜索动态缩减步长，确保策略更新满足 KL 散度约束，保证训练安全稳定。

### 2.2.4 广义优势估计 GAE
#### (1)问题的提出
上面在计算$`g`$时还需要计算优势函数$`A`$。

回想优势函数的定义：
$$
A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)−V^{\pi}(s_t)
$$
可以类似 A2C 的做法，用价值网络$`v(s_t;\omega)`$去近似$`V^{\pi}(s_t)`$。$`Q^{\pi}(s_t,a_t)`$要如何近似呢？

REINFORCE 等算法的做法是直接蒙特卡洛，等待一个 episode 探索结束用样本中的<font style="color:#DF2A3F;">  T 个真实回报 的平均</font>$`\frac{1}{T}\sum_{t=1}^{T}   \tilde{G}_t`$来近似$`Q^\pi`$，其中$`\tilde{G}_t= \sum_{k=t}^{T} \gamma^{k-t} R_{k} = R_{t} + \gamma R_{t+1} + \gamma^2R_{t+2}+...+\gamma^{T}R_{T-t}，\forall \, t = 1, \cdots, T.`$<font style="color:#DF2A3F;"> </font>。优点:无偏估计。缺点:1. 方差较大,  2. 必须等 episode 结束。

A2C 等算法的做法是自举， 算单步 TD 目标，$`Q^{\pi}(s_t,a_t) \leftarrow R_t+\gamma \cdot v\left(s_{t+1} ; {w}\right)`$。优点：1.方差较小，2.用当前步骤的$`(s_t,s_{t+1},R_t)`$就能估计， 缺点：偏差较大。

<!-- 这是一张图片，ocr 内容为：QN(ST, AT) (自举) UT (蒙特卡洛 -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773373368431-6022a55b-6662-4237-9e96-2426899de9a3.png)

能否将两种方法结合呢？ Sutton 在 1988 年提出 TD(**λ**) 解决过这个问题$`^{(注 8) }`$。 舒尔曼在发表 TRPO 后又专门发了一篇论文讨论，在 TD(**λ**) 基础上提出了 GAE，后来将其用到了 TRPO、PPO 等算法的实现中。 

#### (2)GAE 推导与代码实现
先来看**<font style="color:#DF2A3F;">单步 TD是如何估计优势函数</font>**的，由$`A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)−V^{\pi}(s_t)`$，$`V^{\pi}(s_t) \leftarrow V(s_t)`$，$`Q(s_t,a_t) \leftarrow R_t + \gamma V(s_{t+1})`$,故可用$`\hat{A}(s_t,a_t)`$估计$`A^{\pi}(s_t,a_t)`$：
$$
\hat{A}(s_t,a_t) = R_t + \gamma V(s_{t+1}) - V(s_t)
$$
右边正好也是**<font style="color:#DF2A3F;">状态价值函数形式的 TD 差分</font>**：
$$
\delta_t^V = R_t + \gamma V(s_{t+1}) - V(s_t)
$$
回想 2.1.3 节定义的多步 TD 目标估计$`Q^\pi\left(s_t, a_t\right)`$，
$$
y_t = \sum_{i=0}^{n-1} \gamma^i R_{t+i} + \gamma^n V(s_{t+n})
$$
可以得到 k 步优势函数估计：
$$
\hat{A}^{(k)}(s_t,a_t) = \sum_{i=0}^{k-1} \gamma^i R_{t+i} + \gamma^k V(s_{t+k}) - V(s_t)
$$
于是$`\hat{A}^{(1)}(s_t,a_t) = R_t +\gamma V(s_{t+1}) - V(s_t)=\delta_t^V`$
$$
\hat{A}^{(2)}(s_t,a_t) = R_t +\gamma R_{t+1}+ \gamma^2 V(s_{t+2}) - V(s_t) = \delta_t^V+\gamma \delta_{t+1}^V
$$
$$
\hat{A}^{(3)}(s_t,a_t) = R_t +\gamma R_{t+1}+\gamma^2 R_{t+2}+ \gamma^3 V(s_{t+2}) - V(s_t) = \delta_t^V+\gamma \delta_{t+1}^V+\gamma^2 \delta_{t+2}^V
$$
……

得到一族优势函数估计：
$$
\hat{A}^{(1)}(s_t,a_t)，\hat{A}^{(2)}(s_t,a_t)，\hat{A}^{(3)}(s_t,a_t)，...，\hat{A}^{(\infty)}(s_t,a_t)
$$
| estimator | bias | variance |
| --- | --- | --- |
|$`\hat{A}^{(1)}(s_t,a_t)`$| 高 | 低 |
|$`\hat{A}^{(k)}(s_t,a_t)`$| 中 | 中 |
|$`\hat{A}^{(\infty)}(s_t,a_t)`$相当于蒙特卡洛 | 低 | 高 |


**<font style="color:#DF2A3F;">广义优势估计即把所有</font>**$`\hat{A}^{(k)}(s_t,a_t)`$**<font style="color:#DF2A3F;">做指数加权平均 来估计优势函数：</font>**
$$
\begin{aligned}
\hat{A}_t^{\text{GAE}} 
&=
(1-\lambda)
\sum_{k=1}^{\infty}
\lambda^{k-1}
\hat{A}_t^{(k)}\\
&=\sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
\\
&=
\delta_t^V
+
\gamma\lambda
\hat{A}_{t+1}^{\mathrm{GAE}}
\end{aligned}
$$
第二个等号证明
$$
\begin{aligned}
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
&:= (1-\lambda)\left(
\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \cdots
\right) \\

&= (1-\lambda)\Big(
\delta_t^V
+ \lambda(\delta_t^V + \gamma \delta_{t+1}^V)
+ \lambda^2(\delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V)
+ \cdots
\Big) \\

&= (1-\lambda)\Big(
\delta_t^V (1 + \lambda + \lambda^2 + \cdots)
+ \gamma \delta_{t+1}^V (\lambda + \lambda^2 + \lambda^3 + \cdots) \\
&\qquad\qquad
+ \gamma^2 \delta_{t+2}^V (\lambda^2 + \lambda^3 + \lambda^4 + \cdots)
+ \cdots
\Big) \\

&= (1-\lambda)\left(
\delta_t^V \left(\frac{1}{1-\lambda}\right)
+ \gamma \delta_{t+1}^V \left(\frac{\lambda}{1-\lambda}\right)
+ \gamma^2 \delta_{t+2}^V \left(\frac{\lambda^2}{1-\lambda}\right)
+ \cdots
\right) \\
&=\delta_t^V+\gamma\lambda\delta_{t+1}^V+(\gamma\lambda)^2\delta_{t+2}^V + ...\\
&= \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
\end{aligned}
$$
第三个等号证明，把$`\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}=\delta_t^V+\gamma\lambda\delta_{t+1}^V+(\gamma\lambda)^2\delta_{t+2}^V + ...`$中的$`t`$换成$`t+1`$代入$`\hat{A}_{t+1}^{\mathrm{GAE}}`$即可。

**<font style="color:#DF2A3F;">GAE 在实际实现的时候一般用第三个等号，也就是递推式子实现，从最后一个时间步开始反向遍历：</font>**

```python
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy() # shape:[T,1]
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float) # shape:[T,1]
$$
可见，GAE 不能在当前步立即计算，必须等一段数据结束：

+ **TD(0)**：当前步即可用价值网络计算$`\hat{A}(s_t,a_t) = R_t + \gamma V(s_{t+1}) - V(s_t)`$
+ **GAE**：需要一段 rollout 才能反向遍历计算， **但不需要 episode 结束，只需要 rollout 末端的 value 。GAE 反向以 advantage=0 起算，末端 value由构造 td_delta 时决定。**

#### (3)GAE 方差解析
**1 Monte Carlo 的方差为什么大**

Monte Carlo advantage：$`A_t^{MC} = \sum_{l=0}^{T-t-1} \gamma^l R_{t+l} - V(s_t)`$

这里的问题是：未来所有 reward 都是随机变量$`R_t,\; R_{t+1},\; R_{t+2},\; \dots`$

而 return 是$`G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots`$

方差：$`\mathrm{Var}(G_t) \approx \mathrm{Var}(R_t) + \gamma^2 \mathrm{Var}(R_{t+1}) + \dots`$

所以：horizon 越长，variance 越大。如果任务很长（机器人控制就是这样），variance 会非常大。

**2 TD 方法为什么方差小**

TD advantage：$`A_t^{TD} = \delta_t`$其中，$`\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)`$

关键点：未来的随机 return：$`R_{t+1} + R_{t+2} + \dots`$被 value function 的期望值$`V(s_{t+1})`$替代了。而期望值的 variance 显然更小。所以：TD = bootstrap → variance 低。

**3 GAE 的权衡**

GAE 其实是在做：多步 TD 的加权平均$`A_t^{GAE} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots`$

注意：每个$`\delta`$都是$`r + \gamma V - V`$也就是说：未来 reward 不断被 value function 截断（bootstrap）。

例如：$`\delta_{t+1} = R_{t+1} + \gamma V_{t+2} - V_{t+1}`$所以**未来 reward 的随机性被不断替换成 value 预测。**

** 最大的方差减少来自 baseline **$`V(s_t)`$**，而 λ 只是进一步微调 偏差-方差 权衡。  **

**4 奖励塑形和响应函数**

GAE 原论文还从奖励塑形和响应函数的角度分析了这一问题。<font style="color:rgb(6, 10, 38);">通过使用价值函数 </font>_<font style="color:rgb(6, 10, 38);">V</font>_<font style="color:rgb(6, 10, 38);"> 来构造新的奖励（贝尔曼残差 δ ），这相当于把未来的奖励“提前”了（把延迟响应变成了即时响应）。即便经过塑形，长远的未来依然有噪声。 </font>_<font style="color:rgb(6, 10, 38);">λ</font>_<font style="color:rgb(6, 10, 38);"> 就像一个“剪刀”，把太远的、不可靠的未来回报切掉，从而降低方差，虽然这会引入一点偏差。</font>



### 2.2.5 TRPO 的改进：PPO
#### (1)优化目标的变换
TRPO 涉及二阶优化、conjugate gradient、line search、实现复杂，故约翰舒尔曼两年后提出了改进版本，即 PPO。PPO 初始要优化的目标和 TRPO 一样，都是替代目标函数(见 2.2.1 节)：
$$
\max_{\theta} \;\;\;L(\theta)=\mathbb{E}_{s \sim V^{\pi_{\theta_{\text {old }}}}, a \sim \pi_{\theta_{\text {old }}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)} A^{\pi_{\theta_{\text {old }}}}(s, a)\right]
$$
$$
\text{s.t.} \quad D_{K L}\left(\pi_{\theta_{\text {old }}} \| \pi_\theta\right) \leq \delta
$$
PPO 有两种独立的方法解决这个优化问题， PPO 惩罚和 PPO 截断，常用的是PPO 截断。

PPO 惩罚（PPO-Penalty）用拉格朗日乘数法直接将 KL 散度的限制放进了目标函数中，这就变成了一个无约束的优化问题，在迭代的过程中不断更新 KL 散度前的系数：
$$
\max_{\theta} \;\;\;L(\theta)=\mathbb{E}_{s \sim V^{\pi_{\theta_{\text {old }}}}, a \sim \pi_{\theta_{\text {old }}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)} A^{\pi_{\theta_{\text {old }}}}(s, a)-D_{K L}\left(\pi_{\theta_{\text {old }}} \| \pi_\theta\right)\right]
$$
 PPO-截断（PPO-Clip）更加直接，它在目标函数中进行限制，以保证新的参数和旧的参数的差距不会太大，即：
$$
\max_{\theta} \;\;\;L(\theta)=\mathbb{E}_{s \sim V^{\pi_{\theta_{\text {old }}}}, a \sim \pi_{\theta_{\text {old }}}}\left[min（\rho\cdot A,\quad clip(\rho,1-\epsilon,1+\epsilon)\cdot A）\right]
$$
其中$`\rho`$为 ratio，$`\rho=\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)}`$；$`A=A^{\pi_{\theta_{\text {old }}}}(s, a)`$；$`0<\epsilon<1{通常取0.2}`$

#### (2)min 式中 clip 的理解
**clip机制的核心目的就是：****<font style="color:#DF2A3F;">限制策略更新幅度</font>**

先来看看$`min（\rho\cdot A,\quad clip(\rho,1-\epsilon,1+\epsilon)\cdot A）`$的值

|$`min（\rho\cdot A,\quad clip(\rho,1-\epsilon,1+\epsilon)\cdot A）`$的值 |$`优势A\gt0`$|$`优势A\lt0`$|
| --- | :---: | :---: |
|$`\rho \gt 1+\epsilon`$|$`(1+\epsilon)A`$|$`\rho A`$|
|$`\rho \lt 1-\epsilon`$|$`\rho A`$|$`(1-\epsilon)A`$|
|$`1-\epsilon \le \rho \le1+\epsilon`$|$`\rho A`$|$`\rho A`$|


openAI 官网把 min 式写为等价形式去掉 clip：

$`min（\rho\cdot A,\quad clip(\rho,1-\epsilon,1+\epsilon)\cdot A)=min（\rho\cdot A,\quad g(\epsilon,A)）`$其中，$g(\epsilon, A) =
\begin{cases}
(1 + \epsilon) A & A \ge 0 \\
(1 - \epsilon) A & A < 0
\end{cases}$。

<font style="color:#DF2A3F;">上式可以写成</font>$`min（\rho\cdot A,\quad clip(\rho,1-\epsilon,1+\epsilon)\cdot A)=min（\rho\cdot A,\quad (1 \pm \epsilon) A)`$<font style="color:#DF2A3F;">,其中</font>$`\pm`$<font style="color:#DF2A3F;">当 A 正时取正，A 负取负</font>

min 式的另一种等价形式则可以把 A 提出 min，且去掉 clip， 需要注意 A 为负的时候 min 变为 max：

$min（\rho\cdot A,\quad clip(\rho,1-\epsilon,1+\epsilon)\cdot A） =
\begin{cases}
|A|\cdot min(\rho,1 + \epsilon) & A \ge 0 \\
-|A|\cdot max(\rho,1 - \epsilon)& A < 0
\end{cases}$

可以做出$`\rho`$-min 式值 的图，注意**纵轴单位是 |A**|：

<!-- 这是一张图片，ocr 内容为：HIN式的值 3+113-1 意桥大 -->
![](https://cdn.nlark.com/yuque/0/2026/jpeg/57851636/1773885437360-4a74d003-d930-457a-bac0-b941dbd6960f.jpeg)

**理解：A>0 表示当前动作比平均水平****<font style="color:#DF2A3F;">好</font>****，策略在优化器作用下会倾向于****<font style="color:#DF2A3F;">多选</font>****这个动作，但如果 ratio 太大（**$`>1+\epsilon=1.2`$**），表明策略朝着这个****<font style="color:#DF2A3F;">变好（多选好动作）</font>****的方向更新太大，有过度利用旧策略采样的数据来更新的风险，所以用 clip 来阻止再优化这个好方向；A<0 表示当前动作比平均水平****<font style="color:#DF2A3F;">差</font>****，策略在优化器作用下会倾向于****<font style="color:#DF2A3F;">少选</font>****这个动作，但如果ratio 太小（**$`<1-\epsilon=0.8`$**），表明策略朝着这个****<font style="color:#DF2A3F;">变好（少选差动作）</font>****的方向更新太大，有过度利用旧策略采样的数据来更新的风险，所以用 clip 来阻止再优化这个好方向。所以 PPO clip的不是对称地限制策略变化大小，而是只限制“变好”幅度。它的逻辑是：在“变好”，就防止走太快；在“变差”，那不阻拦（优化器自己会避免）。**

**综上，PPO clip 是把目标函数中那些过于乐观的部分直接削平，阻止朝着这些方向更新。也可以从梯度优化的视角来理解。注意 **`**min()**`**式子在 loss 中，当策略更新过度被 clip 时，loss 的梯度对 θ 梯度变为 0。所以 clip 不是简单的“缓和”，而是 直接让梯度失效，强制策略更新停在安全区间内。 误解：很多人以为 PPO clip 只是“缓和激进采样”，其实本质是“让梯度在超出阈值时直接为 0”，这比单纯缩放梯度更严格，更稳定。**



#### (3)PPO 算法流程
##### 无并行环境$`^{(注 12)}`$
:::info
**1.** 输入：初始策略网络参数$`\theta_0`$，初始价值网络参数$`\omega_0`$。

**2.** `for`$`k = 0, 1, 2, \ldots`$（第$`k`$轮数据rollout[交互采样] = 第$`k`$局 episode）`do`

**3.** 通过在环境中运行策略$`\pi(\theta_k)`$，收集一条轨迹$`\tau=(s_0, a_0, R_1, s_1, a_1, R_2, \ldots, s_{T-1}, a_{T-1}, R_T)`$。把每一时间步$`(s_t, a_t, R_{t+1})`$对应的$`[s_t, a_t, R_{t+1}, s_{t+1}, \log\pi_{\theta_k}(a_t\mid s_t), v(s_t;\omega_k), \text{done}]`$写进 rollout buffer（轨迹缓冲区）。

**4.** 基于当前价值网络$`v(s_t;\omega_k)`$，计算优势估计$`\hat{A}_t`$（使用 GAE 估计方法）：
$$
\hat{A}_t^{\text{GAE}}=\delta_t^V+\gamma\lambda\hat{A}_{t+1}^{\text{GAE}}=
 R_{t+1}+\gamma v(s_{t+1};\omega_k)-v(s_t;\omega_k)+\gamma \lambda \hat{A}_{t+1}^{\text{GAE}}.
$$
其中令$`v(s_{t+1};\omega_k) = 0`$（若$`\text{done} = \text{True}`$）。并令$`\hat{A}_{T}^{\text{GAE}}=0`$，反向遍历，得到优势估计列表$`\text{advantage-list}:[\hat{A}_0^{\text{GAE}},\hat{A}_1^{\text{GAE}},\ldots,\hat{A}_{T-1}^{\text{GAE}}]`$。

**5.** 计算回报$`\hat{G}_t`$。注意这里的$`\hat{G}_t`$是用于价值网络回归的目标；不是理论定义的随机回报$`G_t = R_t+\gamma G_{t+1}`$。回想A2C使用单步TD target构造价值目标：$`\hat{G}_t=R_{t+1}+\gamma v(s_{t+1};\omega_k)`$，偏差太大；现在有了GAE方法估算优势A，又优势定义为A=Q-V，沿着采样轨迹，把 Q对应成“从 (t) 开始的回报目标”$`\hat G_t`$，就可以通过第 4 步得到的$`\hat{A}_t^{\text{GAE}}`$计算$`\hat{G}_t=\hat{A}_t^{\text{GAE}}+v(s_t;\omega_k)`$，得到$`\text{value-list}:[\hat{G}_0,\hat{G}_1,\ldots,\hat{G}_{T-1}]`$。

**6.** `for` `epoch` `in` `epochs` `do`

+ 注意：下面 7、8 两步通常对同一组 rollout buffer 里的数据重复 `epochs` 次梯度更新。注意区分 episode 和 epoch，在本 无并行环境 的流程中，episode 是外循环，epoch 是内循环。
+ 下面的$`\arg\max`$/$`\arg\min`$只是优化目标，一次梯度步无法完全达到该极值。

**7.** 通过最大化 PPO-Clip 目标更新策略：
$$
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
$$
式中$`(1\pm\epsilon)`$的正负号与$`\hat{A}_t^{\text{GAE}}`$同号。通常把$`\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_k}(a_t\mid s_t)}`$恒等变换为$`\exp\bigl(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_k}(a_t\mid s_t)\bigr)`$。log 使数值稳定，避免太接近 0 的概率被当做 0 处理$`^{(注 10) }`$。

行为策略（采样时的策略）即 rollout 时的$`\pi_{\theta_k}`$，其$`\log\pi_{\theta_k}(a_t\mid s_t)`$缓存在 rollout buffer 里，在整个多 epoch 更新中不变（ “old log prob”）。当前要优化的$`\pi_\theta`$在每次参数更新后都会变，比值$`\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_k}(a_t\mid s_t)}`$在 epoch 之间会变化。

最大化用带 Adam 的随机梯度上升实现；算一次梯度要对$`T`$个$`\min`$式各自选起作用的分支再反向传播求梯度。若选到了被clip的分支（A为正$`\rho`$太大 或 A为负$`\rho`$太小），对$`\theta`$梯度为0；若选到了没被clip的分支，梯度为$`\dfrac{\hat{A}_t^{\text{GAE}}}{\pi_{\theta_k}(a_t\mid s_t)} \nabla_{\theta} \pi_\theta(a_t\mid s_t)`$，即：
$$
\nabla_{\theta} min()=
\begin{cases} 
    &\qquad\qquad\qquad\qquad\qquad\qquad0,  \quad&if \; clip \\
    &\hat{A}_t^{\text{GAE}} \cdot exp\bigl(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_k}(a_t\mid s_t)\bigr) \cdot \nabla_{\theta} \log\pi_\theta(a_t\mid s_t),  \quad &else
\end{cases} 
$$
**8.** 通过对均方误差回归拟合价值网络：
$$
\omega_{k+1} = \arg\min_\omega \frac{1}{T} \sum_{t=0}^{T-1} \left( v(s_t;\omega) - \hat{G}_t \right)^2
$$
最小化用梯度下降实现。

**9.** `end for` `epoch`

**10.** `end for`$`k`$

:::

回想 PPO 的目标函数是关于状态-动作分布的期望形式$`\mathbb{E}_{(s_t,a_t)\sim \pi_{\theta_{\text{old}}}}[min()]`$，在实现中，这个期望是无法直接计算的，因为并不知道由策略$`\pi_{\theta_{\text{old}}}`$所诱导的完整状态-动作分布。因此，**<font style="color:#DF2A3F;">通过与环境交互采样得到的轨迹来对该期望进行蒙特卡洛近似</font>**，这与 TRPO 中梯度 g 的计算类似。具体而言，在第$`k`$次策略更新时，通过运行策略$`\pi_{\theta_k}`$与环境交互，可以得到一条轨迹$`\tau = (s_0,a_0,R_1,\dots,s_{T-1},a_{T-1},R_T)`$，将每个时间步$`(s_t,a_t,R_{t+1})`$对应的$`[\;s_t, a_t, R_t, s_{t+1}, \log\pi_{\theta_k}(a_t|s_t), v(s_t;\omega_k),\text{if-terminate}\;]`$=【obs，action，reward，next_obs，log_prob，value，done】记录到rollout buffer 中，利用这些采样得到的数据，可以将原本的期望写成样本平均的形式：
$$
\mathbb{E}_{(s_t,a_t)\sim \pi_{\theta_{\text{old}}}}[min()]
\approx rac{1}{T} \sum_{t=0}^{T-1} min()\,.
$$
先按时间顺序采样：rollout 阶段必须按时间推进，s_t -> a_t -> R_t -> s_{t+1}，顺序没变。计算完 adv/ret 后，得到一堆“已标注”的样本 (obs, action, logp_old, adv, ret, value_old)。进入优化阶段时，把这些样本当作监督学习数据来做 SGD，这时不再需要“相邻时间步必须挨着训练”。所以“时间顺序信息”是用于构造目标（adv/ret），不会用于之后每一步梯度更新。直观类比：先按时间拍了一段视频（rollout），再从每一帧提取好标签（adv/ret），训练时随机抽帧喂模型；视频拍摄顺序没丢，只是喂给优化器的顺序可能会随机化。

##### 有并行环境
:::info
**1.** 输入：初始策略网络参数$`\theta_0`$，初始价值网络参数$`\omega_0`$；并行环境数量$`N=\text{num\_envs}`$，每次 rollout (固定交互采样)的时间步长$`M=\text{n\_steps}`$，故每轮采样总样本量（batch size）$`B=N\times M`$。

**2.** 创建向量化并行环境$`^{(注 11) }`$`vec_env`，一次 `step` 同时推进$`N`$个子环境，得到按环境维度堆叠的$`s_t^{(i)},a_t^{(i)},R_{t+1}^{(i)},\text{done}_t^{(i)}`$（$`i=1,\ldots,N`$）。

**3.** `for`$`k = 0, 1, 2, \ldots`$（第$`k`$次策略更新）`do`

**4.** 用当前策略$`\pi(\theta_k)`$在 `vec_env` 中连续采样$`M`$步。每一步都把$`N`$个环境的数据写入 rollout buffer，缓冲区张量形状常为：

+ `obs`：$`[M,N,\text{obs\_dim}]`$
+ `actions`:$`[M,N,\text{act\_dim}]`$
+ `rewards`:$`[M,N]`$
+ `dones`:$`[M,N]`$
+ `log_probs_old`:$`[M,N]`$
+ `values_old`:$`[M,N]`$

注意：若某个子环境 `done=True`，`vec_env` 会只重置该子环境并继续并行采样，不影响其他环境。

**5.** 基于第 4 步缓存的 `values_old`，对每个环境轨迹分别做 GAE，得到优势$`\hat{A}_{t}^{(i)}`$：
$$
\hat{A}_{t}^{(i)}=\delta_{t}^{(i)}+\gamma\lambda(1-\text{done}_t^{(i)})\hat{A}_{t+1}^{(i)},
\quad\\
\delta_t^{(i)}=R_{t+1}^{(i)}+\gamma(1-\text{done}_t^{(i)})v(s_{t+1}^{(i)};\omega_k)-v(s_t^{(i)};\omega_k).
$$
**6.** 计算目标价值（return / value target）：
$$
\hat{G}_{t}^{(i)}=\hat{A}_{t}^{(i)}+v(s_t^{(i)};\omega_k).
$$
于是得到 `advantage` 与 `value_target` 两个$`[M,N]`$张量。



**7.** 一旦上一步完成，接下来的更新就完全不依赖时序关系了，所以数据【obs，action，reward，log_prob，old_value】可以打乱，即使来自不同的环境的数据也能进入一个 minibatch。具体来说，将 rollout buffer 从$`[M,N,\cdots]`$展平为一个大 batch：$`[B,\cdots]`$（其中$`B=M\times N`$）。  
例如 `obs_flat:[B,obs_dim]`、`actions_flat:[B,act_dim]`、`adv_flat:[B]`、`logp_old_flat:[B]`。



**8.** `for` `epoch` `in` `epochs` `do`

+ 对 index$`[0,\ldots,B-1]`$打乱。
+ 按 `minibatch_size` 切分成若干 **minibatch**（每个小批大小为$`b`$，通常$`b \ll B`$）。
+ 每个 minibatch 单独前向、反向、更新参数；遍历完所有 minibatch 记为 1 个 epoch。

**9.** 在每个 minibatch 上最大化 PPO-Clip 策略目标：
$$
L_{\pi}(\theta)=
\frac{1}{b}\sum_{j=1}^{b}
\min\left(
\rho_j(\theta)\hat{A}_j,\;
(\;1\pm \epsilon\;)\hat{A}_j
\right),
$$
**10.** 在每个 minibatch 上最小化价值损失（可配 value clipping）：
$$
L_V(\omega)=\frac{1}{b}\sum_{j=1}^{b}\left(v(s_j;\omega)-\hat{G}_j\right)^2.
$$
最小化通常用 Adam。实践中常把策略损失、价值损失和熵正则合并为总损失一起优化。

**11.** `end for` `epoch`

**12.** `end for`$`k`$

:::

##### 


##### 流行 RL 库 PPO 参数对比
| 参数 | 动手学 RL | SB3 | tianshou | rsl_rl | skrl |
| --- | --- | --- | --- | --- | --- |
| branch / commit | main / 4a151a4b | master / 3246f506 | master / f2402056 | main / 00e13d1a | main / d19e5ed9 |
| rollout_buffer 名称。长度限制。 | transition_dict 充当。<br/>无默认长度限制，收集直到一局 episode 结束 | RolloutBuffer（或 DictRolloutBuffer）。<br/>固定容量 n_steps（默认 2048）× n_envs | ReplayBuffer / VectorReplayBuffer 充当 。<br/>buffer_size 为最大容量（mujoco 示例默认 4096），每次 collection 收集 collection_step_num_env_steps 步（默认 2048） | RolloutStorage。<br/>固定容量 num_steps_per_env（示例常用 24）× num_envs | Memory（如 RandomMemory）。<br/>memory_size 需与 rollouts 一致（默认 16）× num_envs |
| 旧策略 π(θ_k) 保持不变的时间步 | 一局 episode。<br/>与 episode 长度有关 | n_steps 步/环境（默认 2048）。<br/>与 episode 长度无关 | 一次 collection 的 collection_step_num_env_steps 步/环境（默认 2048）。<br/>与 episode 长度无关 | 一次 iteration 的 num_steps_per_env 步/环境（配置项，示例常用 24）。<br/>与 episode 长度无关 | rollouts 步/环境（默认 16）。<br/>与 episode 长度无关 |
| rollout_buffer 每步收集哪些信息 | states, <br/>actions, <br/>rewards, <br/>next_states, <br/>dones | observations, <br/>actions, <br/>rewards, <br/>episode_starts, <br/>values, <br/>log_probs,<br/>（GAE 后追加 advantages, returns；不存 next_obs/dones） | obs, <br/>act, <br/>rew, <br/>terminated, <br/>truncated, <br/>done, <br/>obs_next, <br/>info, <br/>policy；<br/>更新前由 critic 算 v_s，由 policy 算 logp_old | observations, <br/>actions, <br/>rewards, <br/>dones, <br/>values, <br/>actions_log_prob, <br/>distribution_params,<br/>（GAE 后追加 returns, advantages；不存 next_obs） | observations,<br/>states, <br/>actions, <br/>rewards, <br/>terminated, <br/>truncated, <br/>log_prob, <br/>values<br/>（更新前算 returns, advantages；不存 next_obs） |
| batch | 无 | 一次 rollout 全量数据：n_steps × n_envs（默认 2048×n_envs），用于 n_epochs 轮更新 | 一次 collection 收集的全量 transition（默认约 2048×n_envs 条） | 一次 iteration 全量数据：num_steps_per_env × num_envs | 一次 update 前 memory 全量：rollouts × num_envs（默认 16×n_envs） |
| mini_batch | 无 | batch_size（默认 64），n_epochs 轮内 shuffle 切分 | batch_size（默认 64），update_step_num_repetitions 轮内切分；设为 None 则整批更新 | 由 num_mini_batches（默认 4）均分全量 batch | 由 mini_batches（默认 2）均分全量 batch |


#### (4)动手学 PPO 源码
注意动手学 PPO 源码不使用并行环境，与上面无并行环境 PPO 算法流程的区别是，先用单步 TD target：$`\hat{G}_t=R_{t+1}+\gamma v(s_{t+1};\omega_k)`$来估计价值目标， 交换第 4、5 步顺序，GAE 再用$\hat{A}_t^{\text{GAE}}=\delta_t^V+\gamma\lambda\hat{A}_{t+1}^{\text{GAE}}=
 \hat{G}_t-v(s_t;\omega_k)+\gamma \lambda \hat{A}_{t+1}^{\text{GAE}}$。

##### 离散动作版本
```python
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs: int, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一批采样的数据被重复用来更新策略网络的次数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage   # shape:[T,1]
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断 # shape:[T,1]
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
$$
其中新旧策略比例的计算：

```python
old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach() 
log_probs = torch.log(self.actor(states).gather(1, actions)) 
ratio = torch.exp(log_probs - old_log_probs)
$$
##### 离散动作版本应用
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/gif/57851636/1780315185026-530033ee-2ca2-4851-80ed-889f91c6eb5e.gif)

车杆环境如图所示， 一辆小车上铰接了一根杆，智能体的任务是通过在每一 仿真步给小车一个向左或者向右的力(10N)，保持车上的杆竖直。若杆的倾斜度数过大，或者车子离初始位置左右的偏离程度过大，或者坚持时间到达 200 仿真步，则游戏结束。每坚持一步，智能体能获得分数为 1 的奖励，坚持时间越长，则最后的分数越高，坚持 200 步即可获得最高的分数。智能体的状态是一个维数为 4 的向量，每一维都是连续的，其动作是离散的，动作空间大小为 2，具体定义如下:

```plain
Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left   -10.0 N
        1     Push cart to the right  +10.0 N
$$
使用离散动作版本 PPO 训练该任务：

```python
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 10
hidden_dim = 128
gamma = 0.98

lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
print("check1")
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
print("check2")
$$
其中 env = gym.make(env_name)使用已经注册好的环境创建任务。 环境提前在 conda 环境的 gym 库注册，具体注册位置为 envs/horl/lib/python3.8/site-packages/gym/envs/__init__.py ，可见这里设置了max_episode_steps=200。为便于 debug，建议这里将 200 改为 较小的数如 3【注意修改后要重启Jupyter Kernel。 因为 gym 在 第一次 import gym 时 就把环境注册表读进内存了，之后再改 site-packages/gym/envs/__init__.py，当前 Kernel 里仍是旧值。】

```python
register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)
$$
max_episode_step改为 3 后，重启Jupyter Kernel，对训练代码 debug，可见 dones 在第三步变为了 false:

<!-- 这是一张图片，ocr 内容为：SPECIAL VARIABLES FUNCTION VARIABLES AY([-0.0 STATES .04456399. [1,0,1] ACTIONS [1.0,1.0,1.0] REWARDS FALSE,FALSE, DONES -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1779972416715-ca9824b7-751d-4431-a875-b84255240b5a.png)

在 site-packages/gym/envs/classic_control/cartpole.py查看环境的具体实现细节，照例实现了 step()，reset()，render()函数，其中 step()函数中进行了动力学解算，输入动作，输出奖励和状态：

```python
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}
$$
##### 连续动作版本应用
```python
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
$$
#### (5)PPO trick
在实际各种库的 PPO 实现中，用到的工程 trick 还非常多，不然 PPO 得效果可能比最原始的 REINFORCE 还烂，详见讨论：[https://www.zhihu.com/question/1997333117237736115](https://www.zhihu.com/question/1997333117237736115)

1. 向量化架构:利用向量化设计加速批量环境的交互和训练。

2. 权重的正交初始化与偏置的常数初始化:神经网络权重采用正交矩阵初始化，偏置使用固定常数初始化，提高训练稳定性。

3.Adam 优化器的 epsilon 参数:通过设置 Adam 的 epsilon 参数避免数值不稳定。

4.Adam 学习率退火:在训练过程中逐步降低学习率以改善收敛性能。

5. 广义优势估计（GAE）:使用 GAE 计算优势函数以减少方差并提高训练效率。

6. 小批量更新:将训练数据拆分为小批量进行多轮更新，提高样本效率和训练稳定性。

7. 优势函数归一化:对优势函数进行归一化以改善梯度更新的稳定性。

8. 截断的代理目标:使用截断策略目标函数来限制策略更新幅度，保证训练稳定。

9. 值函数损失截断:对值函数损失进行截断，防止过大更新导致训练不稳定。

10. 整体损失与熵奖励:总损失由策略损失、值函数损失及熵奖励组成，以平衡探索与利用。

11. 全局梯度裁剪:对梯度进行全局裁剪，防止梯度爆炸。

12. 调试变量:在训练过程中记录调试信息，便于分析和调试训练过程。

13. 策略网络与值函数网络的独立 MLP:策略网络与值函数网络使用独立的多层感知机结构，避免参数共享带来的干扰。

# 3、价值学习
## 3.1、Q-learning
## 3.2、DQN
### 3.2.1、原始 DQN
### 3.2.2、Double DQN
### 3.2.3、Dueling DQN
# 4、异策略
## 4.1、DDPG
## 4.2、SAC


_________________________________________

# 注
### (注 0) 符号使用
本文在符号的使用上尽量与 PPO 论文一致。

在回报的表示上，本文统一使用$`G_t`$，一些书籍也使用$`G_t`$或者$`U_t`$；在奖励的表示上，本文统一使用大写随机变量$`R_t`$（奖励函数仍记为小写$`r(s,a)`$）。角标$`\pi`$本文标在右上，如$`Q^\pi`$。凡此种种，读者结合上下文理解应不致混淆。

### (注 1) online  和 offline
**在 强化学习（Reinforcement Learning） 里，online 和 offline 指的是数据是如何获得和使用的。核心区别是：训练时是否与环境持续交互来获取新数据。**

**online: **智能体在训练过程中 不断与环境交互，实时收集新数据并更新策略。A2C,DDPG,PPO,SAC，绝大多数经典 RL 算法都是 online。

**offline:**  训练时 不能再和环境交互，只能用 已经收集好的固定数据集。如 CQL，IQL。

本文暂时只讨论 online, 最近 offline 的研究 也在兴起。

### (注 1-2) 蒙特卡洛法
即 通过 大量随机样本 来 近似 真实值。

### (注 1-3) model-based 和 -free
**Model-Based Reinforcement Learning（基于模型的强化学习）**  
	算法会学习或利用环境的动力学模型。环境模型通常表示为$`P(s_{t+1}, R_t \mid s_t, a_t)`$.意思是：在当前状态$`s_t`$下执行动作$`a_t`$，环境会以一定概率产生下一状态$`s_{t+1}`$和奖励$`R_t`$。通过学习或已知这个模型，智能体可以在内部对未来状态进行预测和规划，从而选择更优的动作。

**Model-Free Reinforcement Learning（无模型的强化学习**）  
	算法不去学习环境的动力学模型，而是直接通过与环境交互获得的数据来学习策略或价值函数。智能体根据经验数据不断更新策略，从经验数据直接学习最优策略，而不显式估计环境的状态转移概率或奖励模型。

可以用 机器人学习走路 举例。Model-Based机器人先学会：“如果我抬腿 10°，身体会往前移动 5 cm”即 学动力学模型，然后再计算怎样动作能走得最快。Model-Free机器人不管动力学，只是：尝试动作，看奖励，逐渐学会什么动作好，完全 trial-and-error。

大多数机器人项目都会用 **model-free 强化学习算法**，因为：高维动力学模型很难学；模型误差会破坏策略；仿真可以提供大量样本。

### (注 2) 回报随机性的来源
回报$`G_t = \sum_{i=t}^T \gamma^{(i-t)} R_{i+1} = R_{t+1}+\gamma\;R_{t+2}+\gamma^2\;R_{t+3}+...+\gamma^{T-t}\;R_T`$

**在时间 t，只确定了一件事：当前状态  **$`s_t`$** 以及选定的动作 **$`a_t`$。但后面的：

$s_{t+1},\, s_{t+2},\, \ldots 
$$
 a_{t+1},\, a_{t+2},\, \ldots 
$$
 R_{t+1},\, R_{t+2},\, \ldots$在 **时间 t 都还没发生**。

1、状态转移模型$`P`$随机：相同的$`s_t`$选取相同的动作$`a_t`$可能会转移到不同的$`s_{t+1}`$，导致$`S`$随机 ：$`s_{t+1} \sim P(\,\cdot \mid s_t, a_t\,)`$

2、若采用随机策略，则 动作$A
$`随机：`$a_{t} \sim \pi(\,\cdot \mid s_t)$

3、奖励函数$`r`$本身可能有随机性，如添加了奖励噪声等；奖励函数也可能没有随机性



### (注 3) 强化学习与传统优化算法
在理想化的理论条件下，许多传统优化方法可以解决强化学习问题。如果状态空间有限或可以枚举，策略参数空间是有限的，并且我们能够进行无限次采样，同时可以准确评估每一个策略的真实期望回报，并拥有充足的计算资源进行全局搜索，那么强化学习本质上可以被视为一个黑箱优化问题，即最大化策略的性能目标$`\max_{\pi} J(\pi)`$，  
其中$`J(\pi)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t R_t\right]`$。在这种极端假设下，无论是遗传算法、模拟退火、随机搜索还是其他无梯度优化方法，只要能够不断生成策略并评估其真实期望回报，就可以通过搜索找到使$`J(\pi)`$最大的策略。从理论上讲，只要搜索过程具有一致收敛性并且探索充分，最终都可以逼近最优解。

然而，在现实情形中，即使数据量非常大，也不能简单地认为所有优化算法都同样有效。

**<font style="color:#DF2A3F;">核心的点在数据采样效率。</font>** 强化学习的困难<font style="color:#DF2A3F;">并不在于</font>**<font style="color:#DF2A3F;">是否存在</font>**$`\max_{\pi} J(\pi)`$**<font style="color:#DF2A3F;">的解</font>****，**而<font style="color:#DF2A3F;">在于如何</font>**<font style="color:#DF2A3F;">高效估计这个目标函数</font>**。由于$`J(\pi)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t R_t\right]`$依赖于完整轨迹的采样，如果一种方法每次都要独立评估一个完整策略，并且不能利用轨迹中的中间状态转移信息，那么它的样本效率会非常低。相比之下，基于价值函数的方法利用贝尔曼方程$`V^{\pi}(s)=\mathbb{E}\left[r+\gamma V^{\pi}(s')\mid s\right]`$或动作价值函数的递归结构$`Q^{\pi}(s,a)=\mathbb{E}\left[r+\gamma \max_{a'}Q^{\pi}(s',a')\mid s,a\right]`$，通过自举更新在单条轨迹内部反复利用数据，从而显著提高样本利用率。这种结构化更新方式比单纯的整策略评估更加高效。更进一步地说，强化学习并不是一个普通的静态优化问题。传统优化通常表示为$`\max_x f(x)`$，其中目标函数$`f(x)`$是固定的。而在强化学习中，优化的是$`\max_{\pi}\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t R_t\right]`$，但这个期望中的状态分布本身依赖于策略$`\pi`$。也就是说，策略改变会改变状态访问分布，从而改变目标函数本身。这种“策略与数据分布耦合”的特性，使强化学习具有动态结构。基于动态规划和时间差分学习的方法利用贝尔曼算子进行逐步逼近，而<font style="color:#DF2A3F;">纯粹的黑箱优化方法则忽略了这种时间结构和状态转移关系</font>，因此在复杂高维问题中往往效率极低。



### (注 4) 强化学习与监督学习
【 若将这里的“状态”对应为监督学习中的“示例”、“动作”对应为“标记”,则可看出,强化学习中的“策略”实际上就相当于监督学习中的“分类器”(当动作是离散的)或“回归器”(当动作是连续的),模型的形式并无差别,但不同的是,在强化学习中并没有监督学习中的有标记样本(即“示例-标记”对),换言之,没有人直接告诉机器在什么状态下应该做什么动作,只有等到最终结果揭晓,才能通过“反思”之前的动作是否正确来进行学习.因此,强化学习在某种意义上可看作具有“延迟标记信息”的监督学习问题。】

《机器学习》周志华



### (注 4-2) 梯度上升
对二元函数$`z=f(x,y)`$在 定义域内一 点($`x_0,y_0`$) 求梯度 grad，得向量 
$$
\nabla f(x_0,y_0)=
\left(
\left.\frac{\partial f}{\partial x}\right|_{(x_0,y_0)},
\left.\frac{\partial f}{\partial y}\right|_{(x_0,y_0)}
\right)
$$
梯度的方向是函数值$`z`$在该点增长最快的方向，梯度的反方向是$`z`$在该点下降最快的方向，梯度的模是$`z`$在该点该方向单位距离增长/下降的距离。

与梯度垂直的方向$`z`$的 增长率为 0（ 方向导数为0）。

同理可以推广到多元函数$`f(\theta)`$，当$`dim(\theta)>2`$。

### (注 5) 策略梯度定理证明
见 王树森 7.3 ，或动手学 9.6，或 sutton 13.2
$$
\nabla_\theta \mathbb{E}_S\left[V^{\pi(\theta)}(S)\right]=\mathbb{E}_S\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; {\theta})}\left[\nabla_\theta \ln \pi(A \mid S ; {\theta}) \cdot Q^{\pi(\theta)}(S, A)\right]\right]
$$
完整的证明非常复杂，涉及较多的概率论和随机过程知识。

带基线的策略梯度定理，

见王树森 8.4 或 sutton 13.4



### (注 5-0) 策略网络梯度计算
策略网络$`\pi_\theta(a|s)`$输出的是分布的参数，以高斯分布为例，$`\mathcal{N}(\mu_\theta(s), \sigma_\theta(s))`$的参数（$`\mu,\sigma`$）。  
要计算的梯度是$`\nabla_\theta \ln \pi(a|s; \theta)`$。

根据链式法则，这个梯度确实可以分解为两部分：
$$
\nabla_\theta \ln \pi(a|s; \theta) = \underbrace{\frac{\partial \ln \pi(a|s; \mu, \sigma)}{\partial (\mu, \sigma)}}_{\text{分布对参数的导数}} \cdot \underbrace{\frac{\partial (\mu, \sigma)}{\partial \theta}}_{\text{网络输出对网络参数的导数}}
$$
这里有两个关键点：

1. **第一部分（解析解）**：$`\frac{\partial \ln \pi}{\partial (\mu, \sigma)}`$是概率密度函数对数形式关于其统计量（均值和方差）的导数。对于高斯分布，这是一个有明确解析公式的数学表达式，不需要神经网络参与。
    - 例如，对于一维高斯，$`\ln \pi(a|\mu, \sigma) = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(a-\mu)^2}{2\sigma^2}`$。
    - 对其求导得到关于$`\mu`$和$`\sigma`$的表达式（如$`\frac{a-\mu}{\sigma^2}`$等）。
2. **第二部分（反向传播）**：$`\frac{\partial (\mu, \sigma)}{\partial \theta}`$是神经网络本身的梯度。因为$`\mu`$和$`\sigma`$是网络$`f_\theta(s)`$的输出，这部分完全通过标准的反向传播算法（Backpropagation）自动计算。

**结论**：深度学习框架（如 PyTorch, TensorFlow）会自动将这两部分连接起来。当定义好损失函数（通常是$`-\ln \pi(a|s) \cdot G_t`$或类似形式）并调用 `.backward()` 时，框架会自动执行“先对分布参数求导，再通过网络反向传播”的过程。

代码实现视角（以 PyTorch 为例）

在实际代码中，通常不会手动写出上面的链式法则公式，而是利用框架的自动微分（Autograd）功能：

```python
# 策略网络：输入状态 s，输出分布参数 mu, sigma
class PolicyNetwork(nn.Module):
    def forward(self, state):
        mu = self.mu_layer(state)
        sigma = self.sigma_layer(state) # 通常加 softplus 保证为正
        return mu, sigma

# 构建策略梯度损失 (负号是因为我们要最大化奖励，即最小化负损失)
loss = -(log_prob * advantage).mean()

#  反向传播
optimizer.zero_grad()
loss.backward() 
# 【关键步骤】：
# 此时，autograd 引擎自动执行了你描述的逻辑：
# 1. 计算 loss 对 log_prob 的导数
# 2. 计算 log_prob 对 (mu, sigma) 的导数 (分布内部的数学公式)
# 3. 计算 (mu, sigma) 对 theta 的导数 (神经网络反向传播)
# 最终得到 grad(loss)/grad(theta)

optimizer.step()
$$
### **(注 5-1) 自举  **bootstrapping 
**<font style="color:#DF2A3F;">自举bootstrapping ，</font>**这个词在统计学经常出现。它的字面意思是“拔自己的鞋带，把自己举起来”。虽然自举乍看起来不现实，但是在统计和机器学习是可以做到自举的, 且非常常用。在强化学习中，“自举”的意思就是“用一个估算去更新同类的估算”，类似于“自己把自己给举起来”。 时间差分 TD 就是一种自举。

### **(注 5-1-1) 价值估计分类** 
有一些书，如王树森，将价值估计分为蒙特卡洛法和神经网络法，有 概念层级不一致的问题。

正确的层级应该是：

Value estimation methods

 ├── Monte Carlo

 └── Temporal Difference

        ├── TD(0)

        ├── n-step TD

        └── TD(λ) / GAE



Function representation

 ├── Tabular

 └── Function approximation

        ├── Linear approximation

        └── Neural networks

### (注 5-2) 目标函数重要性采样证明
$$
\begin{aligned}
V_\pi(s)
&= \mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[Q_\pi(s, A)\right] \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \cdot Q_\pi(s, a) \\
&= \sum_{a \in \mathcal{A}} \pi\left(a \mid s ; \boldsymbol{\theta}_{\text {old }}\right)
\cdot \frac{\pi(a \mid s ; \boldsymbol{\theta}_{\text{new }})}{\pi\left(a \mid s ; \boldsymbol{\theta}_{\text {old }}\right)} \cdot Q_\pi(s, a) \\
&= \mathbb{E}_{A \sim \pi\left(\cdot \mid s ; \boldsymbol{\theta}_{\text {old }}\right)}\left[
\frac{\pi(A \mid s ; \boldsymbol{\theta}_{\text{new }})}{\pi\left(A \mid s ; \boldsymbol{\theta}_{\text {old }}\right)} \cdot Q_\pi(s, A)\right].
\end{aligned}
$$
### (注 5-3) 目标函数差值等式证明
 见动手学 11.2

### (注 5-4) KL 散度
**一、定义**

KL散度（Kullback–Leibler Divergence）是信息论中的一个量，用来衡量：

**用分布 Q 去近似真实分布P 时，会损失多少信息。**

**数学定义为：**
$$
D_{KL}(P\parallel Q)
=
 P(x_1)\log\frac{P(x_1)}{Q(x_1)}+P(x_2)\log\frac{P(x_2)}{Q(x_2)}+...$，

连续情况$D_{KL}(P\parallel Q)
=
\int_{-\infty}^{\infty}
P(x)\log\frac{P(x)}{Q(x)}\,dx
$$
**二、性质**

KL 散度是不对称的$`D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)`$

KL 散度>=0.

KL越大 → 两个分布差异越大  
KL=0 → 两个分布完全一样

**三、理解与推导**

**信息论推导**

1. 信息量的概念  
信息论里，一个事件$`x`$的信息量为$`I(x) = -\log P(x)`$
概率越小，信息量越大。

 2. 用错误概率编码  
假设真实概率为$`P(x)`$，但用$`Q(x)`$来编码。  
真实平均信息量是，**信息熵**：$`H(P) = - \sum_x P(x) \log P(x)`$，信息熵告诉我们：**一个事件发生的不可预测性有多大**，或者说**平均需要多少比特的信息来描述这个事件**。  
如果用$`Q`$来编码，平均信息量变为，**交叉熵**：$`H(P,Q) = - \sum_x P(x) \log Q(x)`$,交叉熵衡量的是**两个概率分布之间的差异**。

    - 一个分布是“真实分布” P（目标分布），
    - 一个分布是“预测分布” Q（模型输出）。

交叉熵告诉我们：**用 Q去编码 P时，平均需要多少比特的信息**。

3. 两者的差  
信息损失：$H(P,Q) - H(P)
= -\sum_x P(x) \log Q(x) + \sum_x P(x) \log P(x)
= \sum_x P(x) \log \frac{P(x)}{Q(x)}
= D_{KL}(P \parallel Q)$

**四、KL 散度的改进：杰森-香农散度 (Jensen-Shannon Divergence, JSD)**

JSD是基于KL散度构建的，旨在解决KL散度的非对称性问题，使其成为一个更稳定的距离度量。

 核心思想：JSD通过引入一个“中间人”——即两个分布的平均分布 M = (P + Q) / 2，来衡量P和Q各自到这个中间点的距离。

    数学定义：

    JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)

    其中 M = 0.5 * (P + Q)

    主要特点：

        对称性：JSD(P || Q) 总是等于 JSD(Q || P)。这使得它更像一个真正的“距离”。

        有界性：JSD的值域被限制在 之间（当使用以2为底的对数时），这使得不同场景下的结果更具可比性。

        缓解零值问题：通过引入平均分布M，JSD在一定程度上缓解了KL散度中对零值敏感的问题，因为M(x)在P(x)或Q(x)不为零时通常也不为零。  


### (注 6-0) Hessian 矩阵
Hessian 矩阵（黑塞矩阵，海森矩阵）是一个由函数的二阶偏导数组成的方阵，主要用于多变量微积分和优化理论中。

1. 定义  
假设有一个实值函数$`f(x_1, x_2, \dots, x_n)`$，如果它的所有二阶偏导数都存在，那么它的 Hessian 矩阵$`H(f)`$是一个$`n \times n`$的矩阵，其元素定义为：
$$
H(f)_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$
写成矩阵形式如下：
$$
H(f) = 
\begin{bmatrix}
    \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
    \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$
注意：如果函数$`f`$的二阶偏导数连续（即满足 Clairaut 定理），则混合偏导数相等（$`\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}`$），此时 Hessian 矩阵是对称矩阵。

2. 主要用途

A. 判断极值点性质（多元函数）  
在单变量微积分中，我们用二阶导数$`f''(x)`$判断极大值或极小值。在多变量中，Hessian 矩阵扮演同样的角色。  
当梯度$`\nabla f = 0`$时（即找到驻点），通过检查 Hessian 矩阵的正定性来判断该点的性质：

+ 正定矩阵（所有特征值$`> 0`$）：该点是局部极小值（Local Minimum）。
+ 负定矩阵（所有特征值$`< 0`$）：该点是局部极大值（Local Maximum）。
+ 不定矩阵（特征值有正有负）：该点是鞍点（Saddle Point）。
+ 半正定或半负定：测试失效，需要更高阶的信息来判断。

B. 优化算法（如牛顿法）  
在机器学习和数值优化中，Hessian 矩阵描述了损失函数的曲率。

+ 牛顿法 (Newton's Method) 利用 Hessian 矩阵来更新参数，比仅使用梯度的一阶方法（如梯度下降）收敛更快，因为它考虑了函数的弯曲程度。
+ 更新公式为：$`x_{\text{new}} = x_{\text{old}} - H^{-1} \nabla f`$。
+ 缺点：计算和存储$`n \times n`$的 Hessian 矩阵及其逆矩阵计算量巨大（复杂度$`O(n^3)`$），因此在高维问题（如深度学习）中，通常使用拟牛顿法（如 BFGS）或只使用一阶梯度的方法。

C. 泰勒展开  
在多变量函数的二阶泰勒展开中，Hessian 矩阵出现在二次项中，用于近似函数在某点附近的形态：
$$
f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T H(\mathbf{x}) \Delta \mathbf{x}
$$
### (注 6) Fisher 矩阵
Fisher 信息矩阵（Fisher Information Matrix, 简称 FIM）是统计学和信息论中的一个核心概念，用于衡量一个概率分布或统计模型中所包含的关于未知参数的信息量。  
简单来说，它告诉我们：**通过观测数据，我们能够多精确地估计出模型的参数。**

1. 直观理解
+ 信息量 vs. 不确定性：**Fisher 信息量越大，意味着数据中关于参数的信息越丰富，我们对该参数的估计就越精确（方差越小）**。反之，如果 Fisher 信息量很小，说明数据很“模糊”，很难确定参数的真实值。
+ 曲率解释：从几何角度看，Fisher 矩阵描述了对数似然函数（Log-Likelihood）在真实参数值附近的曲率（弯曲程度）。
    - 如果对数似然函数在峰值处非常尖锐（曲率大），说明稍微偏离真实参数，似然度就会急剧下降，因此我们能很精确地定位参数（高 Fisher 信息）。
    - 如果峰值很平坦（曲率小），说明很多不同的参数值都能产生相似的数据，因此估计不准（低 Fisher 信息）。
3. 数学定义  
假设有一个概率密度函数$`f(x; \theta)`$，其中$`\theta`$是待估计的参数向量（$`\theta = [\theta_1, \theta_2, \dots, \theta_k]^T`$）。  
Fisher 信息矩阵$`I(\theta)`$是一个$`k \times k`$的矩阵，其第$`(i, j)`$个元素定义为：
$$
I(\theta)_{ij} = \mathbb{E}\left[ \left( \frac{\partial}{\partial \theta_i} \ln f(X; \theta) \right) \left( \frac{\partial}{\partial \theta_j} \ln f(X; \theta) \right) \bigg| \theta \right]
$$
或者，在正则性条件下（二阶导数存在且积分与求导可交换），它可以表示为对数似然函数二阶导数的负期望：
$$
I(\theta)_{ij} = -\mathbb{E}\left[ \frac{\partial^2}{\partial \theta_i \partial \theta_j} \ln f(X; \theta) \bigg| \theta \right]
$$
其中：

+$`\ln f(X; \theta)`$是对数似然函数。
+$`\frac{\partial}{\partial \theta} \ln f(X; \theta)`$称为得分函数（Score Function）。Score Function表示：**数据对参数变化的“敏感度”。**如果 Score 的方差很大：说明不同样本给出的“参数方向信息”变化很大→ 数据对参数很敏感→ **信息多。**如果 Score 方差小：→ 数据对参数不敏感→ **信息少。**所以：**Score 的波动程度（方差） = Fisher 信息。**

 4. Fisher 矩阵在物理和机器学习中的应用

+ **统计估计**：Fisher信息可以用于构建更精确的估计方法，如 **最小方差无偏估计（MVUE）**，Fisher信息越大，估计的方差越小。
+ **机器学习中的优化**：在深度学习和强化学习中，标准的梯度下降法（SGD）使用梯度来更新模型参数，但它假定每个参数的变化对损失函数的影响是相等的。而 **自然梯度下降**（Natural Gradient Descent）则使用Fisher信息矩阵来调整梯度更新的方向。具体来说，使用 **Fisher信息的逆矩阵** 来调整梯度，这样能够考虑参数空间的几何结构，从而更加有效地指导优化过程。通过这种方法，优化算法能够更快地收敛，尤其是在有多个局部最优解的复杂模型中。
+ **量化信息量**：Fisher信息量越大，表示数据对参数的估计越精确，反之则精度较低。
+ **参数估计的理论界限**：Fisher信息矩阵还可以帮助推导 **Cramér-Rao下界**，它给出了估计参数时可能达到的最低方差。

  5.Fisher 矩阵和 Hessian 矩阵的关系

| 特性 | Hessian 矩阵 ( H) | Fisher 信息矩阵 ( I ) | 关联 |
| --- | --- | --- | --- |
| **定义** | ∇2ln⁡L(θ;X) | −E[∇2ln⁡L(θ;X)] | 满足 I=−E[H] |
| **依赖数据** | 依赖具体样本 X | 依赖整体分布$P(X) |  |
| **正定性** | 不一定 (可能不定) | 总是半正定 | - |
| **计算来源** | 二阶导数 | 对数似然函数 一阶导数平方的期望 或 二阶导数期望 |  |
| **优化角色** | 牛顿法的核心 | 自然梯度法的核心 | 用 I 近似 H 以获得正定性 |


**详见：**[**https://zhuanlan.zhihu.com/p/589273267**](https://zhuanlan.zhihu.com/p/589273267)

****



****

### **(注 7) 共轭梯度法(Conjugate Gradient，CG) 解 正定矩阵方程**
<!-- 这是一张图片，ocr 内容为：问题背景 通常我们有一个对称正定矩阵A和一个向量B,要解线性方程: AAC B A E RNXN BE RN CER"是我们要求的未知向量 如果A很大,很稀疏(SPARSE),直接用高斯消元或者 U分解会非常耗时或者占用内存,这时候可以用送 代法,而共轭梯度法就是一种专门针对对称正定矩阵的高效迭代方法. 基本思想 共轭梯度法从梯度下降法出发,但引入了"共轭方向"的概念,使每一步都尽量沿着独立方向前进,避免像管. 通梯度下降那样ZIG-ZAG荡,从而加快收敛. 我们可以把方程A2C6转化成优化问题: MIN F(AE) - AT AA - BT E 梯度: VF(AC)B CG方法沿着共轭方向PI.更新 CK+12K+QKPK 其中0K是步长,选择方式保证沿PK:方向梯度下降最快. -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773134928758-549d3cf3-b2ff-4ee4-bf87-9a9b9cc50eeb.png)<!-- 这是一张图片，ocr 内容为：共轭方向的定义 3 两个向量P和Q对矩阵 A来说是A-共轭,如果: P ' A 0 CG方法每一步都沿着与之前方向A-共轭的方向前进,这样保证每次迭代减少误差在不同方向上互不干扰. 共轭梯度法算法步骤 初始化: 初始猜测,ROB-A2C0,PORO - 0AC 2.对K0,1,2,...重复: 1. 计算步长: PTTK PAAPK 2.更新解: CK+1 CKPK 3.更新残差: TK+1 TK ARAPK HA'H+1 4.检查收敛条件,如果 足够小,则停止 5.更新共轭方向: 10A+1 BE PK+1 RK+1+BKPW 3.输出2K+1作为近似解. <特点:在几维空间里,理论上最多N步就能精确求解(对精度敏感的浮点计算可能稍多). -->
![](https://cdn.nlark.com/yuque/0/2026/png/57851636/1773134981554-843460a4-75b3-43df-a493-6329a2952114.png)



### **(注 8) TD(λ)**和资格迹
**<font style="color:#DF2A3F;">TD(λ) 和资格迹是比较古早（1988）的强化学习概念，在现代强化学习中多被 GAE (2015 )替代了</font>**，然而一些资料还是会提及，且**<font style="color:#DF2A3F;">GAE 显然参考了其中的 λ 加权</font>**。在很多描述 TD(λ)资料中，回报用 G 而不是 R 表示；本文统一用 G 表示回报。

TD(λ) 的表达式通常有两种视角：前向视角（Forward View）和后向视角（Backward View）。前向视角定义了理论上的更新目标，后向视角提供了实际可执行的在线算法步骤。

**1、 前向视角：λ-回报 (The λ-return)**  
	这是 TD(λ) 的理论定义。它定义了我们要逼近的目标值$`G_t^\lambda`$。**  
**$`G_t^\lambda`$** 是所有 **$`n`$**-步回报 **$`G_{t:t+n}`$** 的加权平均**，权重由$`\lambda`$控制。
$$
G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}
$$
其中：

                -$`G_{t:t+n}`$**<font style="color:#DF2A3F;"> 是 </font>**$`n`$**<font style="color:#DF2A3F;">-步回报</font>**：
$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
$$
TD(λ)即 **加权平均多个 n-step return**

| n-step |$`G_{t:t+n}`$| 权重 |
| --- | --- | --- |
| 1-step |$`R_{t+1} + \gamma V(S_{t+1})`$| (1−λ) |
| 2-step |$`R_{t+1} + \gamma R_{t+2}  + \gamma^2 V(S_{t+2})`$| (1−λ)λ |
| 3-step |$`R_{t+1} + \gamma R_{t+2}  + \gamma^2 R_{t+3} + \gamma^3 V(S_{t+3})`$| (1−λ)λ^2 |


λ = 0.2         1-step 0.8 	2-step 0.16     3-step 0.032

λ = 0.9	1-step 0.1    2-step 0.09	    3-step 0.081	

TD 目标形式：
$$
y_t^{\lambda}
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}
\left(
\sum_{i=0}^{n-1} \gamma^i R_{t+i}
+
\gamma^n V(s_{t+n})
\right)
$$
_(注：如果回合在 _$`t+n`$_ 之前结束，则 _$`n`$_ 取到回合结束，且最后一项为0)_

                -$`\lambda \in [0, 1]`$：衰减参数。
                -$`(1-\lambda)`$：归一化系数，确保权重之和为 1（当$`\lambda < 1`$时）。

更新规则（理论版）：
$$
V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t^\lambda - V(S_t) \right]
$$
缺点：这个公式需要知道未来的所有奖励才能计算$`G_t^\lambda`$，因此无法用于在线实时学习。

**2、 后向视角：资格迹 (Eligibility Traces)**  
	这是 TD(λ) 的实际算法实现。它引入了资格迹$`E_t(s)`$，使得我们可以每走一步就进行更新，而无需等待未来。 机制：为每个状态（或状态 - 动作对）维护一个追踪变量$`E_t(s)`$。<font style="color:#DF2A3F;">资格迹标记了哪些状态对当前的预测误差“负有责任”</font>。如果一个状态最近被频繁访问且迹值很高，那么当前的误差就会更多地归因于它，从而对其进行大幅修正。

A. 核心变量定义

                - TD 误差 ($`\delta_t`$)：
$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$
_(如果是估计 Q 值，则 _$`\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)`$_)_

                - 资格迹 ($`E_t(s)`$)：  
对于每一个状态$`s`$（或状态 - 动作对），维护一个迹值。

累积迹 (Accumulating Trace) - 最常用：
$$
E_t(s) = 
\begin{cases} 
\gamma \lambda E_{t-1}(s) + 1 & \text{if } s = S_t \\
\gamma \lambda E_{t-1}(s) & \text{if } s \neq S_t 
\end{cases}
$$
替换迹 (Replacing Trace) - 有时性能更好：
$$
E_t(s) = 
\begin{cases} 
1 & \text{if } s = S_t \\
\gamma \lambda E_{t-1}(s) & \text{if } s \neq S_t 
\end{cases}
$$
B. 更新规则（实际执行版）  
在每一步$`t`$，计算出$`\delta_t`$后，对所有状态$`s`$进行更新：
$$
V(s) \leftarrow V(s) + \alpha \cdot \delta_t \cdot E_t(s)
$$
**3、 例子**

比如，假设机器人在时间序列：s1 → s2 → s3 → s4 → reward

普通 TD → 只更新 s3

TD(λ)：reward → 更新 s3 → 更新 s2 → 更新 s1

但影响逐渐衰减：s3 影响最大，s2 次之，s1 最小

衰减因子：γλ

**4、TD(1)和TD(0)**

观察TD(λ)的公式$y_t^{\lambda}
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}
\left(
\sum_{i=0}^{n-1} \gamma^i R_{t+i}
+
\gamma^n V(s_{t+n})
\right)$

可知，λ=0 时，就是普通单步 TD$`(0^0=1)`$,$`\hat{y}_t = R_t + \gamma V(s_{t+1})`$

λ->1 时，权重逐渐向更大的 n 集中，极限情况下只剩 最长的 return，就是蒙特卡洛。 

**TD(0) ----------- TD(λ) ----------- TD(1)  
****只看1步  -----------混合多步  ----------- 看完整episode**

### **(注 9)  无偏估计和**<font style="color:rgb(51, 51, 51);">γ -just</font>
<font style="color:rgb(51, 51, 51);">1、 用</font>$`\hat\theta`$<font style="color:rgb(51, 51, 51);">估计</font>$`\theta`$<font style="color:rgb(51, 51, 51);">,如果满足</font>$`\mathbb{E}(\hat\theta)=\theta`$<font style="color:rgb(51, 51, 51);">,则估计是无偏估计。</font>

<font style="color:rgb(51, 51, 51);">如，</font><font style="color:rgb(6, 10, 38);">用样本均值 </font>$`\bar{x}`$<font style="color:rgb(6, 10, 38);"> 来估计总体均值 </font>$`\mu`$<font style="color:rgb(6, 10, 38);"> 时是无偏的，因为</font>$`\mathbb{E}(\bar{x})=\mu`$

<font style="color:rgb(51, 51, 51);">2、γ -just 即</font>**<font style="color:rgb(51, 51, 51);">代入策略梯度无偏</font>**<font style="color:rgb(51, 51, 51);">，是 GAE 论文提出的，</font>**<font style="color:#DF2A3F;"> 如果一个优势函数估计器是 γ-just 的，那么它用于策略梯度时是无偏的</font>**<font style="color:rgb(51, 51, 51);">。 </font>

<font style="color:rgb(51, 51, 51);"> 即满足：</font>
$$
\mathbb{E}_{s_{0:\infty},\,a_{0:\infty}}
\left[
\hat{A}_t
\nabla_{\theta}\log \pi_{\theta}(a_t \mid s_t)
\right]
=
\mathbb{E}_{s_{0:\infty},\,a_{0:\infty}}
\left[
A^{\pi,\gamma}(s_t,a_t)
\nabla_{\theta}\log \pi_{\theta}(a_t \mid s_t)
\right].
$$
<font style="color:rgb(51, 51, 51);">的优势函数估计</font>$`\hat{A}_t`$<font style="color:rgb(51, 51, 51);">是γ -just 的。</font>

这个条件也可以写成策略梯度定理中熟悉的$`\mathbb{E}_S\mathbb{E}_A`$形式：
$$
\mathbb{E}_{S \sim d_{\gamma}^{\pi}}
\left[
\mathbb{E}_{A \sim \pi(\cdot \mid S)}
\left[
\hat{A}(S,A)\nabla_{\theta}\log \pi(A \mid S)
\right]
\right]
=
\mathbb{E}_{S \sim d_{\gamma}^{\pi}}
\left[
\mathbb{E}_{A \sim \pi(\cdot \mid S)}
\left[
A^{\pi}(S,A)\nabla_{\theta}\log \pi(A \mid S)
\right]
\right]
$$
### **(注 10)  对数概率**
事实上，torch 官方库只对分布提供 log_prob 函数，而没有像 log_prob 那样在 torch.distributions.Distribution 基类里统一提供的 .prob() 方法。官方接口里和“概率”相关、最常用的就是：.log_prob(value)：对数概率（离散是 log 质量；连续是 log 密度）若要“普通概率/密度”，一般用：torch.exp(dist.log_prob(x))注意：对连续分布（如 Normal）这是概率密度 p(x)，不是单点概率；离散分布才是质量函数意义下的概率。有些别的库会叫 prob，但在 PyTorch 的 distributions 里，标准做法就是 log_prob + 需要时再 exp。



### **(注 11)  向量化并行环境**
「并行」和「向量化」是两件事，只是常被放在一起说。

**并行**（在RL资料中，这个词的含义更接近于操作系统中的并发）：$`N`$份环境实例一起参与采样。是否在**同一物理时刻**真正同时执行，取决于底层实现，见下文。

**向量化**：数据和控制流按带 batch 维的张量组织。这里的「向量化」不是线性代数里的向量，也不完全是 CPU 的 SIMD，而是：**把 **$`N`$** 个环境的数据堆成带 batch 维的张量，用一次 API、一次前向传播处理整批。**

#### 对比：非向量化 vs 向量化
非向量化（一次只和一个环境打交道）：

```python
state = env.reset()           # 一个 state，shape 比如 (4,)
action = policy(state)        # 一个 action
next_state, r, done, _ = env.step(action)  # 标量 reward、一个 bool
$$
向量化（一次和$`N`$个环境打交道，接口和数据形状都变了）：

```python
obs = vec_env.reset()         # shape: [N, obs_dim]
actions = policy(obs)         # shape: [N, act_dim]，一次前向算 N 个动作
obs, rewards, dones, _ = vec_env.step(actions)
# rewards: [N], dones: [N]
$$
|  | 单环境 | 向量化 |
| --- | --- | --- |
| obs | 一条样本 | N 条样本堆成矩阵 [N, obs_dim] |
| step 调用 | 每个 env 各调一次 | 一次 step 推进 N 个 env |
| 策略网络 | 输入 [obs_dim] | 输入 [N, obs_dim]，batch 推理 |
| buffer | 一条条 append | 直接写 [M, N, ...] 张量 |


#### 为什么要叫「向量化」
名字来自 NumPy/PyTorch 的习惯：把$`N`$个标量/小数组 stack 成一行或一列，用矩阵运算代替 `for i in range(N)` 循环。

```python
# 非向量化：循环 N 次
for i in range(N):
    a[i] = policy(obs[i])

# 向量化：一次算完
actions = policy(obs_batch)   # obs_batch: [N, obs_dim]
$$
GPU 上 batch 越大，利用率通常越高——这是 PPO 采样快的重要原因。

#### 「并行」和「向量化」的关系
+ **并行** =$`N`$份环境一起采数据
+ **向量化** = 这$`N`$份数据不拆成$`N`$次标量调用，而是堆成 [N, ...]（再和$`M`$步合成 [M, N, ...]），一次 step、一次网络前向处理整批

二者可以独立存在：

|  | 无并行 | CPU 多进程 | GPU 仿真并行 |
| --- | --- | --- | --- |
| 无向量化 | 动手学 RL（单环境） | —（A3C 等分布式 RL，非 PPO vec_env 路线） | —（GPU 仿真天然返回 batch 张量） |
| 向量化 | **环境侧**：Gymnasium SyncVectorEnv / SB3 DummyVecEnv / tianshou DummyVectorEnv（**单线程串行** step；API 批量，可以有 num_envs>1）   **算法侧**：SB3 / tianshou / skrl 消费 batch 数据 | **环境侧**：Gymnasium AsyncVectorEnv / SB3 SubprocVecEnv / tianshou SubprocVectorEnv（**多核：真并行**；**单核：时间片并发，非真并行**）   **算法侧**：SB3 / tianshou / skrl 消费 batch 数据 | **环境侧**：Isaac Gym / Isaac Lab / mjlab 等（GPU **真并行** + batch 张量由仿真器提供）   **算法侧**：rsl_rl / skrl 消费 batch 数据 |


**小结**：向量化强调「batch 形状 + batch 接口 + batch 计算」；并行强调多个环境一起在跑。

#### 并行底层原理
从「串行」、「并发」到「真并行」：

| 程度 | 含义 | 典型例子 |
| --- | --- | --- |
| **串行** | 同一时刻只推进 1 个 env | SB3 DummyVecEnv 内部 for 循环 |
| **并发** | 逻辑上「同时在跑」，物理上交替执行 | 单核 CPU + 多进程（OS 时间片轮转） |
| **真并行** | 同一时刻多份计算单元各算各的 | 多核 CPU + 多进程；GPU SIMT batch 仿真 |


|  | CPU 多进程 | CPU 多线程 | GPU 并行（SIMT 模型，CUDA 编程） |
| --- | --- | --- | --- |
| 并行单位 | N 个 **进程**（各自独立程序） | N 个 **线程**（同一进程内） | N 个 **GPU 线程**（kernel 内，按 warp 编组） |
| 规模 | 几个～几十个 | 几个～几十个 | 几千～几万个 |
| 是否真并行 | **单核**：否，OS 时间片**并发**（同一时刻只跑 1 个进程）   **多核**：是，最多同时跑「核数」个进程 | **单核**：通常并发；CPU 密集还受 GIL 等限制   **多核**：可部分真并行 | **是**，硬件 SIMT 大规模数据并行 |
| 执行独立性 | 完全独立，各自地址空间，可跑不同代码 | 较独立，可分支；共享地址空间 | 同一 warp 内必须走同一指令（分支 diverge 会变慢） |
| 内存与数据 | 进程间内存隔离；靠 IPC / pickle / 共享内存传数据，再 stack 成 batch | 共享堆；各自栈；同步（锁等）较复杂 | 状态排成连续 **显存** 张量 `[num_envs, ...]`（SoA），访存合并（coalesced） |
| 调度 | **OS** 调度进程（时间片轮转或绑核） | **OS** 调度线程 | **GPU 硬件** 调度 warp，不经 OS |
| 适合 | 多核 CPU 上跑多个 gym 环境；进程隔离（一个 env 崩不影响别的） | 任务杂、分支多、I/O 多；RL vec_env 中较少作为主方案 | **同一公式重复算很多遍**（物理积分、碰撞、矩阵乘） |
| RL 中的例子 | SB3 SubprocVecEnv / Gymnasium AsyncVectorEnv / tianshou SubprocVectorEnv | — | Isaac Gym / Isaac Lab / mjlab；底层 PhysX GPU + CUDA kernel |


**SIMT**（Single Instruction, Multiple Threads）是 GPU 的线程执行模型；**CUDA** 是 NVIDIA 提供的、在该模型上编写/launch kernel 的编程平台。二者是「模型 + 工具」关系。



### (注 12) 严格定义的回报与实现中的 `return`
在本文的严格数学记号中，**回报**（return）统一记为$`G_t`$：

$`G_t=\sum_{i=t}^{T}\gamma^{i-t}R_{i+1}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{T-t}R_T`$.

这里的$`R_{t+1}`$是在时间步$`t`$执行动作后从环境得到的**奖励随机变量（或其一次观测值）**，而$`G_t`$是从$`t`$开始的折扣累计奖励，因此在时间$`t`$尚未知道后续轨迹时，$`G_t`$也是随机变量。严格区分二者很重要：$`R`$表示一步奖励，$`G`$表示从当前时刻起的累计回报；$`Q^\pi`$和$`V^\pi`$预测的是$`G_t`$的条件期望。

但在多数强化学习代码中，变量名 `return`、`returns`（有时还包括 `value target`）通常表示根据一条已采样轨迹和价值函数估计计算出的**回报目标**，例如 PPO 中的$`\hat G_t=\hat A_t+v(s_t)`$。它是用于价值网络回归的确定数值/张量，并不等同于严格定义的随机变量$`G_t`$。也就是说，代码中的 `return` 是一种历史命名：它近似或充当理论回报的训练目标；正是“return”这个实现名称与严格数学概念同时存在，才会让同一个词看起来有两种含义。


# 参考
## 书籍
1、《动手学强化学习》 张伟楠 等，[https://hrl.boyuai.com/chapter/1/](https://hrl.boyuai.com/chapter/1/%E5%88%9D%E6%8E%A2%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)，配套代码：[https://github.com/boyu-ai/Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL) 推荐指数：🌟🌟🌟🌟🌟

2、《深度强化学习》王树森 等，[https://github.com/wangshusen/DRL](https://github.com/wangshusen/DRL)， 推荐指数：🌟🌟🌟🌟

3、《Easy RL：强化学习教程》王琦等，[https://github.com/datawhalechina/easy-rl/releases](https://github.com/datawhalechina/easy-rl/releases)，Joy RL[https://github.com/datawhalechina/joyrl-book?tab=readme-ov-file](https://github.com/datawhalechina/joyrl-book?tab=readme-ov-file)		推荐指数：🌟🌟🌟🌟

4、《深度强化学习：基础、研究与应用》 董豪 等，推荐指数：🌟🌟🌟🌟

3、《Reinforcement Learning：An Introduction》Sutton 等，推荐指数：🌟🌟

4、《机器学习》周志华 等，推荐指数：🌟🌟

## 课程
1、OpenAI 官方教程，[https://spinningup.openai.com/en/latest/](https://spinningup.openai.com/en/latest/)，			          推荐指数：🌟🌟🌟🌟🌟

2、深度强化学习，王树森，[https://www.bilibili.com/video/BV12o4y197US/](https://www.bilibili.com/video/BV12o4y197US/?spm_id_from=333.337.search-card.all.click&vd_source=7527ef988b583a6735932bbcc17c9589)，		   推荐指数：🌟🌟🌟🌟

3、强化学习的数学原理，赵世钰，[https://www.bilibili.com/video/BV1sd4y167NS/](https://www.bilibili.com/video/BV1sd4y167NS/?spm_id_from=333.337.search-card.all.click&vd_source=7527ef988b583a6735932bbcc17c9589)，推荐指数：🌟🌟🌟

## RL 库
1、rsl-rl，[https://github.com/leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)

2、stable-baselines3，[https://stable-baselines3.readthedocs.io/en/master/](https://stable-baselines3.readthedocs.io/en/master/)

3、天授，[https://github.com/thu-ml/tianshou](https://github.com/thu-ml/tianshou)

## 论文
### Actor-Critic
1、<font style="color:rgb(34, 34, 34);">Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." </font>_<font style="color:rgb(34, 34, 34);">Machine learning</font>_<font style="color:rgb(34, 34, 34);"> 8.3 (1992): 229-256.</font>

### 时间差分
1、<font style="color:rgb(34, 34, 34);">Sutton, Richard S. "Learning to predict by the methods of temporal differences." </font>_<font style="color:rgb(34, 34, 34);">Machine learning</font>_<font style="color:rgb(34, 34, 34);"> 3.1 (1988): 9-44.</font>

### 策略梯度定理
1、<font style="color:rgb(34, 34, 34);">Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." </font>_<font style="color:rgb(34, 34, 34);">Machine learning</font>_<font style="color:rgb(34, 34, 34);"> 8.3 (1992): 229-256.</font>

### TRPO，PPO 系列
1、Schulman, John, et al. "Trust region policy optimization." International conference on machine learning. PMLR, 2015.

2、Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).

3、Schulman, John, et al. "Proximal policy optimization algorithms." _arXiv preprint arXiv:1707.06347_ (2017).

