### 杂项

常用依赖安装指令

```bash
pip install -r requirements/requirements.txt
```

创建本地虚拟环境

```bash
uv venv
```

或者直接用`python`启动,独立于shell

```bash
python -m uv run python ppo.py
```

`isinstance`的另一个用法

```python
assert isinstance(envs.single_action_space, gym.spaces.Discrete)
```

`with`逻辑意味着保持某个判断为真

```python
with torch.no_grad():
```



### 训练逻辑`ppo.py`

1. 准备环境张量`tensor`.

   ```python
   rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
   ```

   对应$r(s_t^i,a_t^i)$, 分为时间步维度$T$以及并行环境数量$N$.参考
   $$
   \nabla_{\theta}J(\theta)\rightarrow\frac{1}{N}\sum^N_{n=1}\left(\sum^T_{t=1} \ln\nabla_\theta \pi_\theta(a_t|s_t) \right)G_t
   $$

   > 也就是说, $N$个蒙特卡洛采样可以并行发生. 包括环境信息采样并行和公式计算并行 :smile:.

2. `anneal`退火处理
   $$
   f=1-\frac{i-1}{N}
   $$

   $$
   \alpha\leftarrow f\cdot \alpha
   $$

3. `global_step`的作用仅仅是`Timestamp`吗? :thinking:.​
    可用于表示环境总交互量,和训练数据量直接挂钩, 是固定的训练*成本*上限.

   ```python
   for step in range(0, args.num_steps):
               global_step += args.num_envs
               obs[step] = next_obs
               dones[step] = next_done
   ```

   [异步更新]注意这里是`post update`, 在`step 1`计算出的`step 2`的值`obs`,`done`,要等到`step 2`才能记录. 所以`terminal`的`next_obs`和`next_done`都是"游离状态". 

4. GAE广义优势估计

   ```python
   with torch.no_grad():
       next_value = agent.get_value(next_obs).reshape(1, -1)
       advantages = torch.zreos_like(rewards).to(device)
       lastgaelam = 0
   ```

   ```python
   for t in reversed(range(args.num_steps)):
       if t == args.num_steps - 1: # 受上文post update逻辑影响
           nextnonterminal = 1 - next_done
           nextvalues = next_value
       else:
           nextnonterminal = 1 - dones[t + 1]
           nextvalues = values[t + 1]
       delta = rewards[t] + args.gamma * nextvalues * nextnonterminal -values[t] # TD error
       advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
   ```

   主要依赖TD error (单步估计)
   $$
   \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
   $$
   以及对不同步数的优势估计做加权平均
   $$
   \hat{A}_t^{GAE}=\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+...
   $$
   为了方便计算, 采用*逆势回击*, 更新公式为
   $$
   A_t=\delta_t+\gamma\lambda\cdot lastgaelam
   $$

   $$
   lastgaelam\leftarrow A_t
   $$

   `advantages`是相对优势的度量,注意公式
   $$
   \nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}[\nabla_\theta\pi_\theta(s_t,a_t)\hat{A}_t^{GAE}]
   $$

   $$
   \theta\leftarrow \theta+\alpha\nabla_\theta J(\theta)
   $$

   提升更新效率,追求*比预期更好*.

   > `bias`和`variance`的区别:采样越少,偏差越大,方差越小. 反之亦然.

