*本markdown主要记录作业/项目中常犯的错误*.

### `mc_prediction_first_visit`

```python
######## TODO ########
    counts = defaultdict(int)
    while True: # 应该使用num_episodes限定采样次数
      state, _ = env.reset()
      terminated = False
      episode = []
      while not terminated:
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append([state, action, reward, terminated]) 
        # episode的第四个位置应该是next_state
        state = next_state
      for s in env.observation_space:
        # 可以使用set()来优化"首次出现"这个概念
        for i, step in enumerate(episode):
          if step[0] == s:
            G = compute_episode_return(episode[i:], gamma = 1.0)
            counts[s] += 1
            V[s] = (V[s] * (counts[s] - 1) + G)/counts[s]
            break
    ######################
```

Enhanced snippet: `episode`的第四个`next_state`是环境动力学最关键的一步.

```python
######## TODO ########
    counts = defaultdict(int)
    for i in range(num_episodes):
      state, _ = env.reset()
      terminated = False
      episode = []
      while not terminated:
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append([state, action, reward, next_state])
        state = next_state
      visited = set()
      for i, step in enumerate(episode):
        s = step[0]
        if s not in visited:
          visited.add(s)
          G = compute_episode_return(episode[i:], gamma = discount_factor)
          counts[s] += 1
          V[s] = (V[s] * (counts[s] - 1) + G)/counts[s]
          break
    ######################
```

关注均值更新公式
$$
V(s)\leftarrow \frac{iV(s)+G_{i+1}}{i+1}
$$

### `epsilon_greedy_policy_fn`

```python
def epsilon_greedy_policy_fn(state, Q, eps, env):
    if np.random.rand() < eps:
        return env.action_space.sample()
    return np.argmax(Q[state]) # 返回array中最大值对应的index
```

$$
\pi=\begin{cases}\epsilon/m+1-\epsilon,\ a=\arg\max Q(s,a)\\ \epsilon/m\end{cases}
$$

这里把逻辑等效变换为有$\epsilon/m$概率随机选择, $1-\epsilon$概率选择当前最优.

### 状态更新

```python
state = next_state
```

对于`off-policy TD control (QLearning)`,不需要调用`rollout`来获取一整段`episode`,(事实上`policy`不参与`Q`表的更新,因此不像`V`表那样依赖于某个特定`policy`产生的`episode`,而是作为`behavior policy`承担exploration的功能)
$$
Q(s,a)\leftarrow Q(s,a)+\alpha(r+\gamma\max Q(s',a')-Q(s,a))
$$
对于SARSA(`on-policy TD control`),公式看起来相似
$$
Q(s,a)\leftarrow Q(s,a)+\alpha(r+\gamma Q(s',\pi(s'))-Q(s,a))
$$
但是注意代码逻辑

```python
action = policy(state)
next_state, reward, terminated, truncated, info = env.step(action)
next_action = policy(next_state)
```

`Q`表的更新completely controlled于`policy`. 这似乎导致其无法找到事实上的最优路径, 而且不是很稳定,有时会陷入局部死循环 :thinking:

### 关于*终止状态*的学习

```python
	state = next_state # 应该在终止之后
    if terminated:
          break
     Q[state][action] = Q[state][action] + alpha * (reward + discount_factor * np.max(Q[next_state]) - Q[state][action]) # 应该在终止之前
```

核心逻辑:

1. `if terminated`, 没有更新状态的必要. `state = next_state`放后面.
2. `if terminated`,仍然需要更新*当前*的`Q(s,a)`. `env.step(action)`返回的是*下一个*状态是否为终止状态,但无论如何当前状态总是要更新的.
3. `state`必须在`Q(s,a)`更新后更新.
4. ![img](../CG lab/img.png)

### 关于*起始状态*的学习

对于$Monte\ Carlo$采样,起始状态本身就算不采取任何$step$,都算是一次$visit$.