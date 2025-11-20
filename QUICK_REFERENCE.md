# 快速参考：MARL 训练数据记录

## 🚀 快速开始

### 1. 运行训练
```bash
cd /home/chenshuo/PycharmProjects/move_sim
python -m SMPL.src.MARL_training
```

### 2. 查看训练数据
训练完成后，检查这些文件：
```
final_training_logs/
├── optimal_hyperparams.txt      # 最优超参数
└── training_metrics.csv         # 详细训练指标
```

### 3. 生成分析报告
```bash
python SMPL/src/analyze_training_logs.py
```

生成的文件：
```
final_training_logs/
├── 01_rewards.png               # 奖励曲线
├── 02_losses.png                # 损失曲线
├── 03_summary.png               # 性能总结
└── training_report.txt          # 详细统计报告
```

## 📊 关键指标说明

### 奖励 (Rewards)
- **exo_reward**: 外骨骼策略在每个 iteration 的平均回合回报
- **human_reward**: 人类策略在每个 iteration 的平均回合回报
- 更高更好，通常是负数（惩罚为主）

### 损失 (Losses)
- **policy_loss**: 策略目标函数的损失，越小越好
- **vf_loss**: 价值函数的预测误差，越小越好
- 用于优化策略网络的梯度

### 熵 (Entropy)  
- **entropy**: 策略分布的熵，表示策略的随机性
- 高熵 = 动作分布平均 = 更多探索
- 低熵 = 集中在某些动作 = 更多开发
- 通常需要逐渐下降的熵来达到收敛

### 其他指标
- **episode_return_mean**: 所有智能体的平均回报
- **episode_len_mean**: 平均回合长度（环境步数）
- **num_episodes**: 本 iteration 产生的回合数

## 📈 如何读取和分析数据

### 用 Pandas 读取 CSV
```python
import pandas as pd

# 读取训练指标
df = pd.read_csv('final_training_logs/training_metrics.csv')

# 查看前几行
print(df.head())

# 基本统计
print(df.describe())

# 绘制某个指标
import matplotlib.pyplot as plt
plt.plot(df['iteration'], df['exo_reward'])
plt.xlabel('Iteration')
plt.ylabel('Exo Reward')
plt.show()
```

### 对比不同超参数组合
```python
# 从网格搜索结果中读取数据
import glob
import pandas as pd

# 查找所有网格搜索试验
trials = glob.glob('ray_results/marl_exo_reward_grid/PPO_MARL_EXO_ENV_*/progress.csv')

# 比较最终性能
for trial in trials:
    df = pd.read_csv(trial)
    final_reward = df['env_runners/module_episode_returns_mean/exo_policy'].iloc[-1]
    trial_name = trial.split('/')[-2]
    print(f"{trial_name}: {final_reward:.2f}")
```

## 🔍 常见问题

### Q: 为什么奖励是负数？
**A**: 这很正常。环境中大部分是惩罚项（位置错误、能耗等），奖励接近 0 的负数是好的。

### Q: 如何判断训练是否收敛？
**A**: 
- 奖励曲线变平缓
- 策略损失和 VF 损失逐渐减小
- 熵逐渐降低
- 连续多个 iteration 没有显著改进

### Q: training_metrics.csv 中有 NaN 值？
**A**: 可能是某些 iteration 没有产生相应的指标。检查 `num_episodes` 是否为 0。

### Q: 如何导出数据到 Excel？
```python
import pandas as pd

# 读取 CSV
df = pd.read_csv('final_training_logs/training_metrics.csv')

# 写入 Excel
df.to_excel('final_training_logs/training_metrics.xlsx', index=False)
```

### Q: 如何对比多次训练运行？
```python
import pandas as pd
import os

all_data = []
for run_dir in ['run_1', 'run_2', 'run_3']:
    csv_file = f'{run_dir}/final_training_logs/training_metrics.csv'
    df = pd.read_csv(csv_file)
    df['run'] = run_dir
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
print(combined.groupby('run')['exo_reward'].describe())
```

## 📋 文件清单

| 文件 | 描述 | 何时生成 |
|-----|------|--------|
| `optimal_hyperparams.txt` | 最优超参数 | 运行开始时 |
| `training_metrics.csv` | 每个 iteration 的详细指标 | 每个 iteration 结束时 |
| `reward_terms_by_episode.csv` | 每个 episode 的奖励分量 | 每个 episode 结束时 |
| `01_rewards.png` | 奖励曲线图 | 运行 `analyze_training_logs.py` |
| `02_losses.png` | 损失曲线图 | 运行 `analyze_training_logs.py` |
| `03_summary.png` | 性能总结图 | 运行 `analyze_training_logs.py` |
| `training_report.txt` | 统计报告 | 运行 `analyze_training_logs.py` |

## 🎯 典型工作流

1. **准备阶段**
   ```bash
   # 编辑参数
   vim SMPL/src/MARL_training.py  # 调整 max_iters, save_interval 等
   ```

2. **训练阶段**
   ```bash
   python -m SMPL.src.MARL_training
   # 这将运行网格搜索 + 最优超参数训练
   # 大约需要 10-30 分钟（取决于硬件）
   ```

3. **分析阶段**
   ```bash
   python SMPL/src/analyze_training_logs.py
   # 生成图表和报告
   ```

4. **评估阶段**
   ```bash
   # 查看生成的可视化和报告
   cat final_training_logs/training_report.txt
   open final_training_logs/03_summary.png  # macOS
   # 或
   display final_training_logs/03_summary.png  # Linux
   ```

5. **模型部署**
   ```bash
   # 最好的模型已保存在 best_policy/
   # 最近的检查点在 final_policy_checkpoints/
   ```

## 🔧 自定义日志

### 添加新指标
编辑 `MARL_training.py` 的 `_init_metrics_csv()` 和 `_log_training_metrics()` 函数：

```python
# 在 _init_metrics_csv() 中添加表头
writer.writerow([
    "iteration",
    "exo_reward",
    "human_reward",
    # 添加新列
    "avg_step_time",
    "gpu_memory_used",
    # ...
])

# 在 _log_training_metrics() 中提取新指标
avg_step_time = metrics.get("perf", {}).get("avg_step_time")
gpu_mem = metrics.get("perf", {}).get("gpu_memory_used")

# 写入时包含新值
writer.writerow([
    iteration,
    exo_reward,
    human_reward,
    avg_step_time,
    gpu_mem,
    # ...
])
```

### 更改日志位置
```python
# 在 main() 函数中
log_dir = "/path/to/your/logs"
```

---

**最后更新**: 2025年11月20日  
**相关文件**: 
- `SMPL/src/MARL_training.py` - 主训练脚本
- `SMPL/src/analyze_training_logs.py` - 分析脚本
- `TRAINING_LOG_GUIDE.md` - 详细指南
