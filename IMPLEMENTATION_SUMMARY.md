# MARL 训练数据记录系统 - 实现总结

## ✅ 已完成的工作

### 1. 核心日志记录功能 (`MARL_training.py`)

添加了两个新函数来自动记录训练数据：

#### `_init_metrics_csv(csv_file)`
- **作用**: 初始化 CSV 文件，创建表头
- **调用时机**: 训练开始前
- **记录的字段**:
  ```
  iteration, exo_reward, human_reward, episode_return_mean, 
  episode_len_mean, policy_loss_human, policy_loss_exo,
  vf_loss_human, vf_loss_exo, entropy_human, entropy_exo, 
  num_episodes
  ```

#### `_log_training_metrics(result, csv_file, iteration)`
- **作用**: 提取每个 iteration 的训练指标，记录到 CSV
- **调用时机**: 每个 `algo.train()` 之后
- **自动提取的指标**:
  - 外骨骼和人类的平均奖励
  - 回合统计（返回值、长度、数量）
  - 优化器指标（策略损失、价值函数损失、熵）

#### 改进的 `main()` 函数
- 创建 `final_training_logs/` 目录
- 保存最优超参数到 `optimal_hyperparams.txt`
- 初始化并更新 `training_metrics.csv`
- 添加进度提示输出

### 2. 数据分析工具 (`analyze_training_logs.py`)

完整的分析和可视化脚本，包含：

#### 函数列表
- `load_logs()`: 加载 CSV 和超参数文件
- `plot_rewards()`: 绘制奖励曲线
- `plot_losses()`: 绘制 4 个子图（策略损失、价值函数损失、熵、回合统计）
- `plot_performance_summary()`: 绘制综合性能总结
- `generate_report()`: 生成完整分析报告

#### 生成的输出文件
```
final_training_logs/
├── 01_rewards.png          # 奖励变化曲线
├── 02_losses.png           # 4 个优化指标子图
├── 03_summary.png          # 性能总结 (4 个面板)
└── training_report.txt     # 详细统计文本报告
```

### 3. 文档

#### `TRAINING_LOG_GUIDE.md` (详细指南)
- 📊 记录的内容详细说明
- 📁 输出目录结构
- 🔄 工作流程说明
- 📈 数据分析示例代码
- 🔧 配置调整指南
- ⚙️ 故障排除

#### `QUICK_REFERENCE.md` (快速参考)
- 🚀 快速开始步骤
- 📊 关键指标说明
- 📈 数据读取和分析方法
- 🔍 常见问题解答
- 📋 文件清单
- 🎯 典型工作流

### 4. 验证工具 (`verify_logging_system.py`)

完整的系统验证脚本，检查：
- ✓ MARL_training.py 配置
- ✓ 分析脚本完整性
- ✓ 文档存在性
- ✓ 依赖模块可用性
- ✓ 生成样本测试数据

---

## 📊 数据记录流程

```
┌─────────────────────────────────────────────────────────┐
│                  MARL Training Pipeline                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │      网格搜索 (param_search)         │
         │  - 运行所有超参数组合               │
         │  - 每个组合 7 iteration             │
         │  - 选择最优组合                     │
         │                                     │
         │  输出: ray_results/*/progress.csv   │
         └─────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │      加载最优超参数 (main)           │
         │  - 获取最优配置                     │
         │  - 创建 final_training_logs/        │
         │  - 保存最优超参数到 txt             │
         │  - 初始化 training_metrics.csv      │
         └─────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │      训练循环 (max_iters = 6)       │
         │  - 每次迭代:                        │
         │    * algo.train()                   │
         │    * 提取结果指标                   │
         │    * 写入 training_metrics.csv      │
         │    * 保存最优模型/检查点            │
         │                                     │
         │  输出: 更新 training_metrics.csv    │
         │  输出: best_policy/, checkpoints    │
         └─────────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │      分析阶段 (analyze_logs.py)     │
         │  - 读取 CSV 和超参数文件            │
         │  - 生成多个可视化图表               │
         │  - 统计分析并生成报告               │
         │                                     │
         │  输出: PNG 图表和 txt 报告          │
         └─────────────────────────────────────┘
```

---

## 🎯 关键特性

### 1. **自动化记录**
- 无需手动干预
- 每个 iteration 自动保存数据
- 防止数据丢失

### 2. **结构化存储**
- 使用标准 CSV 格式
- 易于导入 Excel, Python, R 等工具
- 便于长期存储和分析

### 3. **完整信息**
- 从超参数到性能指标
- 从网格搜索到最优模型
- 从 iteration 级到 episode 级数据

### 4. **易于分析**
- 提供现成的分析脚本
- 自动生成可视化
- 支持自定义扩展

### 5. **追踪能力**
- 记录训练过程中的所有关键指标
- 便于对比不同运行
- 支持长期性能趋势分析

---

## 📂 文件树

```
/home/chenshuo/PycharmProjects/move_sim/
├── SMPL/src/
│   ├── MARL_training.py              ✅ 已增强
│   │   ├── main()                    - 添加日志初始化和记录
│   │   ├── _init_metrics_csv()       - 新增
│   │   └── _log_training_metrics()   - 新增
│   │
│   └── analyze_training_logs.py      ✅ 新增
│       ├── load_logs()
│       ├── plot_rewards()
│       ├── plot_losses()
│       ├── plot_performance_summary()
│       └── generate_report()
│
├── TRAINING_LOG_GUIDE.md             ✅ 新增 (详细指南)
├── QUICK_REFERENCE.md                ✅ 新增 (快速参考)
├── verify_logging_system.py          ✅ 新增 (验证脚本)
│
├── final_training_logs/              (运行后生成)
│   ├── optimal_hyperparams.txt
│   ├── training_metrics.csv
│   ├── 01_rewards.png
│   ├── 02_losses.png
│   ├── 03_summary.png
│   └── training_report.txt
│
└── ray_results/                      (网格搜索结果)
    └── marl_exo_reward_grid/
        └── PPO_MARL_EXO_ENV_*/
            └── progress.csv
```

---

## 🚀 使用步骤

### 第一步: 验证系统
```bash
cd /home/chenshuo/PycharmProjects/move_sim
python verify_logging_system.py
```

### 第二步: 运行训练
```bash
python -m SMPL.src.MARL_training
```

输出信息示例：
```
Iteration:                     0
Episode Metrics:
  Num Episodes (this iter):    12
  Episode Return Mean:         -258.5
  ...
  exo_policy           : -150.32
  human_policy         : -120.45
...
[BEST] Updated at iter 0, reward=-150.32
[SAVE] Saved checkpoint
...
Training iteration:  6
[BEST] save_path/absolute/path/best_policy
[LOGS] Training metrics saved to /path/to/final_training_logs/training_metrics.csv
```

### 第三步: 生成分析报告
```bash
cd /home/chenshuo/PycharmProjects/move_sim
python SMPL/src/analyze_training_logs.py
```

生成的文件：
```
✓ Analysis script found
✓ Exporting rewards curve...
✓ Exporting loss curves...
✓ Exporting performance summary...
✓ Generating statistical report...

Report generated successfully!
Output directory: /absolute/path/final_training_logs
Generated files:
  - 01_rewards.png
  - 02_losses.png
  - 03_summary.png
  - training_report.txt
```

### 第四步: 查看结果
```bash
# 查看文本报告
cat final_training_logs/training_report.txt

# 查看超参数
cat final_training_logs/optimal_hyperparams.txt

# 查看 CSV 数据
head final_training_logs/training_metrics.csv

# 打开图表 (Linux)
display final_training_logs/03_summary.png
```

---

## 📝 数据示例

### optimal_hyperparams.txt
```
=== 最优超参数组合 ===
w_pos_err: 0.4
w_proprio_err: 0.4
w_activation: 0.4
w_exo_energy: 0.2
w_exo_smooth: 0.2
theta_pos_err: 0.5
theta_proprio_err: 0.3
theta_activation: 0.1
theta_exo_energy: 0.1
theta_exo_smooth: 4
```

### training_metrics.csv (示例)
```
iteration,exo_reward,human_reward,episode_return_mean,...
0,-150.32,-120.45,-270.77,...
1,-145.20,-118.60,-263.80,...
2,-142.15,-115.30,-257.45,...
3,-140.50,-112.75,-253.25,...
4,-138.80,-110.20,-249.00,...
5,-137.25,-108.50,-245.75,...
```

### training_report.txt (部分内容)
```
============================================================
TRAINING ANALYSIS REPORT
============================================================

OPTIMAL HYPERPARAMETERS:
------------------------------------------------------------
w_pos_err               : 0.4
...

TRAINING STATISTICS:
------------------------------------------------------------

Initial (Iteration 0):
  Exo Reward:           -150.32
  Human Reward:         -120.45

Final (Last Iteration):
  Exo Reward:           -137.25
  Human Reward:         -108.50

Improvement:
  Exo:               13.07 (8.7%)
  Human:             11.95 (9.9%)
```

---

## 🔧 扩展和定制

### 添加新的记录指标
```python
# 1. 在 _init_metrics_csv() 中添加列
writer.writerow([
    "iteration",
    "exo_reward",
    "human_reward",
    # 新增字段
    "exploration_ratio",
    "avg_episode_timesteps",
])

# 2. 在 _log_training_metrics() 中提取值
exploration_ratio = result.get("some_path", {}).get("metric")
avg_timesteps = metrics.get("env_runners", {}).get("avg_episode_len")

# 3. 写入数据
writer.writerow([
    iteration,
    exo_reward,
    human_reward,
    exploration_ratio,
    avg_timesteps,
])
```

### 改变日志目录
```python
# 在 main() 中修改
log_dir = "/your/custom/path/logs"
abs_log_dir = os.path.abspath(log_dir)
```

### 修改分析图表样式
```python
# 在 analyze_training_logs.py 中修改
plt.rcParams['figure.figsize'] = (16, 10)  # 图表大小
plt.rcParams['font.size'] = 14              # 字体大小
```

---

## ⚠️ 注意事项

1. **硬盘空间**: 检查点文件会占用空间，确保有足够空间
2. **计算资源**: 训练和分析都需要时间，建议使用 GPU
3. **依赖包**: 分析脚本需要 `pandas` 和 `matplotlib`（安装方式见下）
4. **覆盖问题**: 新训练会覆盖旧的日志文件，需要时手动备份

### 安装必要的包
```bash
pip install pandas matplotlib
```

---

## 📞 故障排除

### Q: 没有生成 training_metrics.csv
**A**: 
- 检查 `algo.train()` 是否正常运行
- 确认 `final_training_logs/` 目录可写
- 查看控制台是否有错误信息

### Q: CSV 文件中有 NaN
**A**: 
- 某个 iteration 可能没有产生相应指标
- 检查 `num_episodes` 是否为 0
- 增加 `num_env_runners` 或 `num_envs_per_env_runner`

### Q: 分析脚本报错
**A**: 
- 确保已安装 pandas 和 matplotlib
- 检查 CSV 文件是否有效
- 尝试手动读取 CSV: `pd.read_csv('final_training_logs/training_metrics.csv')`

### Q: 如何比对多次训练？
**A**: 
```python
import pandas as pd
import glob

# 收集所有运行的数据
runs = []
for log_dir in glob.glob('*/final_training_logs/training_metrics.csv'):
    df = pd.read_csv(log_dir)
    df['run'] = log_dir.split('/')[0]
    runs.append(df)

combined = pd.concat(runs)
print(combined.groupby('run')['exo_reward'].describe())
```

---

## 📊 典型输出示例

```
╔══════════════════════════════════════════════════════════╗
║         MARL Training Log System - Ready to Use          ║
╚══════════════════════════════════════════════════════════╝

✓ All checks passed! Training log system is ready.

Next steps:
1. Run: python -m SMPL.src.MARL_training
2. Wait for training to complete
3. Run: python SMPL/src/analyze_training_logs.py
4. View results in: final_training_logs/

Generated files structure:
  final_training_logs/
  ├── optimal_hyperparams.txt       ✓ 最优超参数
  ├── training_metrics.csv          ✓ 每 iteration 的指标
  ├── 01_rewards.png                ✓ 奖励曲线
  ├── 02_losses.png                 ✓ 优化指标
  ├── 03_summary.png                ✓ 性能总结
  └── training_report.txt           ✓ 统计报告
```

---

## 🎓 学习资源

- `TRAINING_LOG_GUIDE.md` - 详细的理论和实践指南
- `QUICK_REFERENCE.md` - 快速查找和常见问题
- `analyze_training_logs.py` - 可视化和分析的示例代码
- `verify_logging_system.py` - 系统验证和测试数据

---

**实现完成于**: 2025年11月20日  
**系统状态**: ✅ 就绪  
**版本**: 1.0  
**兼容版本**: Ray 2.51.1, RLlib 新 API 栈
