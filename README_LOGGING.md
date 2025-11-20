# MARL 训练数据记录系统

## 快速开始

### 1️⃣ 验证系统
```bash
python verify_logging_system.py
```

### 2️⃣ 运行训练
```bash
python -m SMPL.src.MARL_training
```

### 3️⃣ 生成分析报告
```bash
python SMPL/src/analyze_training_logs.py
```

### 4️⃣ 查看结果
```bash
ls -la final_training_logs/
cat final_training_logs/training_report.txt
display final_training_logs/03_summary.png  # Linux
```

---

## 📁 生成的文件

| 文件 | 描述 |
|-----|------|
| `optimal_hyperparams.txt` | 网格搜索找到的最优超参数 |
| `training_metrics.csv` | 每个 iteration 的详细训练指标 |
| `01_rewards.png` | 策略奖励变化曲线 |
| `02_losses.png` | 优化损失指标 |
| `03_summary.png` | 性能综合总结 |
| `training_report.txt` | 统计分析报告 |

---

## 📊 记录的指标

### 性能指标
- **exo_reward**: 外骨骼策略回报
- **human_reward**: 人类策略回报
- **episode_return_mean**: 平均回合回报
- **episode_len_mean**: 平均回合长度

### 优化指标
- **policy_loss**: 策略梯度损失
- **vf_loss**: 价值函数预测误差
- **entropy**: 策略随机性

---

## 📖 详细文档

| 文件 | 内容 |
|-----|------|
| `IMPLEMENTATION_SUMMARY.md` | 📋 实现总结 |
| `TRAINING_LOG_GUIDE.md` | 📚 详细指南 |
| `QUICK_REFERENCE.md` | ⚡ 快速参考 |

---

## 🔍 核心改进

### MARL_training.py
- ✅ 自动创建日志目录
- ✅ 保存最优超参数
- ✅ 记录每个 iteration 的训练指标
- ✅ 每个 episode 的奖励细节（已存在的功能）

### 新增脚本
- ✅ `analyze_training_logs.py` - 完整的数据分析和可视化
- ✅ `verify_logging_system.py` - 系统验证工具

### 新增文档
- ✅ 详细的使用指南
- ✅ 快速参考手册
- ✅ 故障排除指南

---

## 💡 使用示例

### 读取和分析数据
```python
import pandas as pd

# 读取训练指标
df = pd.read_csv('final_training_logs/training_metrics.csv')

# 查看前几行
print(df.head())

# 计算改进
initial_reward = df['exo_reward'].iloc[0]
final_reward = df['exo_reward'].iloc[-1]
improvement = final_reward - initial_reward
print(f"Improvement: {improvement:.2f}")

# 绘制
import matplotlib.pyplot as plt
plt.plot(df['iteration'], df['exo_reward'], label='Exo')
plt.plot(df['iteration'], df['human_reward'], label='Human')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
```

### 对比不同超参数
```bash
# 查看网格搜索结果
head -1 ray_results/marl_exo_reward_grid/PPO_MARL_EXO_ENV_*/progress.csv | 
  grep "exo_policy"

# 查看最优超参数
cat final_training_logs/optimal_hyperparams.txt
```

---

## 🛠️ 自定义

### 添加新的记录指标
编辑 `MARL_training.py` 的 `_init_metrics_csv()` 和 `_log_training_metrics()` 函数。

### 修改日志位置
```python
log_dir = "/your/custom/path"  # 在 main() 中修改
```

### 调整训练参数
```python
max_iters = 10  # 增加训练迭代次数
save_interval = 50  # 改变检查点保存间隔
```

---

## ⚙️ 系统要求

### 必需
- Python 3.8+
- Ray 2.51.1+
- PyTorch

### 可选（用于分析）
- pandas (用于数据处理)
- matplotlib (用于可视化)

### 安装可选依赖
```bash
pip install pandas matplotlib
```

---

## 📞 帮助

### 查看完整文档
```bash
cat TRAINING_LOG_GUIDE.md          # 详细指南
cat QUICK_REFERENCE.md             # 快速参考
cat IMPLEMENTATION_SUMMARY.md      # 实现细节
```

### 验证系统
```bash
python verify_logging_system.py
```

### 常见问题
- 参考 `QUICK_REFERENCE.md` 中的 FAQ 部分
- 参考 `TRAINING_LOG_GUIDE.md` 中的故障排除部分

---

## 📝 文件清单

```
/home/chenshuo/PycharmProjects/move_sim/
├── SMPL/src/
│   ├── MARL_training.py ✅ 已改进
│   └── analyze_training_logs.py ✅ 新增
│
├── IMPLEMENTATION_SUMMARY.md ✅ 新增
├── TRAINING_LOG_GUIDE.md ✅ 新增
├── QUICK_REFERENCE.md ✅ 新增
├── verify_logging_system.py ✅ 新增
└── README_LOGGING.md (本文件)

(运行后生成)
└── final_training_logs/
    ├── optimal_hyperparams.txt
    ├── training_metrics.csv
    ├── 01_rewards.png
    ├── 02_losses.png
    ├── 03_summary.png
    └── training_report.txt
```

---

## ✅ 已验证

- ✓ 核心日志记录功能
- ✓ 分析和可视化脚本
- ✓ 完整的文档
- ✓ 系统验证工具
- ✓ 示例数据生成

---

**状态**: 🟢 就绪  
**版本**: 1.0  
**最后更新**: 2025年11月20日
