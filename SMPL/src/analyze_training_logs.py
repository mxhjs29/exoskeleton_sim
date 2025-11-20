"""
训练日志分析脚本
用于可视化和分析最优超参数训练的结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_logs(log_dir="final_training_logs"):
    """加载训练日志"""
    metrics_file = os.path.join(log_dir, "training_metrics.csv")
    hyperparams_file = os.path.join(log_dir, "optimal_hyperparams.txt")
    
    # 读取指标
    metrics = pd.read_csv(metrics_file)
    
    # 读取最优超参数
    hyperparams = {}
    if os.path.exists(hyperparams_file):
        with open(hyperparams_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # 跳过标题行
                if ':' in line:
                    k, v = line.strip().split(':')
                    try:
                        hyperparams[k] = float(v)
                    except:
                        hyperparams[k] = v
    
    return metrics, hyperparams


def plot_rewards(metrics, save_path=None):
    """绘制奖励曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(metrics['iteration'], metrics['exo_reward'], 
            marker='o', label='Exo Policy', linewidth=2)
    ax.plot(metrics['iteration'], metrics['human_reward'], 
            marker='s', label='Human Policy', linewidth=2)
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title('Policy Returns During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_losses(metrics, save_path=None):
    """绘制损失曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Policy Loss
    axes[0, 0].plot(metrics['iteration'], metrics['policy_loss_human'], 
                    marker='o', label='Human', linewidth=2)
    axes[0, 0].plot(metrics['iteration'], metrics['policy_loss_exo'], 
                    marker='s', label='Exo', linewidth=2)
    axes[0, 0].set_title('Policy Loss', fontweight='bold')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Value Function Loss
    axes[0, 1].plot(metrics['iteration'], metrics['vf_loss_human'], 
                    marker='o', label='Human', linewidth=2)
    axes[0, 1].plot(metrics['iteration'], metrics['vf_loss_exo'], 
                    marker='s', label='Exo', linewidth=2)
    axes[0, 1].set_title('Value Function Loss', fontweight='bold')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 0].plot(metrics['iteration'], metrics['entropy_human'], 
                    marker='o', label='Human', linewidth=2)
    axes[1, 0].plot(metrics['iteration'], metrics['entropy_exo'], 
                    marker='s', label='Exo', linewidth=2)
    axes[1, 0].set_title('Policy Entropy', fontweight='bold')
    axes[1, 0].set_xlabel('Training Iteration')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Episode Details
    axes[1, 1].plot(metrics['iteration'], metrics['episode_len_mean'], 
                    marker='o', linewidth=2, color='green', label='Avg Length')
    ax2 = axes[1, 1].twinx()
    ax2.plot(metrics['iteration'], metrics['num_episodes'], 
             marker='s', linewidth=2, color='orange', label='Num Episodes')
    axes[1, 1].set_title('Episode Statistics', fontweight='bold')
    axes[1, 1].set_xlabel('Training Iteration')
    axes[1, 1].set_ylabel('Episode Length', color='green')
    ax2.set_ylabel('Number of Episodes', color='orange')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_performance_summary(metrics, hyperparams, save_path=None):
    """绘制性能总结"""
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. 奖励对比
    ax1 = fig.add_subplot(gs[0, 0])
    policies = ['Exo', 'Human']
    rewards = [
        metrics['exo_reward'].iloc[-1],
        metrics['human_reward'].iloc[-1]
    ]
    colors = ['#1f77b4', '#ff7f0e']
    ax1.bar(policies, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Final Episode Return')
    ax1.set_title('Final Rewards', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 损失对比
    ax2 = fig.add_subplot(gs[0, 1])
    losses = [
        metrics['policy_loss_human'].iloc[-1],
        metrics['policy_loss_exo'].iloc[-1]
    ]
    ax2.bar(policies, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Final Policy Loss')
    ax2.set_title('Final Policy Losses', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 熵对比
    ax3 = fig.add_subplot(gs[0, 2])
    entropy = [
        metrics['entropy_human'].iloc[-1],
        metrics['entropy_exo'].iloc[-1]
    ]
    ax3.bar(policies, entropy, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Final Policy Entropy')
    ax3.set_title('Final Policy Entropies', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 训练进度
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(metrics['iteration'], metrics['exo_reward'], 
            marker='o', label='Exo Reward', linewidth=2.5, markersize=6)
    ax4.plot(metrics['iteration'], metrics['human_reward'], 
            marker='s', label='Human Reward', linewidth=2.5, markersize=6)
    ax4.fill_between(metrics['iteration'], metrics['exo_reward'], 
                     metrics['human_reward'], alpha=0.2)
    ax4.set_xlabel('Training Iteration')
    ax4.set_ylabel('Episode Return')
    ax4.set_title('Training Progress', fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 添加超参数信息作为文本
    if hyperparams:
        param_text = "Optimal Hyperparameters:\n"
        for k, v in list(hyperparams.items())[:5]:
            param_text += f"{k}: {v}\n"
        if len(hyperparams) > 5:
            param_text += f"... ({len(hyperparams)} total)"
        
        fig.text(0.98, 0.02, param_text, 
                ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def generate_report(log_dir="final_training_logs", output_dir=None):
    """生成完整的分析报告"""
    if output_dir is None:
        output_dir = log_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading training logs...")
    metrics, hyperparams = load_logs(log_dir)
    
    print("Generating visualizations...")
    
    # 生成图表
    plot_rewards(metrics, 
                os.path.join(output_dir, "01_rewards.png"))
    plot_losses(metrics, 
               os.path.join(output_dir, "02_losses.png"))
    plot_performance_summary(metrics, hyperparams,
                            os.path.join(output_dir, "03_summary.png"))
    
    # 生成统计报告
    report_file = os.path.join(output_dir, "training_report.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("OPTIMAL HYPERPARAMETERS:\n")
        f.write("-" * 60 + "\n")
        for k, v in hyperparams.items():
            f.write(f"{k:25s}: {v}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("TRAINING STATISTICS:\n")
        f.write("-" * 60 + "\n")
        
        # Initial vs Final
        f.write("\nInitial (Iteration 0):\n")
        f.write(f"  Exo Reward:        {metrics['exo_reward'].iloc[0]:10.2f}\n")
        f.write(f"  Human Reward:      {metrics['human_reward'].iloc[0]:10.2f}\n")
        
        f.write("\nFinal (Last Iteration):\n")
        f.write(f"  Exo Reward:        {metrics['exo_reward'].iloc[-1]:10.2f}\n")
        f.write(f"  Human Reward:      {metrics['human_reward'].iloc[-1]:10.2f}\n")
        
        # Improvements
        exo_improvement = metrics['exo_reward'].iloc[-1] - metrics['exo_reward'].iloc[0]
        human_improvement = metrics['human_reward'].iloc[-1] - metrics['human_reward'].iloc[0]
        
        f.write("\nImprovement:\n")
        f.write(f"  Exo:               {exo_improvement:10.2f} ({exo_improvement/abs(metrics['exo_reward'].iloc[0])*100:.1f}%)\n")
        f.write(f"  Human:             {human_improvement:10.2f} ({human_improvement/abs(metrics['human_reward'].iloc[0])*100:.1f}%)\n")
        
        # Statistics
        f.write("\n" + "=" * 60 + "\n")
        f.write("DETAILED STATISTICS:\n")
        f.write("-" * 60 + "\n")
        f.write(metrics.describe().to_string())
        
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("BEST/WORST ITERATIONS:\n")
        f.write("-" * 60 + "\n")
        best_exo_iter = metrics['exo_reward'].idxmax()
        best_human_iter = metrics['human_reward'].idxmax()
        f.write(f"Best Exo Reward:   Iteration {best_exo_iter} ({metrics['exo_reward'].iloc[best_exo_iter]:.2f})\n")
        f.write(f"Best Human Reward: Iteration {best_human_iter} ({metrics['human_reward'].iloc[best_human_iter]:.2f})\n")
    
    print("\nReport generated successfully!")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("\nGenerated files:")
    print(f"  - 01_rewards.png")
    print(f"  - 02_losses.png")
    print(f"  - 03_summary.png")
    print(f"  - training_report.txt")


if __name__ == "__main__":
    # 生成默认报告
    generate_report()
    
    # 显示所有图表
    plt.show()
