"""
检查策略中是否实现了观测值标准化
"""

import pickle
from pathlib import Path
import pprint

def inspect_policy_architecture(checkpoint_dir: str = "/home/chenshuo/PycharmProjects/move_sim/best_policy"):
    """
    检查策略的网络架构，看是否包含标准化层
    """
    
    checkpoint_path = Path(checkpoint_dir)
    print(f"{'='*70}")
    print("🔍 检查策略中的标准化实现")
    print(f"{'='*70}\n")
    
    policies = [
        ("exo_policy", "learner_group/learner/rl_module/exo_policy"),
        ("human_policy", "learner_group/learner/rl_module/human_policy"),
    ]
    
    for policy_name, policy_path in policies:
        print(f"\n[{policy_name.upper()}]")
        print(f"{'─'*70}")
        
        # 1. 检查 class_and_ctor_args.pkl
        ctor_file = checkpoint_path / policy_path / "class_and_ctor_args.pkl"
        
        if ctor_file.exists():
            print(f"\n✓ 检查: {policy_path}/class_and_ctor_args.pkl")
            
            try:
                with open(ctor_file, "rb") as f:
                    ctor_data = pickle.load(f)
                
                if isinstance(ctor_data, dict):
                    # 获取类信息
                    policy_class = ctor_data.get("class")
                    print(f"  类: {policy_class}")
                    
                    # 获取构造函数参数
                    if "ctor_args_and_kwargs" in ctor_data:
                        ctor_args = ctor_data["ctor_args_and_kwargs"]
                        print(f"  构造函数参数数量: {len(ctor_args)}")
                        
                        # 检查参数内容
                        for i, arg in enumerate(ctor_args):
                            print(f"\n  参数 {i}:")
                            _print_nested_dict(arg, indent=4)
            
            except Exception as e:
                print(f"  ✗ 读取失败: {e}")
        
        # 2. 检查 module_state.pkl 中的权重名称
        state_file = checkpoint_path / policy_path / "module_state.pkl"
        
        if state_file.exists():
            print(f"\n✓ 检查: {policy_path}/module_state.pkl")
            
            try:
                with open(state_file, "rb") as f:
                    state_data = pickle.load(f)
                
                if isinstance(state_data, dict):
                    print(f"  权重数量: {len(state_data)}")
                    print(f"  权重名称:")
                    
                    # 查找是否有标准化相关的权重
                    normalization_keywords = [
                        "batch_norm", "layer_norm", "group_norm", 
                        "norm", "normalize", "scale", "bias",
                        "mean", "var", "std"
                    ]
                    
                    for key in sorted(state_data.keys()):
                        # 标记标准化相关的权重
                        is_norm = any(kw in key.lower() for kw in normalization_keywords)
                        marker = "✓ [NORM]" if is_norm else "  "
                        print(f"    {marker} {key}")
                    
                    # 统计
                    norm_keys = [k for k in state_data.keys() 
                               if any(kw in k.lower() for kw in normalization_keywords)]
                    
                    if norm_keys:
                        print(f"\n  ⭐ 发现 {len(norm_keys)} 个标准化相关的权重！")
                    else:
                        print(f"\n  ✗ 未发现标准化相关的权重")
            
            except Exception as e:
                print(f"  ✗ 读取失败: {e}")


def inspect_custom_policy_code():
    """
    检查 custom_policy.py 中的代码实现
    """
    
    print(f"\n{'='*70}")
    print("📝 检查 custom_policy.py 中的实现")
    print(f"{'='*70}\n")
    
    policy_file = Path("/home/chenshuo/PycharmProjects/move_sim/SMPL/src/learning/custom_policy.py")
    
    if not policy_file.exists():
        print(f"✗ 文件不存在: {policy_file}")
        return
    
    try:
        with open(policy_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 搜索关键词
        keywords = {
            "标准化实现": ["normalize", "standardiz", "BatchNorm", "LayerNorm", "GroupNorm"],
            "输入处理": ["preprocess", "input", "observation"],
            "统计信息": ["mean", "std", "var", "scale"],
            "滤波器": ["filter", "running_mean", "running_var"],
        }
        
        print("🔍 代码中的关键词搜索:\n")
        
        for category, kw_list in keywords.items():
            found_kws = [kw for kw in kw_list if kw in content]
            
            if found_kws:
                print(f"✓ [{category}] 发现: {found_kws}")
                
                # 显示包含这些关键词的行
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    for kw in found_kws:
                        if kw in line and not line.strip().startswith('#'):
                            print(f"    第 {i} 行: {line.strip()[:80]}")
                            break
            else:
                print(f"✗ [{category}] 未发现")
        
        # 检查类定义
        print(f"\n✓ 类定义检查:")
        
        class_patterns = ["class HumanModel", "class ExoModel", "def forward", "def _forward_train", "def _forward_exploration"]
        for pattern in class_patterns:
            if pattern in content:
                print(f"  ✓ 发现: {pattern}")
    
    except Exception as e:
        print(f"✗ 读取失败: {e}")


def check_environment_normalization():
    """
    检查环境中的标准化实现
    """
    
    print(f"\n{'='*70}")
    print("🌍 检查环境中的标准化")
    print(f"{'='*70}\n")
    
    env_file = Path("/home/chenshuo/PycharmProjects/move_sim/SMPL/src/env/MARL_env.py")
    
    if not env_file.exists():
        print(f"✗ 文件不存在: {env_file}")
        return
    
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 检查关键方法
        methods_to_check = ["reset", "step", "_get_obs", "get_observation"]
        
        print("🔍 环境中的方法检查:\n")
        
        for method in methods_to_check:
            if f"def {method}" in content:
                print(f"✓ 发现方法: {method}()")
                
                # 在方法内搜索标准化代码
                norm_keywords = ["normalize", "standardiz", "mean", "std", "/ ", "- "]
                found_norm = any(kw in content for kw in norm_keywords)
                
                if found_norm:
                    print(f"  ⚠ 可能包含标准化逻辑")
        
        # 搜索全局的标准化代码
        print(f"\n✓ 全局搜索标准化实现:\n")
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # 查找观测值处理
            if ('observation' in line.lower() or 'obs' in line.lower()) and \
               any(kw in line for kw in ['normalize', 'standardiz', '/ ', '- ', '*', 'mean', 'std']):
                print(f"  第 {i} 行: {line.strip()[:100]}")
    
    except Exception as e:
        print(f"✗ 读取失败: {e}")


def _print_nested_dict(obj, indent=0, max_depth=3):
    """
    递归打印嵌套字典
    """
    indent_str = " " * indent
    
    if indent > max_depth * 4:
        return
    
    if isinstance(obj, dict):
        for key, value in list(obj.items())[:10]:  # 只显示前10个
            if isinstance(value, dict):
                print(f"{indent_str}  {key}: dict ({len(value)} 个键)")
                _print_nested_dict(value, indent + 4, max_depth)
            elif isinstance(value, (list, tuple)):
                print(f"{indent_str}  {key}: {type(value).__name__} (长度: {len(value)})")
                if value and isinstance(value[0], dict):
                    _print_nested_dict(value[0], indent + 4, max_depth)
            else:
                value_str = str(value)[:50]
                print(f"{indent_str}  {key}: {type(value).__name__} = {value_str}")
        
        if len(obj) > 10:
            print(f"{indent_str}  ... 还有 {len(obj) - 10} 个键")
    
    elif isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], dict):
            print(f"{indent_str}包含 {len(obj)} 个 dict 元素")
            _print_nested_dict(obj[0], indent + 4, max_depth)


def summary():
    """
    总结和建议
    """
    
    print(f"\n{'='*70}")
    print("📌 检查总结与建议")
    print(f"{'='*70}\n")
    
    print("""
根据检查结果，标准化可能出现在以下位置：

┌─ 情况 A: 标准化在策略中实现
│  ├─ 表现: 权重中包含 BatchNorm、LayerNorm 等层
│  ├─ 影响: 推理时无需额外处理
│  └─ 验证: 查看权重名称是否包含 'norm'
│
├─ 情况 B: 标准化在环境中实现
│  ├─ 表现: MARL_env.py 的 reset()/step() 返回标准化后的观测值
│  ├─ 影响: 推理时需要使用相同的标准化逻辑
│  └─ 验证: 查看环境代码的观测值处理
│
└─ 情况 C: 无标准化处理
   ├─ 表现: 直接使用原始观测值
   ├─ 影响: 推理时无需任何处理
   └─ 验证: 权重和代码中都未发现标准化

🔧 下一步建议：

1️⃣ 如果发现权重包含 'norm':
   └─ ✓ 推理时无需额外处理

2️⃣ 如果环境代码包含标准化:
   └─ 需要提取平均值和标准差
   └─ 在推理时应用相同的变换

3️⃣ 如果都没有发现:
   └─ 直接使用原始观测值进行推理
""")


if __name__ == "__main__":
    # 1. 检查策略架构
    inspect_policy_architecture()
    
    # 2. 检查代码实现
    inspect_custom_policy_code()
    
    # 3. 检查环境
    check_environment_normalization()
    
    # 4. 总结
    summary()