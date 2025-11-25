"""
将训练好的策略部署到实机
从 best_policy 提取所需文件并创建一个轻量级的部署包
"""

import os
import shutil
import json
from pathlib import Path
import pickle

def extract_policies_for_deployment(
    source_checkpoint: str = "/home/chenshuo/PycharmProjects/move_sim/best_policy",
    deploy_dir: str = "./deployed_policy"
) -> None:
    """
    从完整的 checkpoint 中提取部署所需的最小文件集
    
    Args:
        source_checkpoint: 训练好的 best_policy 路径
        deploy_dir: 部署文件的输出目录
    """
    
    source_path = Path(source_checkpoint)
    deploy_path = Path(deploy_dir)
    
    print(f"[INFO] 从 {source_checkpoint} 提取部署文件...")
    print(f"[INFO] 部署目录: {deploy_dir}")
    
    # 1. 创建部署目录结构
    print("\n[STEP 1] 创建部署目录结构...")
    (deploy_path / "learner_group" / "learner" / "rl_module" / "exo_policy").mkdir(
        parents=True, exist_ok=True
    )
    (deploy_path / "learner_group" / "learner" / "rl_module" / "human_policy").mkdir(
        parents=True, exist_ok=True
    )
    
    # 2. 复制核心文件：exo_policy 权重
    print("\n[STEP 2] 复制 exo_policy 文件...")
    exo_source = (
        source_path / "learner_group" / "learner" / "rl_module" / "exo_policy"
    )
    exo_deploy = deploy_path / "learner_group" / "learner" / "rl_module" / "exo_policy"
    
    required_exo_files = [
        "module_state.pkl",
        "class_and_ctor_args.pkl",
        "metadata.json"
    ]
    
    for file in required_exo_files:
        src_file = exo_source / file
        if src_file.exists():
            shutil.copy2(src_file, exo_deploy / file)
            print(f"  ✓ 复制: {file}")
        else:
            print(f"  ⚠ 文件不存在: {file}")
    
    # 3. 复制核心文件：human_policy 权重
    print("\n[STEP 3] 复制 human_policy 文件...")
    human_source = (
        source_path / "learner_group" / "learner" / "rl_module" / "human_policy"
    )
    human_deploy = deploy_path / "learner_group" / "learner" / "rl_module" / "human_policy"
    
    required_human_files = [
        "module_state.pkl",
        "class_and_ctor_args.pkl",
        "metadata.json"
    ]
    
    for file in required_human_files:
        src_file = human_source / file
        if src_file.exists():
            shutil.copy2(src_file, human_deploy / file)
            print(f"  ✓ 复制: {file}")
        else:
            print(f"  ⚠ 文件不存在: {file}")
    
    # 4. 复制元数据文件
    print("\n[STEP 4] 复制元数据文件...")
    metadata_files = [
        ("rllib_checkpoint.json", source_path / "rllib_checkpoint.json"),
        ("learner_group/metadata.json", source_path / "learner_group" / "metadata.json"),
        ("learner_group/learner/metadata.json", source_path / "learner_group" / "learner" / "metadata.json"),
        ("learner_group/learner/rl_module/metadata.json", source_path / "learner_group" / "learner" / "rl_module" / "metadata.json"),
    ]
    
    for rel_path, src_file in metadata_files:
        deploy_file = deploy_path / rel_path
        deploy_file.parent.mkdir(parents=True, exist_ok=True)
        if src_file.exists():
            shutil.copy2(src_file, deploy_file)
            print(f"  ✓ 复制: {rel_path}")
    
    # 5. 创建部署配置文件
    print("\n[STEP 5] 创建部署配置文件...")
    deploy_config = {
        "deployment_info": {
            "source": str(source_checkpoint),
            "created_at": str(Path(source_checkpoint).stat().st_mtime),
            "policies": ["human_policy", "exo_policy"],
            "deployment_type": "inference_only"
        },
        "required_files": {
            "exo_policy": [
                "learner_group/learner/rl_module/exo_policy/module_state.pkl",
                "learner_group/learner/rl_module/exo_policy/class_and_ctor_args.pkl"
            ],
            "human_policy": [
                "learner_group/learner/rl_module/human_policy/module_state.pkl",
                "learner_group/learner/rl_module/human_policy/class_and_ctor_args.pkl"
            ]
        },
        "optional_files": {
            "metadata": [
                "rllib_checkpoint.json",
                "learner_group/metadata.json"
            ]
        }
    }
    
    config_file = deploy_path / "deployment_config.json"
    with open(config_file, "w") as f:
        json.dump(deploy_config, f, indent=2)
    print(f"  ✓ 创建: deployment_config.json")
    
    # 6. 输出部署信息
    print("\n" + "="*70)
    print("[SUCCESS] 部署文件准备完成！")
    print("="*70)
    
    # 计算文件大小
    total_size = sum(
        f.stat().st_size 
        for f in deploy_path.rglob("*") 
        if f.is_file()
    )
    
    print(f"\n部署位置: {deploy_path}")
    print(f"总文件大小: {total_size / 1024 / 1024:.2f} MB")
    print(f"\n部署文件结构:")
    
    for root, dirs, files in os.walk(deploy_path):
        level = root.replace(str(deploy_path), "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            file_path = Path(root) / file
            file_size = file_path.stat().st_size / 1024
            print(f"{subindent}📄 {file} ({file_size:.2f} KB)")
    
    print("\n" + "="*70)


def verify_deployment(deploy_dir: str = "./deployed_policy") -> bool:
    """
    验证部署文件的完整性
    
    Args:
        deploy_dir: 部署目录
        
    Returns:
        True 如果部署文件完整，False 否则
    """
    
    print(f"\n[INFO] 验证部署文件完整性...")
    deploy_path = Path(deploy_dir)
    
    required_files = [
        "rllib_checkpoint.json",
        "learner_group/learner/rl_module/exo_policy/module_state.pkl",
        "learner_group/learner/rl_module/exo_policy/class_and_ctor_args.pkl",
        "learner_group/learner/rl_module/human_policy/module_state.pkl",
        "learner_group/learner/rl_module/human_policy/class_and_ctor_args.pkl",
    ]
    
    all_ok = True
    for file_path in required_files:
        full_path = deploy_path / file_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"  ✓ {file_path} ({size_kb:.2f} KB)")
        else:
            print(f"  ✗ {file_path} (缺失)")
            all_ok = False
    
    if all_ok:
        print("\n[SUCCESS] ✅ 所有必要文件都已准备好！")
    else:
        print("\n[ERROR] ❌ 部分文件缺失，请检查！")
    
    return all_ok


if __name__ == "__main__":
    # 提取部署文件
    extract_policies_for_deployment(
        source_checkpoint="/home/chenshuo/PycharmProjects/move_sim/best_policy",
        deploy_dir="/home/chenshuo/PycharmProjects/move_sim/deployed_policy"
    )
    
    # 验证部署
    verify_deployment("./deployed_policy")