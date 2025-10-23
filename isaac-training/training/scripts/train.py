import argparse
import os
import hydra
import datetime
import wandb
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
print("Config path: ", FILE_PATH)

# 使用 @hydra.main 装饰器，这是 hydra 的入口点。
# 它会去 config_path 指定的路径下查找 config_name 指定的配置文件（例如 train.yaml），
# 并将其内容解析为一个 DictConfig 对象，作为参数 cfg 传递给 main 函数。
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # 1. 定义您的本地资产文件夹路径
    local_assets_path = "/home/ljn/isaac_sim_assets/Assets/Isaac/2023.1.1"
    print(f"[Info] 准备将本地资产路径 '{local_assets_path}' 添加到启动配置中。")

    # 2. 创建一个配置字典，包含所有启动选项
    simulation_app_config = {
        # 通用设置
        "headless": cfg.headless,
        "anti_aliasing": 1
    }
    
    # Simulation App
    sim_app = SimulationApp(simulation_app_config)
    print(f"[Ok] Isaac Sim 已启动，并且路径 '/Isaac' 已成功映射到本地文件夹。")

    # while True:
    #     sim_app.update()

    # Use Wandb to monitor training
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # Navigation Training Environment
    from env import NavigationEnv
    env = NavigationEnv(cfg)

    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    
    # 实例化一个LeePositionController,负责将期望的速度/姿态转换为电机推力。
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    # 创建一个VelController,该模块会接收智能体的速度指令（动作），并使用内部的 controller 来控制无人机。
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    # 使用 TransformedEnv 类将原始环境 env 和组合后的转换模块 Compose(*transforms) 包装起来，创建一个新的转换后环境。智能体将与这个 transformed_env 交互。
    # .train() 将环境设置为训练模式。
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)     

    # PPO Policy
    # 初始化 PPO 策略/智能体。
    # 传入算法配置、环境的观测空间 (observation_spec)、动作空间 (action_spec) 和计算设备 (device)。
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    # checkpoint = "/home/xinmingh/RLDrones/navigation/scripts/nav-ros/navigation_runner/ckpts/checkpoint_36000.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # Episode Stats Collector
    # 回合统计收集器 (Episode Stats Collector)
    # 从环境的观测空间定义中，筛选出所有用于统计的键（通常键是一个元组，且第一个元素是 "stats"）。
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    # 实例化 EpisodeStats 类，用于收集和计算每个回合的统计数据（如奖励、步长等）。
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    # 强化学习数据收集器 (RL Data Collector)
    # 初始化 SyncDataCollector，这是一个用于在环境中运行策略并同步收集数据的核心组件。
    collector = SyncDataCollector(
        transformed_env, # 要从中收集数据的环境。
        policy=policy, # 用于与环境交互的策略。
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, # 每个批次收集多少帧数据。
        total_frames=cfg.max_frame_num, # 总共要收集的帧数，决定了训练的长度。
        device=cfg.device, # 数据存放的设备（CPU 或 GPU）。
        return_same_td=True, # 是否原地更新返回的 TensorDict 对象，可以节省内存。
        exploration_type=ExplorationType.RANDOM, # 探索类型，RANDOM 表示从策略输出的分布中随机采样动作。
    )

    # # Calculate total iterations for the progress bar
    # # 为进度条计算总迭代次数。
    # total_iterations = cfg.max_frame_num // (cfg.env.num_envs * cfg.algo.training_frame_num)
    # # 使用 tqdm 包装数据收集器 collector，以在终端中显示训练进度。
    # progress_bar = tqdm(enumerate(collector), total=total_iterations, desc="Training Progress")

    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Info
        # 创建一个 info 字典，用于存储当前训练步骤的一些统计信息，如环境帧数和采样帧率。
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # Train Policy
        # 使用当前收集的数据训练 PPO 策略，并获取训练损失统计信息。
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats) # log training loss info

        # Calculate and log training episode stats
        # 从收集的数据中提取回合统计信息，并将其添加到 episode_stats 收集器中。
        episode_stats.add(data)
        # 每当所有智能体完成一个回合时，计算并记录回合统计信息。
        if len(episode_stats) >= transformed_env.num_envs: # evaluate once if all agents finished one episode
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

            # Find the reward key in stats and update the progress bar
            reward_key = next((k for k in stats.keys() if "reward" in k), None)
            if reward_key:
                progress_bar.set_postfix({"reward": f"{stats[reward_key]:.2f}"})

        # Evaluate policy and log info
        # 每隔 cfg.eval_interval 步评估一次策略。
        if i % cfg.eval_interval == 0:
            print("[NavRL]: start evaluating policy at training step: ", i)
            env.enable_render(True)

            env.eval()
            eval_info = evaluate(
                env=transformed_env, 
                policy=policy,
                seed=cfg.seed, 
                cfg=cfg,
                exploration_type=ExplorationType.MEAN
            ) 
            env.enable_render(not cfg.headless) # It's good practice to turn it off after use
            env.train()
            env.reset()
            info.update(eval_info)
            print("\n[NavRL]: evaluation done.")
        
        # Update wand info
        run.log(info)


        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", i)
            print("Checkpoint path: ", ckpt_path)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    