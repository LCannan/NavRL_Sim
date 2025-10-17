import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp # 唯一的外部依赖
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
print("Config path: ", FILE_PATH)

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # ======================= [核心修改部分] =======================
    # 定义你的本地资产文件夹路径
    # 建议使用绝对路径以避免混淆
    local_assets_path = "/home/ljn/isaac_sim_assets/Assets/Isaac/4.2"
    print(f"[NavRL] Using local assets from: {local_assets_path}")

    # 创建SimulationApp的配置字典
    sim_config = {
        "headless": cfg.headless,
        "anti_aliasing": 1,
        # 使用carb.settings来配置底层设置
        "carb.settings": {
            # 禁用对NVIDIA公共服务器的默认挂载
            # 这是让他不去网上搜索资源的关键！
            "/persistent/app/omniverse/mounts/NVIDIA": "[]", 
            
            # 添加一个新的挂载点，名为 "MyLocalAssets"
            # 它将指向你的本地文件夹
            "/persistent/app/omniverse/mounts/MyLocalAssets": f'{{ "target": "{local_assets_path}" }}',
            
            # (可选) 禁用一些可能尝试联网的功能
            "/app/enableSharedFabric": False,
            "/ngx/enabled": False,
        }
    }
    
    # 使用上面定义好的配置来初始化SimulationApp
    sim_app = SimulationApp(sim_config)
    # ======================= [修改结束] =======================

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

    # 确保在你的env.py或者相关USD资产路径中使用新的挂载点
    # 例如，如果你的无人机USD文件在 "assets/Drones/my_drone.usd"
    # 那么在代码中引用它时，路径应该写成:
    # "omniverse://MyLocalAssets/Drones/my_drone.usd"
    
    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    
    
    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    # checkpoint = "/home/xinmingh/RLDrones/navigation/scripts/nav-ros/navigation_runner/ckpts/checkpoint_36000.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)
    
    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )
    
    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Info
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        # Train Policy
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats) # log training loss info
        # Calculate and log training episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs: # evaluate once if all agents finished one episode
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
            
        # Evaluate policy and log info
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
            env.enable_render(not cfg.headless)
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
            
    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()
    
if __name__ == "__main__":
    main()