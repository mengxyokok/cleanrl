from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: tuple,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    render_env: bool = False,
):
    """
    Evaluate a trained SAC model.

    Args:
        model_path: Path to the saved model file
        make_env: Function to create environment
        env_id: Environment ID
        eval_episodes: Number of episodes to evaluate
        run_name: Name for the evaluation run
        Model: Tuple of (Actor, SoftQNetwork) classes
        device: Device to run evaluation on
        capture_video: Whether to capture videos
        render_env: Whether to render the environment in real-time
    """
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name, render_env)])
    Actor, SoftQNetwork = Model
    
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    
    # Load model weights
    actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf1.load_state_dict(qf1_params)
    qf2.load_state_dict(qf2_params)
    qf1.eval()
    qf2.eval()
    # note: qf1 and qf2 are not used in evaluation, only actor is needed

    obs, _ = envs.reset()
    episodic_returns = []
    episodic_lengths = []
    
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            # Use deterministic action (mean) for evaluation
            _, _, actions = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        next_obs, _, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None:
                    continue
                if "episode" not in info:
                    continue
                episode_return = float(info["episode"]["r"])
                episode_length = int(info["episode"]["l"])
                print(f"eval_episode={len(episodic_returns)}, episodic_return={episode_return:.2f}, episodic_length={episode_length}")
                episodic_returns += [episode_return]
                episodic_lengths += [episode_length]
        obs = next_obs

    # Print summary statistics
    print(f"\nEvaluation Summary:")
    print(f"  Episodes: {eval_episodes}")
    if len(episodic_returns) > 0:
        print(f"  Mean Return: {sum(episodic_returns) / len(episodic_returns):.2f}")
        print(f"  Std Return: {torch.tensor(episodic_returns).std().item():.2f}")
        print(f"  Min Return: {min(episodic_returns):.2f}")
        print(f"  Max Return: {max(episodic_returns):.2f}")
        print(f"  Mean Length: {sum(episodic_lengths) / len(episodic_lengths):.2f}")

    return episodic_returns


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model file")
    parser.add_argument("--env-id", type=str, default="Hopper-v4", help="Environment ID")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--capture-video", action="store_false", help="Whether to capture videos")
    parser.add_argument("--render-env", action="store_true", help="Whether to render the environment")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on (cpu/cuda)")
    args = parser.parse_args()

    # Import SAC classes
    from cleanrl.sac_continuous_action import Actor, SoftQNetwork, make_env

    device = torch.device(args.device)
    run_name = f"eval_{Path(args.model_path).stem}"

    evaluate(
        model_path=args.model_path,
        make_env=make_env,
        env_id=args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=run_name,
        Model=(Actor, SoftQNetwork),
        device=device,
        capture_video=args.capture_video,
        render_env=args.render_env,
    )

