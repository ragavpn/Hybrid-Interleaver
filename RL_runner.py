import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from RL_env import BEROptimizationEnv

def lr_schedule(progress_remaining):
    min_lr = 1e-5
    return max(1e-4 * progress_remaining, min_lr)

# Define the environment
env = DummyVecEnv([lambda: BEROptimizationEnv(input_length=1024, snr_values=[0.2 * i for i in range(11)], iterations=5)])

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# PPO Model training or loading
model_file = "ppo_ber_optimization.zip"
if os.path.exists(model_file):
    print("Loading the saved model...")
    model = PPO.load(model_file, env=env, device=device)
else:
    print("No saved model found. Creating a new model...")
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=lr_schedule, n_steps=1024, batch_size=32, device=device)
    model.learn(total_timesteps=10000)  # Reduced total timesteps for faster feedback
    model.save(model_file)

# Run the trained model for evaluation
episode_rewards = []  # To store the rewards for each episode

for episode in range(100):  # Reduced number of episodes for faster output
    state = env.reset()
    done = False
    episode_reward = 0
    step_count = 0  # Track the steps in each episode
    
    try:
        while not done:
            action, _ = model.predict(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step_count += 1

            if step_count >= 1000:  # Stop episode after 50 steps
                done = True

    except Exception as e:
        print(f"Error during episode {episode + 1}: {e}")
    
    # Log and save rewards for analysis
    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward} after {step_count} steps")

print("Evaluation completed!")
