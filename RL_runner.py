import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from RL_env import BEROptimizationEnv
from collections import defaultdict

def lr_schedule(progress_remaining):
    min_lr = 1e-5
    return max(1e-4 * progress_remaining, min_lr)

# Hyperparmeters for training - can be modified
episodes = 100
timesteps = 5000
stepcount_thresh = 1000

# Initialize structures
config_count = defaultdict(int)
episode_rewards = []

# Initialize the environment
env = DummyVecEnv([lambda: BEROptimizationEnv(input_length=1024, snr_values=[0.2 * i for i in range(11)])])

# Load the model if it exists, otherwise create a new model - this is done so that the model 
# can be trained incrementally without starting from scratch

model_file = "ppo_ber_optimization.zip"
if os.path.exists(model_file):
    print("Loading the saved model.")
    model = PPO.load(model_file, env=env)
else:
    print("No saved model found. Creating a new model.")
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=lr_schedule, n_steps=1024, batch_size=32)
    model.learn(total_timesteps=5000)
    model.save(model_file)

# Training Loop - the best configuration , i.e, the value of rows and columns that gives the 
# lowest BER is tracked and it is displayed after each episode. Furthermore, at the end of training
# the best configuration is displayed along with the number of times it was selected as the best
# configuration during training.

for episode in range(episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    step_count = 0

    try:
        while not done:
            action, _ = model.predict(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step_count += 1

            if step_count >= stepcount_thresh:
                done = True

    except Exception as e:
        print(f"Error during episode {episode + 1}: {e}")
    
    best_rows, best_cols, best_ber = env.envs[0].get_best_configuration()
    if best_rows is not None and best_cols is not None:
        config_count[(best_rows, best_cols)] += 1

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward.item():.2f} after {step_count} steps")  # Use .item()
    print(f"Best Configuration: Rows = {best_rows}, Cols = {best_cols}, BER = {best_ber}")

# Display the best configuration found during training
if config_count:
    best_config = max(config_count, key=config_count.get)
    print(f"\nBest Configuration Overall: Rows = {best_config[0]}, Cols = {best_config[1]}")
    print(f"Selected {config_count[best_config]} times as the best configuration.")
else:
    print("No best configuration found during training.")

