# Bit Error Rate (BER) Optimization Using Hybrid Interleavers and Reinforcement Learning

## Aim of the Project

The primary goal of this project is to minimize the **Bit Error Rate (BER)** in a communication system using advanced interleaving techniques and optimization strategies. Specifically, we aimed to implement and compare different interleaving strategies (block, 3GPP, and hybrid interleavers), evaluate their performance under various signal-to-noise ratio (SNR) conditions, and finally optimize the interleaver configuration using **Reinforcement Learning (RL)** to reduce BER further.

By the end of the project, we aimed to not only evaluate BER for different configurations but also apply an RL agent to dynamically adjust interleaving configurations (rows, columns) for achieving the lowest possible BER.

The base paper can be downloaded from [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9532864).

---

## Project Overview

The project was carried out in three major stages:

1. **Basic Implementation**: Implementing the block interleaver, 3GPP interleaver, and hybrid interleaver with BER calculation.
2. **Graph-Based Performance Analysis**: Simulating these interleavers over various SNR values and plotting the BER vs. SNR graphs for different input lengths.
3. **Reinforcement Learning Integration**: Using PPO (Proximal Policy Optimization) to optimize the interleaver configuration dynamically, with the RL agent learning to reduce BER over multiple episodes by adjusting the rows and columns of the interleaver.

Each stage had unique challenges and improvements over the previous one, eventually leading to a robust system capable of reducing BER dynamically based on learned behavior.

---

## Where We Started: The Basic Interleaver Implementation

The project began with the implementation of **block interleavers** and **3GPP interleavers**, which are common techniques used in digital communication systems to improve BER performance. The core concept of an interleaver is to reorder bits before transmission so that bursts of errors (due to channel noise) are spread out, making them easier to correct on the receiving side.

### 1. Block Interleaver

A **block interleaver** works by filling a matrix of size `rows x cols` with input data, and then reading it out in column-major order. This reorders the bits in a way that spreads consecutive bit errors across the entire block, reducing the chance of error bursts effectively.

- **Formula**: The interleaver matrix has `rows * cols = input_length` where the input data is divided into `rows` rows and `cols` columns. After transposing the matrix, the output is a sequence that has been scrambled in a way that combats burst errors effectively.

```python
matrix = np.reshape(input_data, (rows, cols))
interleaved_data = matrix.T.flatten()
```

### 2. 3GPP Interleaver

The **3GPP interleaver** is used in turbo coding systems and provides a more complex way of reordering data with variable row and column permutations based on specific algorithms.

- **Formula**: The 3GPP interleaver uses predefined permutations depending on the input size to generate an interleaved sequence that is resilient to errors.

Both the block and 3GPP interleavers were evaluated independently for varying input lengths (16, 64, 256, and 1024), and BER values were calculated for different SNR values ranging from 0 to 2 dB.

---

## Then we progressed: Plotting Performance Graphs and Hybrid Interleaver

In the second stage, we focused on enhancing the interleaving process by developing a **hybrid interleaver** that combined the strengths of both block and 3GPP interleavers.

### 1. Hybrid Interleaver Implementation

The hybrid interleaver splits the input data into multiple sub-blocks and applies block interleaving to one half and 3GPP interleaving to the other half. This dual-interleaving approach attempts to gain the advantages of both techniques to reduce BER further.

- **Formula**: For input lengths of 16, 8 sub-blocks were created with alternating block and 3GPP interleaving. For larger input lengths (64, 256, 1024), the input data was divided into two sections: one for block interleaving and the other for 3GPP interleaving.

```python
block_data = input_data[:K // 2]
gpp_data = input_data[K // 2:]
interleaved_block = block_interleaver(block_data, K // 4, 2)
interleaved_gpp = three_gpp_interleaver(gpp_data)
```

### 2. BER Performance Simulation

After implementing these interleavers, we simulated their performance over multiple iterations and across different SNR values to generate **BER vs. SNR** plots. Each interleaver's performance was compared based on the BER values obtained from 5 iterations of simulation for each SNR value.

We observed that the **hybrid interleaver** consistently outperformed individual interleavers in scenarios with higher input lengths and moderate SNR values. The results were saved as 12 graphs (three for each input length) comparing the performance of **no interleaver**, **block interleaver**, and **hybrid interleaver**.

---

## Finally we arrived : Final Reinforcement Learning (RL) Optimization

In the final stage, we incorporated ***Reinforcement Learning (RL)** using the **PPO (Proximal Policy Optimization)** algorithm to optimize the configuration of the interleaver (i.e., the number of rows and columns).

### 1. RL Environment Setup

We created a custom Gym environment, `BEROptimizationEnv`, where the RL agent's action was to select the configuration (`rows`, `cols`) for the hybrid interleaver. The observation space was the current BER, and the reward function was designed to maximize BER reduction.

- **Reward Function**: The reward was the difference between the previous BER and the current BER, scaled to encourage the agent to minimize BER as much as possible

```python
reward = (self.previous_ber - average_ber) * 100
```

- **Exploration Penaltyn**: To encourage exploration and avoid repeated configurations, a small penalty was added if the agent revisited a configuration

```python
reward = (self.previous_ber - average_ber) * 100
```

### 2. PPO Algorithm

The PPO algorithm was chosen because it strikes a balance between exploration and exploitation, which is essential when searching for optimal configurations in a continuous action space like ours.

The RL agent was trained for multiple episodes, where it learned to optimize the interleaver configuration to reduce BER dynamically based on SNR values and feedback from the communication system.

---

## Challenges and Solutions

### Initial Reward Stagnation:
- Initially, the reward function wasn’t producing much variation, leading to stagnation in learning. We addressed this by amplifying the reward difference and adding penalties for repeated configurations, encouraging more diverse exploration.

### Action Space Granularity:
- The action space needed to be dynamic, allowing the agent to explore finer adjustments to the rows and columns to find the optimal configuration.

### Reward Scaling:
- We found that multiplying the BER difference by 100 in the reward function helped the agent make better adjustments during training, leading to faster convergence.

---

## Future Improvements

### Advanced RL Algorithms:
- Deep Q-Networks (DQN) or A3C (Asynchronous Advantage Actor-Critic) could be explored to enhance the agent's learning capabilities, especially in environments with more complex reward structures.

### Adaptive Interleaving:
- We could explore dynamically switching between block and 3GPP interleaving during a single transmission to further minimize BER based on real-time feedback.

### Multi-Agent Learning:
- Multiple agents could be trained to optimize different aspects of the communication system (such as modulation schemes, power control) in parallel, alongside interleaving.

---

## Conclusion

This project showcased the successful implementation of interleaving techniques and their optimization using reinforcement learning to minimize BER. We started with basic block and 3GPP interleavers, progressed to hybrid interleaving, and finally integrated RL to dynamically optimize the configuration. The results demonstrate that hybrid interleaving, when combined with RL optimization, significantly improves BER performance over a range of input lengths and SNR values.

Through continuous adjustments, we demonstrated that RL can be a powerful tool in communication system optimization, and with further exploration of advanced algorithms, even greater improvements can be achieved.

---

## Setup
- **Conda Environment**: Skip the first two lines if you have already installed conda.

```bash
# Download anaxonda3
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Install anaconda3 (follow the on-screen instructions)
bash ./Anaconda3-2024.02-1-Linux-x86_64.sh

# Create new environment named 'mlops' with necessary specifications
conda create -n networks python=3.8
conda activate networks
conda install pytorch orchvision torchaudio cudatoolkit
conda install -c conda-forge gym=0.21.0 opencv=4.5.5
pip install stable-baselines3==1.5.0 pygame
```
<br>

- **Training**: Code is given below to train the agent.

```bash
# Start training the RL agent
python3 RL_runner.py

# For basic implementation and graph viewwing
cd Older_Approaches
python3 runner.py

```
