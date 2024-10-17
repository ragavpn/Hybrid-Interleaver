import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from module import *

def run_simulation(input_lengths, snr_values, iterations, rows, cols, 
                   no_interleaver_decoder = soft_decision_decoder, 
                   block_interleaver_decoder = soft_decision_decoder, 
                   hybrid_interleaver_decoder = soft_decision_decoder):
    
    no_interleaver_bers = np.zeros((len(input_lengths), iterations, len(snr_values)))
    block_interleaver_bers = np.zeros((len(input_lengths), iterations, len(snr_values)))
    hybrid_interleaver_bers = np.zeros((len(input_lengths), iterations, len(snr_values)))

    for length_idx, input_length in enumerate(input_lengths):
        for iter_num in range(iterations):
            for snr_idx, snr in enumerate(snr_values):

                input_data = np.random.randint(0, 2, input_length)
                no_interleaver_bers[length_idx, iter_num, snr_idx] = simulate_no_interleaver(input_data, snr, no_interleaver_decoder)
                block_interleaver_bers[length_idx, iter_num, snr_idx] = simulate_block_interleaver(input_data, snr, rows[length_idx], cols[length_idx], block_interleaver_decoder)
                hybrid_interleaver_bers[length_idx, iter_num, snr_idx] = simulate_hybrid_interleaver(input_data, snr, hybrid_interleaver_decoder)

    fig, axes = plt.subplots(1, len(input_lengths), figsize=(18, 6))
    for length_idx, input_length in enumerate(input_lengths):
        ax = axes[length_idx]
        for iter_num in range(iterations):
            ax.plot(snr_values, no_interleaver_bers[length_idx, iter_num], marker='o', label=f'Iter {iter_num + 1}')
        ax.set_title(f'No Interleaver - {input_length} bits')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'Graphs/Stiched/no_interleaver_plots.png')
    plt.show()

    fig, axes = plt.subplots(1, len(input_lengths), figsize=(18, 6))
    for length_idx, input_length in enumerate(input_lengths):
        ax = axes[length_idx]
        for iter_num in range(iterations):
            ax.plot(snr_values, block_interleaver_bers[length_idx, iter_num], marker='o', label=f'Iter {iter_num + 1}')
        ax.set_title(f'Block Interleaver - {input_length} bits')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'Graphs/Stiched/block_interleaver_plots.png')
    plt.show()

    fig, axes = plt.subplots(1, len(input_lengths), figsize=(18, 6))
    for length_idx, input_length in enumerate(input_lengths):
        ax = axes[length_idx]
        for iter_num in range(iterations):
            ax.plot(snr_values, hybrid_interleaver_bers[length_idx, iter_num], marker='o', label=f'Iter {iter_num + 1}')
        ax.set_title(f'Hybrid Interleaver - {input_length} bits')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'Graphs/Stiched/hybrid_interleaver_{input_length}.png')
    plt.show()


input_lengths = [16, 64, 256, 1024]
snr_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
iterations = 5 

rows = [2, 4, 16, 32]
cols = [8, 16, 16, 32]

run_simulation(input_lengths, snr_values, iterations, rows, cols)
