import numpy as np
import matplotlib.pyplot as plt
from module import *

def run_simulation(input_length, snr_values, iterations, rows, cols, 
                   no_interleaver_decoder = soft_decision_decoder, 
                   block_interleaver_decoder = soft_decision_decoder, 
                   hybrid_interleaver_decoder = soft_decision_decoder):
    
    # Prepare arrays to store BER values for each iteration
    no_interleaver_bers = np.zeros((iterations, len(snr_values)))
    block_interleaver_bers = np.zeros((iterations, len(snr_values)))
    hybrid_interleaver_bers = np.zeros((iterations, len(snr_values)))

    for iter_num in range(iterations):
        for snr_idx, snr in enumerate(snr_values):
            # Generate random input data
            input_data = np.random.randint(0, 2, input_length)

            # Run simulation for No Interleaver
            no_interleaver_bers[iter_num, snr_idx] = simulate_no_interleaver(input_data, snr, no_interleaver_decoder)

            # Run simulation for Block Interleaver
            block_interleaver_bers[iter_num, snr_idx] = simulate_block_interleaver(input_data, snr, rows, cols, block_interleaver_decoder)

            # Run simulation for Hybrid Interleaver
            hybrid_interleaver_bers[iter_num, snr_idx] = simulate_hybrid_interleaver(input_data, snr, hybrid_interleaver_decoder)

    # Plotting 5 different lines for No Interleaver
    plt.figure()
    for iter_num in range(iterations):
        plt.plot(snr_values, no_interleaver_bers[iter_num], marker='o', label=f'Iteration {iter_num + 1}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title(f'BER vs SNR (No Interleaver) - {input_length} bits')
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(f'implementation_graphs/no_interleaver_{input_length}.png')

    # Plotting 5 different lines for Block Interleaver
    plt.figure()
    for iter_num in range(iterations):
        plt.plot(snr_values, block_interleaver_bers[iter_num], marker='o', label=f'Iteration {iter_num + 1}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title(f'BER vs SNR (Block Interleaver) - {input_length} bits')
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(f'implementation_graphs/block_interleaver_{input_length}.png')

    # Plotting 5 different lines for Hybrid Interleaver
    plt.figure()
    for iter_num in range(iterations):
        plt.plot(snr_values, hybrid_interleaver_bers[iter_num], marker='o', label=f'Iteration {iter_num + 1}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title(f'BER vs SNR (Hybrid Interleaver) - {input_length} bits')
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(f'implementation_graphs/hybrid_interleaver_{input_length}.png')

# Run the simulation for different input lengths
snr_values = [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
iterations = 5

run_simulation(16, snr_values, iterations, rows=2, cols=8)
run_simulation(64, snr_values, iterations, rows=4, cols=16)
run_simulation(256, snr_values, iterations, rows=16, cols=16)
run_simulation(1024, snr_values, iterations, rows=32, cols=32)