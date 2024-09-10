import numpy as np
import matplotlib.pyplot as plt

# BPSK Modulation and Demodulation
def bpsk_modulate(bits):
    return 2 * bits - 1  # Map 0 -> -1 and 1 -> 1

def bpsk_demodulate(received_signal):
    return (received_signal >= 0).astype(int)  # Map positive values to 1 and negative to 0

# AWGN Noise Addition
def add_awgn_noise(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * np.random.randn(*signal.shape)
    return signal + noise

# Bit Error Rate Calculation
def calculate_ber(original_bits, received_bits):
    return np.sum(original_bits != received_bits) / len(original_bits)

def max_log_map_decoder(received_bits, Lc=1.0):
    num_bits = len(received_bits)
    alpha = np.zeros(num_bits)
    beta = np.zeros(num_bits)
    gamma = np.zeros(num_bits)
    decoded_bits = np.zeros(num_bits)

    # Forward recursion
    for i in range(1, num_bits):
        gamma[i] = Lc * received_bits[i]
        alpha[i] = max(alpha[i-1] + gamma[i], alpha[i-1] - gamma[i])

    # Backward recursion
    for i in range(num_bits-2, -1, -1):
        beta[i] = max(beta[i+1] + gamma[i+1], beta[i+1] - gamma[i+1])

    # Calculate LLR and make decisions
    for i in range(num_bits):
        LLR = alpha[i] + beta[i] + Lc * received_bits[i]
        decoded_bits[i] = 1 if LLR >= 0 else 0

    return decoded_bits

def log_map_decoder(received_bits, Lc=1.0):
    num_bits = len(received_bits)
    alpha = np.zeros(num_bits)
    beta = np.zeros(num_bits)
    gamma = np.zeros(num_bits)
    decoded_bits = np.zeros(num_bits)

    # Forward recursion
    for i in range(1, num_bits):
        gamma[i] = Lc * received_bits[i]
        alpha[i] = np.logaddexp(alpha[i-1] + gamma[i], alpha[i-1] - gamma[i])

    # Backward recursion
    for i in range(num_bits-2, -1, -1):
        beta[i] = np.logaddexp(beta[i+1] + gamma[i+1], beta[i+1] - gamma[i+1])

    # Calculate LLR and make decisions
    for i in range(num_bits):
        LLR = alpha[i] + beta[i] + Lc * received_bits[i]
        decoded_bits[i] = 1 if LLR >= 0 else 0

    return decoded_bits

def threshold_max_log_map_decoder(received_bits, Lc=1.0, threshold=0.5):
    num_bits = len(received_bits)
    alpha = np.zeros(num_bits)
    beta = np.zeros(num_bits)
    gamma = np.zeros(num_bits)
    decoded_bits = np.zeros(num_bits)

    # Forward recursion
    for i in range(1, num_bits):
        gamma[i] = Lc * received_bits[i]
        alpha[i] = max(alpha[i-1] + gamma[i], alpha[i-1] - gamma[i])

    # Backward recursion
    for i in range(num_bits-2, -1, -1):
        beta[i] = max(beta[i+1] + gamma[i+1], beta[i+1] - gamma[i+1])

    # Calculate LLR and make decisions
    for i in range(num_bits):
        LLR = alpha[i] + beta[i] + Lc * received_bits[i]
        decoded_bits[i] = 1 if LLR >= threshold else 0

    return decoded_bits



# Block Interleaver
def block_interleaver(input_data, rows, cols):
    if len(input_data) != rows * cols:
        raise ValueError("Input data size must match rows * cols")
    
    matrix = np.reshape(np.array(input_data), (rows, cols))
    interleaved_data = matrix.T.flatten()
    
    return interleaved_data

# 3GPP Interleaver
def three_gpp_interleaver(input_data):
    K = len(input_data)
    
    if 1 <= K <= 24:
        R = 1
    elif 25 <= K <= 159:
        R = 5
    elif (160 <= K <= 200) or (481 <= K <= 530):
        R = 10
    else:
        R = 20
    
    C = 1
    while C * R < K:
        C += 1
    
    matrix = np.zeros((R, C), dtype=int)
    for idx, bit in enumerate(input_data):
        row = idx // C
        col = idx % C
        matrix[row, col] = bit
    
    for i in range(R):
        if C % 2 == 0:
            matrix[i] = np.roll(matrix[i], shift=i)
        else:
            matrix[i] = np.roll(matrix[i], shift=-i)

    row_permutation_pattern = np.random.permutation(R)
    matrix = matrix[row_permutation_pattern, :]
    
    interleaved_data = matrix.flatten()[:K]
    
    return interleaved_data

# Hybrid Interleaver
def hybrid_interleaver(input_data):
    # Determine the number of sub-interleavers based on the input length
    K = len(input_data)
    interleaved_result = []
    if K == 16:
        for i in range(8):
            # Extract the sub-data for this interleaver
            sub_data = input_data[i * 2: (i + 1) * 2]

            # For half of the subgroups, use Block Interleaver
            if i % 2 == 0:
                interleaved_block = block_interleaver(sub_data, 2, 1)
                interleaved_result.append(interleaved_block)

            # For the other half of the subgroups, use 3GPP Interleaver
            else:
                interleaved_gpp = three_gpp_interleaver(sub_data)
                interleaved_result.append(interleaved_gpp)
    
    else:
        block_data = input_data[:K // 2]
        gpp_data = input_data[K // 2:]

        # Use Block Interleaver for the first half of the data
        interleaved_block = block_interleaver(block_data, K // 4, 2)
        interleaved_result.append(interleaved_block)

        # Use 3GPP Interleaver for the second half of the data
        interleaved_gpp = three_gpp_interleaver(gpp_data)
        interleaved_result.append(interleaved_gpp)

    # Combine all interleaved subgroups
    hybrid_interleaved = np.concatenate(interleaved_result)

    return hybrid_interleaved


def simulate_no_interleaver(input_data, snr_db, decoder):
    modulated_signal = bpsk_modulate(input_data)
    noisy_signal = add_awgn_noise(modulated_signal, snr_db)
    received_signal = bpsk_demodulate(noisy_signal)
    decoded_bits = decoder(received_signal)
    ber = calculate_ber(input_data, decoded_bits)
    return ber

def simulate_block_interleaver(input_data, snr_db, rows, cols, decoder):
    interleaved_data = block_interleaver(input_data, rows, cols)
    modulated_signal = bpsk_modulate(interleaved_data)
    noisy_signal = add_awgn_noise(modulated_signal, snr_db)
    received_signal = bpsk_demodulate(noisy_signal)
    decoded_bits = decoder(received_signal)
    ber = calculate_ber(input_data, decoded_bits)
    return ber

def simulate_hybrid_interleaver(input_data, snr_db, decoder):
    interleaved_data = hybrid_interleaver(input_data)
    modulated_signal = bpsk_modulate(interleaved_data)
    noisy_signal = add_awgn_noise(modulated_signal, snr_db)
    received_signal = bpsk_demodulate(noisy_signal)
    decoded_bits = decoder(received_signal)
    ber = calculate_ber(input_data, decoded_bits)
    return ber

def run_simulation(input_length, snr_values, iterations, rows, cols, no_interleaver_decoder, block_interleaver_decoder, hybrid_interleaver_decoder):
    no_interleaver_bers = []
    block_interleaver_bers = []
    hybrid_interleaver_bers = []

    for snr in snr_values:
        no_interleaver_iter_bers = []
        block_interleaver_iter_bers = []
        hybrid_interleaver_iter_bers = []

        for _ in range(iterations):
            input_data = np.random.randint(0, 2, input_length)

            ber_no_interleaver = simulate_no_interleaver(input_data, snr, no_interleaver_decoder)
            no_interleaver_iter_bers.append(ber_no_interleaver)

            ber_block_interleaver = simulate_block_interleaver(input_data, snr, rows, cols, block_interleaver_decoder)
            block_interleaver_iter_bers.append(ber_block_interleaver)

            ber_hybrid_interleaver = simulate_hybrid_interleaver(input_data, snr, hybrid_interleaver_decoder)
            hybrid_interleaver_iter_bers.append(ber_hybrid_interleaver)

        no_interleaver_bers.append(np.mean(no_interleaver_iter_bers))
        block_interleaver_bers.append(np.mean(block_interleaver_iter_bers))
        hybrid_interleaver_bers.append(np.mean(hybrid_interleaver_iter_bers))

    plt.figure()
    plt.plot(snr_values, no_interleaver_bers, marker='o', label='No Interleaver')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title(f'BER vs SNR (No Interleaver) - {input_length} bits')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'implementation_graphs/no_interleaver_{input_length}.png')

    plt.figure()
    plt.plot(snr_values, block_interleaver_bers, marker='o', label='Block Interleaver')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title(f'BER vs SNR (Block Interleaver) - {input_length} bits')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'implementation_graphs/block_interleaver_{input_length}.png')

    plt.figure()
    plt.plot(snr_values, hybrid_interleaver_bers, marker='o', label='Hybrid Interleaver')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title(f'BER vs SNR (Hybrid Interleaver) - {input_length} bits')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'implementation_graphs/hybrid_interleaver_{input_length}.png')

snr_values = [0, 2, 4, 6, 8, 10]
iterations = 5

run_simulation(16, snr_values, iterations, rows=2, cols=8, no_interleaver_decoder=log_map_decoder, block_interleaver_decoder=max_log_map_decoder, hybrid_interleaver_decoder=max_log_map_decoder)
run_simulation(64, snr_values, iterations, rows=4, cols=16, no_interleaver_decoder=threshold_max_log_map_decoder, block_interleaver_decoder=max_log_map_decoder, hybrid_interleaver_decoder=max_log_map_decoder)
run_simulation(256, snr_values, iterations, rows=16, cols=16, no_interleaver_decoder=max_log_map_decoder, block_interleaver_decoder=max_log_map_decoder, hybrid_interleaver_decoder=log_map_decoder)
run_simulation(1024, snr_values, iterations, rows=32, cols=32, no_interleaver_decoder=threshold_max_log_map_decoder, block_interleaver_decoder=max_log_map_decoder, hybrid_interleaver_decoder=threshold_max_log_map_decoder)
