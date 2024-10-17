import numpy as np

def bpsk_modulate(bits):
    return 2 * bits - 1  # Map 0 -> -1 and 1 -> 1

def bpsk_demodulate(received_signal):
    return (received_signal >= 0).astype(int)  # Map positive values to 1 and negative to 0

def add_noise(signal, snr_db):
    snr_linear = 10**(snr_db / 10)
    noise_variance = 1 / (2 * snr_linear)
    noise = np.sqrt(noise_variance) * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal

def calculate_ber(original_signal, decoded_signal):
    errors = np.sum(original_signal != decoded_signal)
    ber = errors / len(original_signal)
    return ber

def turbo_decoder_with_snr(received_signal, snr_db, max_iterations=5):
    decoded_signal = np.zeros_like(received_signal)

    snr_linear = 10**(snr_db / 10)
    noise_variance = 1 / (2 * snr_linear)
    
    for iteration in range(max_iterations):
        soft_decoded = received_signal / noise_variance
        decoded_bits = np.where(soft_decoded > 0, 1, 0)
        decoded_signal = soft_decoded

        if has_converged(decoded_bits):
            break
    
    return decoded_bits

def has_converged(decoded_bits, tolerance=1e-3):
    return np.all(np.abs(np.diff(decoded_bits)) < tolerance)

def iterative_decoder(received_signal, snr_db, max_iterations=5):
    decoded_signal = np.zeros_like(received_signal)

    for iteration in range(max_iterations):
        decoded_signal = turbo_decoder_with_snr(received_signal, snr_db)
        
        if has_converged(decoded_signal):
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return decoded_signal

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

def block_interleaver(input_data, rows, cols):
    """ Block interleaver with padding for cases where input_data size != rows * cols """
    # Pad the input data to match the size of rows * cols
    if len(input_data) < rows * cols:
        padded_data = np.pad(input_data, (0, rows * cols - len(input_data)), 'constant', constant_values=0)
    else:
        padded_data = input_data[:rows * cols]  # Truncate if the data is too large
    
    # Reshape into rows x cols matrix
    matrix = np.reshape(padded_data, (rows, cols))
    
    # Perform interleaving by transposing and flattening the matrix
    interleaved_data = matrix.T.flatten()
    
    return interleaved_data

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

def block_deinterleaver(interleaved_data, rows, cols):
    """ Block deinterleaver with padding handling """
    # Reshape the interleaved data into a cols x rows matrix
    matrix = np.reshape(interleaved_data, (cols, rows))
    
    # Perform deinterleaving by transposing and flattening the matrix
    deinterleaved_data = matrix.T.flatten()
    
    return deinterleaved_data[:rows * cols]  # Return only the original length of data

def hybrid_deinterleaver(interleaved_data):
    K = len(interleaved_data)
    deinterleaved_result = []
    if K == 16:
        for i in range(8):
            sub_data = interleaved_data[i * 2: (i + 1) * 2]

            if i % 2 == 0:
                deinterleaved_block = block_deinterleaver(sub_data, 2, 1)
                deinterleaved_result.append(deinterleaved_block)
            else:
                deinterleaved_gpp = three_gpp_interleaver(sub_data)  # Assuming 3GPP interleaver is its own inverse
                deinterleaved_result.append(deinterleaved_gpp)
    else:
        block_data = interleaved_data[:K // 2]
        gpp_data = interleaved_data[K // 2:]

        deinterleaved_block = block_deinterleaver(block_data, K // 4, 2)
        deinterleaved_result.append(deinterleaved_block)

        deinterleaved_gpp = three_gpp_interleaver(gpp_data)  # Assuming 3GPP interleaver is its own inverse
        deinterleaved_result.append(deinterleaved_gpp)

    hybrid_deinterleaved = np.concatenate(deinterleaved_result)

    return hybrid_deinterleaved

def soft_decision_decoder(received_signal, snr):
    """ Soft-decision decoding based on likelihood of bit correctness """
    decoded_data = []
    threshold = 0.5  # This can be tuned based on SNR or other factors
    for bit in received_signal:
        if bit + snr/10 > threshold:  # This simulates soft-decoding based on SNR.
            decoded_data.append(1)
        else:
            decoded_data.append(0)
    return decoded_data

def simulate_no_interleaver(input_data, snr_db, decoder):
    modulated_signal = bpsk_modulate(input_data)
    noisy_signal = add_noise(modulated_signal, snr_db)
    decoded_bits = decoder(noisy_signal, snr_db)
    ber = calculate_ber(input_data, decoded_bits)
    return ber

def simulate_block_interleaver(input_data, snr_db, rows, cols, decoder):
    interleaved_data = block_interleaver(input_data, rows, cols)
    modulated_signal = bpsk_modulate(interleaved_data)
    noisy_signal = add_noise(modulated_signal, snr_db)
    decoded_bits = decoder(noisy_signal, snr_db)
    deinterleaved_bits = block_deinterleaver(decoded_bits, rows, cols)
    ber = calculate_ber(input_data, deinterleaved_bits)
    return ber

def simulate_hybrid_interleaver(input_data, snr_db, decoder):
    interleaved_data = hybrid_interleaver(input_data)
    modulated_signal = bpsk_modulate(interleaved_data)
    noisy_signal = add_noise(modulated_signal, snr_db)
    decoded_bits = decoder(noisy_signal, snr_db)
    deinterleaved_bits = hybrid_deinterleaver(decoded_bits)
    ber = calculate_ber(input_data, deinterleaved_bits)
    return ber