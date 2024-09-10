# The 3GPP interleaver’s pattern primarily depends on the input length. 
# Unlike block interleavers, which might use parameters like the number of rows (M) 
# and columns (N), the 3GPP interleaver adjusts its interleaving pattern based on the length of the input data.

import numpy as np

# Block Interleaver
def block_interleaver(input_data, rows, cols):
    if len(input_data) != rows * cols:
        raise ValueError("Input data size must match rows * cols")
    
    # Reshape the input into a matrix and transpose it
    matrix = np.reshape(np.array(input_data), (rows, cols))
    
    # Transpose the matrix and flatten it
    interleaved_data = matrix.T.flatten()
    
    return interleaved_data

# 3GPP Interleaver
def three_gpp_interleaver(input_data):
    K = len(input_data)
    
    # Determine number of rows (R)
    if 1 <= K <= 24:
        R = 1
    elif 25 <= K <= 159:
        R = 5
    elif (160 <= K <= 200) or (481 <= K <= 530):
        R = 10
    else:
        R = 20
    
    # Determine number of columns (C)
    if 481 <= K <= 530:
        C = 53
    else:
        C = 1
        while C * R < K:
            C += 1
    
    # Input data is written into the matrix row by row
    matrix = np.zeros((R, C), dtype=int)
    
    for idx, bit in enumerate(input_data):
        row = idx // C
        col = idx % C
        matrix[row, col] = bit
    
    # Perform intra-row permutations
    for i in range(R):
        if C % 2 == 0:  # If C is even, shift even rows
            matrix[i] = np.roll(matrix[i], shift=i)
        else:  # If C is odd, shift odd rows
            matrix[i] = np.roll(matrix[i], shift=-i)
    
    # Inter-row permutations based on a random permutation pattern
    row_permutation_pattern = np.random.permutation(R)
    matrix = matrix[row_permutation_pattern, :]
    
    # Flatten the matrix to get the interleaved sequence
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



# Example data for sizes 16, 64, 256, and 1024
input_data_16 = list(range(16))
input_data_64 = list(range(64))
input_data_256 = list(range(256))
input_data_1024 = list(range(1024))

# Block interleaver usage (e.g., with a 2x8 matrix for data size 16)
block_interleaved_16 = block_interleaver(input_data_16, 2, 8)
print("Block Interleaved Data (16):", block_interleaved_16)

# 3GPP interleaver usage for data size 16
gpp_interleaved_16 = three_gpp_interleaver(input_data_16)
print("3GPP Interleaved Data (16):", gpp_interleaved_16)

# Hybrid interleaver usage for data size 16
hybrid_interleaved_16 = hybrid_interleaver(input_data_16)
print("Hybrid Interleaved Data (16):", hybrid_interleaved_16)

# Block interleaver usage (e.g., with a 4x16 matrix for data size 64)
block_interleaved_64 = block_interleaver(input_data_64, 4, 16)
print("Block Interleaved Data (64):", block_interleaved_64)

# 3GPP interleaver usage for data size 64
gpp_interleaved_64 = three_gpp_interleaver(input_data_64)
print("3GPP Interleaved Data (64):", gpp_interleaved_64)

# Hybrid interleaver usage for data size 64
hybrid_interleaved_64 = hybrid_interleaver(input_data_64)
print("Hybrid Interleaved Data (64):", hybrid_interleaved_64)

# Block interleaver usage (e.g., with a 16x16 matrix for data size 256)
block_interleaved_256 = block_interleaver(input_data_256, 16, 16)
print("Block Interleaved Data (256):", block_interleaved_256)

# 3GPP interleaver usage for data size 256
gpp_interleaved_256 = three_gpp_interleaver(input_data_256)
print("3GPP Interleaved Data (256):", gpp_interleaved_256)

# Hybrid interleaver usage for data size 256
hybrid_interleaved_256 = hybrid_interleaver(input_data_256)
print("Hybrid Interleaved Data (256):", hybrid_interleaved_256)

# Block interleaver usage (e.g., with a 32x32 matrix for data size 1024)
block_interleaved_1024 = block_interleaver(input_data_1024, 32, 32)
print("Block Interleaved Data (1024):", block_interleaved_1024)

# 3GPP interleaver usage for data size 1024
gpp_interleaved_1024 = three_gpp_interleaver(input_data_1024)
print("3GPP Interleaved Data (1024):", gpp_interleaved_1024)

# Hybrid interleaver usage for data size 1024
hybrid_interleaved_1024 = hybrid_interleaver(input_data_1024)
print("Hybrid Interleaved Data (1024):", hybrid_interleaved_1024)


