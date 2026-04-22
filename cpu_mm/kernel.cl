// Optimized for CPU using Vectorization (SIMD)

__kernel void matmul_cpu_vec(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size) 
{
    // Each thread still handles one row...
    const int row = get_global_id(1);
    
    // ...but now each thread handles eight columns at once
    // You must spawn fewer threads in dimension 0 (size / 8)
    const int col = get_global_id(0) * 8; 

    if (col == 0 && row == 0)
    {
        printf("OpenCL local range: (%d, %d)\n", get_local_size(0), get_local_size(1));
    }

    if (row < size && col < size) {
        
        // Accumulator for 8 separate dot products simultaneously
        float8 sum = (float8)(0.0f);
        
        for (int k = 0; k < size; k++) {
            // 1. Read ONE element of A and broadcast it across a float8 vector
            // e.g., if A is 2.5, a_vec becomes [2.5, 2.5, 2.5, 2.5, ...]
            float8 a_vec = (float8)(A[row * size + k]);
            
            // 2. Load 8 contiguous elements of B in a single fast memory read
            // This is incredibly cache-friendly and the CPU prefetcher loves it
            float8 b_vec = vload8(0, &B[k * size + col]);
            
            // 3. Multiply and accumulate 8 values in one hardware clock cycle (SIMD)
            sum += a_vec * b_vec;
        }
        
        // Write the 8 completed results back to global memory
        vstore8(sum, 0, &C[row * size + col]);
    }
}