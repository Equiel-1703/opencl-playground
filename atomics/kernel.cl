__kernel void inc_kernel(volatile __global atomic_int *counter, __global int *output_array)
{
    int tid = get_global_id(0);
    
    int previous_val = atomic_fetch_add_explicit(
        counter, 
        1,                       // The value to add
        memory_order_relaxed,    // We only care about the count itself, not strict memory ordering [cite: 55]
        memory_scope_device      // The counter is in global memory, shared across the whole device [cite: 55]
    );

    output_array[tid] = previous_val;
}