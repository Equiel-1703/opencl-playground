__kernel void vector_sum(__global const float *a, __global const float *b, __global float *result, int size)
{
    int i = get_global_id(0);
    if (i < size) {
        result[i] = a[i] + b[i];
    }
}