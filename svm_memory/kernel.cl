__kernel void double_kernel(__global int *input, __global int *output, const int size)
{
    int i = get_global_id(0);

    if (i < size)
    {
        output[i] = input[i] * 2;
    }
}