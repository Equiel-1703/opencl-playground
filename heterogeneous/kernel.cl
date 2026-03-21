__kernel void hello_kernel(__global char *output, int out_size) {
    int i = get_global_id(0);
    const char msg[] = "Hello world from GPU!";
    if (i < out_size && i < sizeof(msg))
    {
        output[i] = msg[i];
    }
}