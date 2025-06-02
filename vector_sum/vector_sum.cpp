#include <iostream>

#include "../common/include/OCLInterface.hpp"

#define ARRAY_SIZE 1024

int main()
{
    OCLInterface open_cl;

    // Select the default OpenCL platform
    open_cl.selectDefaultPlatform();
    // Select the default device of type GPU
    open_cl.selectDefaultDevice(CL_DEVICE_TYPE_GPU);

    // Now let's create a program with our kernel source code
    cl::Program program = open_cl.createProgram("vector_sum.cl");

    // Now let's create a kernel object
    cl::Kernel kernel = open_cl.createKernel(program, "vector_sum");

    // With our kernel created, now we need to set up the memory objects for execution
    // Let's start with the host memory
    float a[ARRAY_SIZE], b[ARRAY_SIZE], result[ARRAY_SIZE];

    for (size_t i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = b[i] = i + 1;
    }

    // Now the device memory objects
    cl::Buffer buffer_a = open_cl.createBuffer(ARRAY_SIZE * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a);
    cl::Buffer buffer_b = open_cl.createBuffer(ARRAY_SIZE * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b);
    cl::Buffer buffer_result = open_cl.createBuffer(ARRAY_SIZE * sizeof(float), CL_MEM_WRITE_ONLY);

    // Set the kernel arguments
    kernel.setArg(0, buffer_a);
    kernel.setArg(1, buffer_b);
    kernel.setArg(2, buffer_result);
    kernel.setArg(3, static_cast<cl_int>(ARRAY_SIZE));

    // Now we can execute the kernel
    cl::NDRange global_range(ARRAY_SIZE);
    cl::NDRange local_range(1); // Using 1 work item per work group

    open_cl.executeKernel(kernel, global_range, local_range);

    // Read the result back to the host
    open_cl.readBuffer(buffer_result, result, ARRAY_SIZE * sizeof(float));

    // Print the result
    std::cout << "\nResult of vector sum: [";
    for (size_t i = 0; i < ARRAY_SIZE; i++)
    {
        std::cout << result[i] << (i < ARRAY_SIZE - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    return EXIT_SUCCESS;
}
