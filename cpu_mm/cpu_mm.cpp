#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include "../common/include/common.hpp"

std::string getKernelCode(const char *file_path)
{
    std::ifstream kernel_file(file_path);

    if (!kernel_file.is_open())
    {
        std::cerr << "ERROR: Unable to open kernel file '" << file_path << "'." << std::endl;
        throw std::runtime_error("Kernel file not found");
    }

    std::string output(
        (std::istreambuf_iterator<char>(kernel_file)),
        (std::istreambuf_iterator<char>()));

    kernel_file.close();

    return output;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " [MATRIX_DIM]" << std::endl;
        return EXIT_FAILURE;
    }

    const size_t matrix_dim = std::stoul(argv[1]);

    if (matrix_dim % 8 != 0)
    {
        std::cerr << "Error: MATRIX_DIM must be a multiple of 8 for this kernel to work correctly." << std::endl;
        return EXIT_FAILURE;
    }

    // CPU and GPU OpenCL vars
    cl::Platform cpu_plat, gpu_plat;
    cl::Device cpu, gpu;

    // Flags
    bool is_cpu_set = false;
    bool is_gpu_set = false;

    // Get available platforms
    std::vector<cl::Platform> available_platforms;
    cl::Platform::get(&available_platforms);

    if (available_platforms.size() == 0)
    {
        std::cerr << "Sorry, no OpenCL platforms found. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Printing available platforms
    std::cout << "Available platforms:" << std::endl;
    for (auto &i : available_platforms)
    {
        std::cout << "\t- " << i.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }
    std::cout << std::endl;

    // Printing available devices in each platform and selecting the CPU and GPU
    std::vector<cl::Device> available_devices;
    for (auto &p : available_platforms)
    {
        std::cout << "Available devices in '" << p.getInfo<CL_PLATFORM_NAME>() << "' platform:" << std::endl;

        // Get devices
        available_devices.clear();
        p.getDevices(CL_DEVICE_TYPE_ALL, &available_devices);

        // Print them
        for (auto &d : available_devices)
        {
            std::cout << "\t- " << d.getInfo<CL_DEVICE_NAME>() << std::endl;

            // Check if it's a CPU or GPU and if we haven't set one of them yet, set it
            if (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU && !is_cpu_set)
            {
                cpu = d;
                cpu_plat = p;
                is_cpu_set = true;
            }
            else if (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU && !is_gpu_set)
            {
                gpu = d;
                gpu_plat = p;
                is_gpu_set = true;
            }
        }
        std::cout << std::endl;
    }

    // Print selected CPU and GPU. If we didn't find one, exit
    if (is_cpu_set)
    {
        std::cout << "CPU selected: '" << cpu.getInfo<CL_DEVICE_NAME>() << "' from '" << cpu_plat.getInfo<CL_PLATFORM_NAME>() << "' platform." << std::endl;
    }
    else
    {
        std::cerr << "Could not find any CPU device in the available platforms. Exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (is_gpu_set)
    {
        std::cout << "GPU selected: '" << gpu.getInfo<CL_DEVICE_NAME>() << "' from '" << gpu_plat.getInfo<CL_PLATFORM_NAME>() << "' platform." << std::endl;
    }
    else
    {
        std::cerr << "Could not find any GPU device in the available platforms. Exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // ------- Creating context and command queues for CPU and GPU -------
    cl::Context cpu_context(cpu);
    cl::CommandQueue cpu_queue(cpu_context, cpu);

    cl::Context gpu_context(gpu);
    cl::CommandQueue gpu_queue(gpu_context, gpu);

    // Error variable
    cl_int err;

    // Get memory alignment info for the CPU device.
    cl_uint align_bits = cpu.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
    cl_uint align_bytes = align_bits / 8;

    std::cout << "\nDevice memory base address alignment for CPU: " << align_bytes << " bytes" << std::endl;

    // Getting OpenCL code from file
    std::string kernel_code = getKernelCode("kernel.cl");

    // Creating and building program for cpu
    cl::Program program(cpu_context, kernel_code);

    if (program.build(cpu) != CL_SUCCESS)
    {
        std::cerr << "Error building kernel for CPU: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cpu) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Creating kernel
    cl::Kernel kernel(program, "matmul_cpu_vec", &err);
    CHECK_ERROR(err);

    // Creating SVM buffers for input and output
    const size_t buffer_size_bytes = sizeof(float) * matrix_dim * matrix_dim;

    std::cout << "\nAllocating matrices of size " << matrix_dim << "x" << matrix_dim << " (" << buffer_size_bytes / (1024 * 1024) << " MB each) in SVM memory..." << std::endl;

    float *host_mat1 = static_cast<float *>(
        clSVMAlloc(cpu_context(), CL_MEM_READ_WRITE, buffer_size_bytes, 0));

    float *host_mat2 = static_cast<float *>(
        clSVMAlloc(cpu_context(), CL_MEM_READ_WRITE, buffer_size_bytes, 0));

    float *host_result = static_cast<float *>(
        clSVMAlloc(cpu_context(), CL_MEM_READ_WRITE, buffer_size_bytes, 0));

    // I'm using coarse-grained SVM, so we need to explicitly map the input buffer before writing to it.
    // This ensures that it's safe to access the memory.
    CHECK_ERROR(cpu_queue.enqueueMapSVM(
        host_mat1,
        CL_TRUE,                    // Block until the mapping is ready
        CL_MAP_WRITE | CL_MAP_READ, // We will write and read from this buffer
        buffer_size_bytes           // Map the entire buffer
        ));

    CHECK_ERROR(cpu_queue.enqueueMapSVM(
        host_mat2,
        CL_TRUE,                    // Block until the mapping is ready
        CL_MAP_WRITE | CL_MAP_READ, // We will write and read from this buffer
        buffer_size_bytes           // Map the entire buffer
        ));

    // Check memory alignment of host_arr and host_result
    uintptr_t addr_host_arr_1 = reinterpret_cast<uintptr_t>(host_mat1);
    uintptr_t addr_host_arr_2 = reinterpret_cast<uintptr_t>(host_mat2);

    if (addr_host_arr_1 % align_bytes == 0 && addr_host_arr_2 % align_bytes == 0)
    {
        std::cout << "matrix 1 is properly aligned at address: " << host_mat1 << std::endl;
        std::cout << "matrix 2 is properly aligned at address: " << host_mat2 << std::endl;
    }
    else
    {
        std::cerr << "Warning: host_input is NOT properly aligned! Address: " << host_mat1 << std::endl;
    }

    // Initializing matrices with some values
    for (size_t i = 0; i < matrix_dim; i++)
    {
        for (size_t j = 0; j < matrix_dim; j++)
        {
            size_t idx = i * matrix_dim + j;

            host_mat1[idx] = 1.0f;
            host_mat2[idx] = 1.0f;
        }
    }

    // Unmap the now that we are done writing and reading. This returns control of the memory back to OpenCL.
    CHECK_ERROR(cpu_queue.enqueueUnmapSVM(host_mat1));
    CHECK_ERROR(cpu_queue.enqueueUnmapSVM(host_mat2));

    // Setting kernel arguments
    kernel.setArg(0, host_mat1);
    kernel.setArg(1, host_mat2);
    kernel.setArg(2, host_result);
    kernel.setArg(3, static_cast<int>(matrix_dim));

    std::cout << "Launching kernel in CPU..." << std::endl;

    // Launching kernel in CPU
    cl::NDRange global_range(matrix_dim / 8, matrix_dim);
    // cl::NDRange local_range(8,8);
    cl::NDRange local_range = cl::NullRange;

    std::cout << "Global range: (" << global_range[0] << ", " << global_range[1] << ")" << std::endl;
    // std::cout << "Local range: (" << local_range[0] << ", " << local_range[1] << ")" << std::endl;
    std::cout << "Local range: (letting OpenCL decide)" << std::endl;

    // Start measuring time
    auto start_time = std::chrono::steady_clock::now();

    CHECK_ERROR(cpu_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_range, local_range));
    CHECK_ERROR(cpu_queue.finish()); // Wait for the kernel to finish before trying to read the results

    auto end_time = std::chrono::steady_clock::now();
    
    // Get duration in milliseconds
    int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Now, we map the output buffer to read the results back.
    CHECK_ERROR(cpu_queue.enqueueMapSVM(
        host_result,
        CL_TRUE,          // Block until the mapping is ready
        CL_MAP_READ,      // We will only read from this buffer
        buffer_size_bytes // Map the entire buffer
        ));

    // Print the results.
    std::cout << "\nFirst 10 results after kernel run:" << std::endl;
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << "* mapped_result[" << i << "] = " << host_result[i] << std::endl;
    }
    std::cout << std::endl;

    // Print execution time
    std::cout << "Kernel execution time on CPU: " << duration_ms << " ms" << std::endl;

    // 3. Unmap the buffer now that we are done reading it.
    CHECK_ERROR(cpu_queue.enqueueUnmapSVM(host_result));

    // Ensure all operations are finished before exiting
    cpu_queue.finish();

    // Delete SVM buffers
    clSVMFree(cpu_context(), host_mat1);
    clSVMFree(cpu_context(), host_mat2);
    clSVMFree(cpu_context(), host_result);

    return EXIT_SUCCESS;
}
