#include <iostream>
#include <vector>
#include <fstream>

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

int main()
{
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
    cl::Kernel kernel(program, "double_kernel", &err);
    CHECK_ERROR(err);

    // Creating SVM buffers for input and output
    const size_t buffer_len = 1024;
    const size_t buffer_size_bytes = sizeof(int) * buffer_len;

    int *host_input = static_cast<int *>(
        clSVMAlloc(cpu_context(), CL_MEM_READ_WRITE, buffer_size_bytes, 0));

    int *host_result = static_cast<int *>(
        clSVMAlloc(cpu_context(), CL_MEM_READ_WRITE, buffer_size_bytes, 0));

    // I'm using coarse-grained SVM, so we need to explicitly map the input buffer before writing to it.
    // This ensures that it's safe to access the memory.
    CHECK_ERROR(cpu_queue.enqueueMapSVM(
        host_input,
        CL_TRUE,                    // Block until the mapping is ready
        CL_MAP_WRITE | CL_MAP_READ, // We will write and read from this buffer
        buffer_size_bytes           // Map the entire buffer
        ));

    // Check memory alignment of host_arr and host_result
    uintptr_t addr_host_arr = reinterpret_cast<uintptr_t>(host_input);

    if (addr_host_arr % align_bytes == 0)
    {
        std::cout << "host_input is properly aligned at address: " << host_input << std::endl;
    }
    else
    {
        std::cerr << "Warning: host_input is NOT properly aligned! Address: " << host_input << std::endl;
    }

    // Initializing input data
    for (size_t i = 0; i < buffer_len; i++)
    {
        host_input[i] = static_cast<int>(i) + 1;
    }

    // Print first 10 input values
    std::cout << "\nFirst 10 input values:" << std::endl;
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << "* host_arr[" << i << "] = " << host_input[i] << std::endl;
    }
    std::cout << std::endl;

    // Unmap the input buffer now that we are done writing and reading. This returns control of the memory back to OpenCL.
    CHECK_ERROR(cpu_queue.enqueueUnmapSVM(host_input));

    // Setting kernel arguments
    kernel.setArg(0, host_input);
    kernel.setArg(1, host_result);
    kernel.setArg(2, static_cast<int>(buffer_len));

    std::cout << "Launching kernel in CPU..." << std::endl;

    // Launching kernel in CPU
    CHECK_ERROR(cpu_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(buffer_len), cl::NullRange));

    // Now, we map the output buffer to read the results back.
    CHECK_ERROR(cpu_queue.enqueueMapSVM(
        host_result,
        CL_TRUE,                   // Block until the mapping is ready
        CL_MAP_READ,               // We will only read from this buffer
        buffer_size_bytes          // Map the entire buffer
        ));

    // Print the results.
    std::cout << "\nFirst 10 results after kernel run:" << std::endl;
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << "* mapped_result[" << i << "] = " << host_result[i] << std::endl;
    }
    std::cout << std::endl;

    // 3. Unmap the buffer now that we are done reading it.
    CHECK_ERROR(cpu_queue.enqueueUnmapSVM(host_result));

    // Ensure all operations are finished before exiting
    cpu_queue.finish();

    // Delete SVM buffers
    clSVMFree(cpu_context(), host_input);
    clSVMFree(cpu_context(), host_result);

    return EXIT_SUCCESS;
}
