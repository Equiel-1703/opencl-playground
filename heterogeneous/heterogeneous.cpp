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

    // Getting OpenCL code from file
    std::string kernel_code = getKernelCode("kernel.cl");
    
    // Creating program for cpu
    cl::Program program(cpu_context, kernel_code);

    if (program.build(cpu) != CL_SUCCESS)
    {
        std::cerr << "Error building kernel for CPU: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cpu) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Creating buffer for 32 chars (enough) in the CPU context
    const size_t len_buffer = 32;
    std::string host_output(len_buffer, '\0');
    cl::Buffer output_buffer(cpu_context, CL_MEM_READ_WRITE, sizeof(char) * len_buffer);

    // Creating kernel and setting arguments
    cl::Kernel kernel(program, "hello_kernel");

    kernel.setArg(0, output_buffer);
    kernel.setArg(1, static_cast<int>(len_buffer));

    std::cout << "\nLaunching kernel in CPU..." << std::endl;

    // Launching kernel in CPU
    CHECK_ERROR(cpu_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(len_buffer), cl::NullRange));

    // Reading output
    CHECK_ERROR(cpu_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(char) * len_buffer, (void *)host_output.data()));

    std::cout << "Output: " << host_output.c_str() << std::endl;

    cpu_queue.finish();

    return EXIT_SUCCESS;
}
