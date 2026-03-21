#include <iostream>
#include <vector>
#include <fstream>

#include "../common/include/common.hpp"

std::string getKernelCode(const char *file_name)
{
    std::ifstream kernel_file(file_name);
    std::string output, line;

    if (!kernel_file.is_open())
    {
        std::cerr << "ERROR: Unable to open kernel file '" << file_name << "'." << std::endl;
        throw std::runtime_error("Kernel file not found");
    }

    while (std::getline(kernel_file, line))
    {
        output += line += "\n";
    }

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

    if (is_cpu_set)
    {
        std::cout << "CPU selected: '" << cpu.getInfo<CL_DEVICE_NAME>() << "' from '" << cpu_plat.getInfo<CL_PLATFORM_NAME>() << "' platform." << std::endl;
    }
    else
    {
        std::cerr << "Could not find any CPU device in the available platforms." << std::endl;
    }

    if (is_gpu_set)
    {
        std::cout << "GPU selected: '" << gpu.getInfo<CL_DEVICE_NAME>() << "' from '" << gpu_plat.getInfo<CL_PLATFORM_NAME>() << "' platform." << std::endl;
    }
    else
    {
        std::cerr << "Could not find any GPU device in the available platforms." << std::endl;
    }

    return EXIT_SUCCESS;
}
