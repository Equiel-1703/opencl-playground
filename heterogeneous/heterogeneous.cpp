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

    // Printing available devices in each platform
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
        }
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
