#define CL_HPP_TARGET_OPENCL_VERSION 200 // OpenCL 2.0
#include <CL/cl.hpp>

#include <iostream>
#include <vector>
#include <fstream>

#include "../common/include/common.hpp"
#include "../common/include/OCLTools.hpp"

int main()
{
    OCLTools ocl_tools;
    std::vector<cl::Platform> available_platforms;
    cl::Platform::get(&available_platforms);
    if (available_platforms.size() == 0)
    {
        std::cerr << "Sorry, no OpenCL platforms found. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Available platforms:" << std::endl;
    for (auto &i : available_platforms)
    {
        std::cout << "\t- " << i.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    // Selecting platform
    cl::Platform default_platform = available_platforms.at(0);

    // Check available GPU devices
    std::vector<cl::Device> available_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &available_devices);

    if (available_devices.size() == 0)
    {
        std::cerr << "Sorry, looks like this platform doesn't have any GPU. Exiting" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Available GPUs in platform '" << default_platform.getInfo<CL_PLATFORM_NAME>() << "':" << std::endl;
    for (auto &i : available_devices)
    {
        std::cout << "\t- " << i.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    // Selecting device
    cl::Device default_device = available_devices.at(0);

    std::cout << "Using '" << default_device.getInfo<CL_DEVICE_NAME>() << "' of platform '" << default_device.getInfo<CL_DEVICE_PLATFORM>().getInfo<CL_PLATFORM_NAME>() << "'" << std::endl;

    // Creating context with selected device
    cl::Context context(default_device);

    // Creating host data
    const int size = 51;
    char *output = new char[size];

    // Creating commands queue
    cl::CommandQueue queue(context, default_device);

    // Creating program
    cl::Program::Sources sources;

    sources.push_back(ocl_tools.getKernelCode("kernel.cl"));

    cl::Program program(context, sources);

    if (program.build(default_device) != CL_SUCCESS)
    {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(EXIT_FAILURE);
    }

    cl::Kernel kernel(program, "hello_kernel");

    // Launching kernel
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * size);

    kernel.setArg(0, output_buffer);
    kernel.setArg(1, size);

    CHECK_ERROR(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange));

    // Reading output
    CHECK_ERROR(queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(char) * size, output));

    std::cout << "\nOutput: " << output << std::endl;

    // Cleaning up
    delete[] output;
    output = nullptr;
    queue.finish();

    return EXIT_SUCCESS;
}
