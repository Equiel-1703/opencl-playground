#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200 // OpenCL 2.0
#include <CL/cl.hpp>

#include <iostream>
#include <fstream>
#include <vector>

class OCLInterface
{
private:
    cl::Platform selected_platform;
    cl::Device selected_device;
    cl::Context context;

    cl::CommandQueue command_queue;

    void createContext();
    void createCommandQueue();
    
    std::string getKernelCode(const char *file_name);
public:
    std::vector<cl::Platform> getAvailablePlatforms();
    void selectPlatform(cl::Platform p);
    void selectDefaultPlatform();

    std::vector<cl::Device> getAvailableDevices(cl_device_type device_type);
    void selectDevice(cl::Device d);
    void selectDefaultDevice();

    cl::Program createProgram(const char *file_name);
    cl::Kernel createKernel(const cl::Program &program, const char *kernel_name);

    void executeKernel(cl::Kernel &kernel, const std::vector<cl::Buffer> &buffers, const cl::NDRange &global_range, const cl::NDRange &local_range);
    void readBuffer(const cl::Buffer &buffer, void *ptr, size_t size);
    void writeBuffer(const cl::Buffer &buffer, const void *ptr, size_t size);

    cl::Context getContext() const { return context; }
    cl::Device getSelectedDevice() const { return selected_device; }
    cl::Platform getSelectedPlatform() const { return selected_platform; }
    cl::CommandQueue getCommandQueue() const { return command_queue; }
};
