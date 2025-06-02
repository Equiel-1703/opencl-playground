#include "../include/OCLInterface.hpp"

#include <stdexcept>

void OCLInterface::createContext()
{
    this->context = cl::Context(this->selected_device);
}

void OCLInterface::createCommandQueue()
{
    this->command_queue = cl::CommandQueue(this->context, this->selected_device);
}


std::string OCLInterface::getKernelCode(const char *file_name)
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

std::vector<cl::Platform> OCLInterface::getAvailablePlatforms()
{
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);

    return platforms;
}

void OCLInterface::selectPlatform(cl::Platform p)
{
    if (p() == nullptr)
    {
        throw std::runtime_error("Invalid OpenCL platform selected");
    }

    this->selected_platform = p;

    std::cout << "Selected OpenCL platform: " << this->selected_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
}

void OCLInterface::selectDefaultPlatform()
{
    std::vector<cl::Platform> platforms = this->getAvailablePlatforms();

    if (platforms.empty())
    {
        throw std::runtime_error("No OpenCL platforms found");
    }

    this->selectPlatform(platforms[0]);
}

std::vector<cl::Device> OCLInterface::getAvailableDevices(cl_device_type device_type = CL_DEVICE_TYPE_ALL)
{
    if (this->selected_platform() == nullptr)
    {
        throw std::runtime_error("No OpenCL platform selected");
    }

    std::vector<cl::Device> devices;
    this->selected_platform.getDevices(device_type, &devices);

    return devices;
}

void OCLInterface::selectDevice(cl::Device d)
{
    if (d() == nullptr)
    {
        throw std::runtime_error("Invalid OpenCL device selected");
    }

    this->selected_device = d;

    this->createContext();
    this->createCommandQueue();

    std::cout << "Selected OpenCL device: " << this->selected_device.getInfo<CL_DEVICE_NAME>() << std::endl;
}

void OCLInterface::selectDefaultDevice()
{
    std::vector<cl::Device> devices = this->getAvailableDevices();

    if (devices.empty())
    {
        throw std::runtime_error("No OpenCL devices found");
    }

    this->selectDevice(devices[0]);
}

cl::Program OCLInterface::createProgram(const char *file_name)
{
    std::string kernel_code = this->getKernelCode(file_name);
    cl::Program program(this->context, kernel_code, true);

    if (program() == nullptr)
    {
        std::cerr << "ERROR: Failed to create OpenCL program from file '" << file_name << "'." << std::endl;
        std::cerr << "OpenCL error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->selected_device) << std::endl;
        throw std::runtime_error("Failed to create OpenCL program");
    }

    std::cout << "OpenCL program created and builded successfully from file '" << file_name << "'." << std::endl;
    return program;
}

cl::Kernel OCLInterface::createKernel(const cl::Program &program, const char *kernel_name)
{
    cl::Kernel kernel(program, kernel_name);

    if (kernel() == nullptr)
    {
        std::cerr << "ERROR: Failed to create OpenCL kernel '" << kernel_name << "'." << std::endl;
        throw std::runtime_error("Failed to create OpenCL kernel");
    }

    std::cout << "OpenCL kernel '" << kernel_name << "' created successfully." << std::endl;
    return kernel;
}

void OCLInterface::executeKernel(cl::Kernel &kernel, const std::vector<cl::Buffer> &buffers, const cl::NDRange &global_range, const cl::NDRange &local_range)
{
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        kernel.setArg(static_cast<cl_uint>(i), buffers.at(i));
    }

    this->command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_range, local_range);
    this->command_queue.finish();

    std::cout << "OpenCL kernel executed successfully." << std::endl;
}

void OCLInterface::readBuffer(const cl::Buffer &buffer, void *ptr, size_t size)
{
    this->command_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, ptr);
}

void OCLInterface::writeBuffer(const cl::Buffer &buffer, const void *ptr, size_t size)
{
    this->command_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, ptr);
}
