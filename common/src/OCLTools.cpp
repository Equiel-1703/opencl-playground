#include "../include/OCLTools.hpp"

std::string OCLTools::getKernelCode(const char *file_name)
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