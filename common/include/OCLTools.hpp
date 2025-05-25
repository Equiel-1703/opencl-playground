#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200 // OpenCL 2.0
#include <CL/cl.hpp>

#include <iostream>
#include <fstream>

class OCLTools
{
private:
    /* data */
public:
    std::string getKernelCode(const char *file_name);
};

