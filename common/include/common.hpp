#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200 // OpenCL 2.0

#include <CL/opencl.hpp>
#include <iostream>

#define CHECK_ERROR(err_id) check_error(err_id, __LINE__, __FILE__)

static void check_error(cl_int err, int line, const char *file)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "ERROR (" << file << " at line " << line << "): " << err << std::endl;
    }
}