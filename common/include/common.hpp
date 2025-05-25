#pragma once

#include <CL/cl.h>
#include <iostream>

#define CHECK_ERROR(err_id) check_error(err_id, __LINE__, __FILE__)

static void check_error(cl_int err, int line, const char *file)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "ERROR (" << file << " at line" << line << "): " << err << std::endl;
    }
}