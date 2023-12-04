// OpenCL-filter-img.cpp : Defines the entry point for the application.
//
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include "BMP.h"

cl::Program program;  // The program that will run on the device.    
cl::Context context;                // The context which holds the device.    
cl::Device device;                  // The device where the kernel will run.

// Create a low-pass filter mask.
const int lpMaskSize = 5;
float lpMask[lpMaskSize][lpMaskSize] = 
{
    {.04,.04,.04,.04,.04},
    {.04,.04,.04,.04,.04},
    {.04,.04,.04,.04,.04},
    {.04,.04,.04,.04,.04},
    {.04,.04,.04,.04,.04},
};

// Create a high-pass filter mask.
const int hpMaskSize = 5;
float hpMask[hpMaskSize][hpMaskSize] = 
{
    {-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1},
    {-1,-1,24,-1,-1},
    {-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1},
};

const int SubSize = 16;

// Return a device found in this OpenCL platform.
cl::Device getDefaultDevice() {

    // Search for all the OpenCL platforms available and check
    // if there are any.
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No platforms found!" << std::endl;
        exit(1);
    }

    // Search for all the devices on the first platform
    // and check if there are any available.
    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()) {
        std::cerr << "No devices found!" << std::endl;
        exit(1);
    }

    // Return the first device found.
    return devices.front();
}

// Inicialize device and compile kernel code.
void initializeDevice()
{
    // Select the first available device.
    device = getDefaultDevice();

    // Read OpenCL kernel file as a string.
    context = cl::Context(device);
    std::ifstream kernel_file("../../../kernel.cl");
    std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

    // Compile kernel program which will run on the device.
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
    program = cl::Program(context, sources);
    std::stringstream ss;
    ss << "-D SUB_SIZE=" << SubSize; // defines for kernel function
    auto err = program.build(ss.str().c_str());
    if (err != CL_BUILD_SUCCESS)
    {
        std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
            << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }
}

void FilterImageGPU(const BMP& bmpIn, BMP& bmpOut)
{
    const auto imgWidth = bmpIn.bmp_info_header.width;
    const auto imgHeight = bmpIn.bmp_info_header.height;
    const uint32_t bytesPP = bmpIn.bmp_info_header.bit_count / 8;
    cl::Buffer inImg(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgWidth * imgHeight * bytesPP, (void*) bmpIn.data.data());
    cl::Buffer grayImg(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, imgWidth * imgHeight * bytesPP);
    cl::Buffer lpfImg(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, imgWidth * imgHeight * bytesPP);
    cl::Buffer lpMaskBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, lpMaskSize * lpMaskSize * sizeof(lpMask[0]), lpMask);
    cl::Buffer hpfImg(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, imgWidth * imgHeight * bytesPP);
    cl::Buffer hpMaskBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, hpMaskSize * hpMaskSize * sizeof(hpMask[0]), hpMask);

    cl::Kernel grayKernel(program, "rgbToGray");
    grayKernel.setArg(0, inImg);
    grayKernel.setArg(1, grayImg);
    grayKernel.setArg(2, bytesPP);

    cl::Kernel lpfKernel(program, "filterImageCached");
    lpfKernel.setArg(0, grayImg);
    lpfKernel.setArg(1, lpfImg);
    lpfKernel.setArg(2, bytesPP);
    lpfKernel.setArg(3, lpMaskSize);
    lpfKernel.setArg(4, lpMaskBuf);

    cl::Kernel hpfKernel(program, "filterImageCached");
    hpfKernel.setArg(0, lpfImg);
    hpfKernel.setArg(1, hpfImg);
    hpfKernel.setArg(2, bytesPP);
    hpfKernel.setArg(3, hpMaskSize);
    hpfKernel.setArg(4, hpMaskBuf);

    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(grayKernel, cl::NullRange, cl::NDRange(imgWidth, imgHeight));
    queue.enqueueNDRangeKernel(lpfKernel, cl::NullRange, cl::NDRange(imgWidth, imgHeight), cl::NDRange(SubSize, SubSize));
    queue.enqueueNDRangeKernel(hpfKernel, cl::NullRange, cl::NDRange(imgWidth, imgHeight), cl::NDRange(SubSize, SubSize));
    queue.enqueueReadBuffer(hpfImg, CL_TRUE, 0, bmpOut.data.size() * sizeof(bmpOut.data[0]), &bmpOut.data[0]);
}

int main()
{
    BMP bmp("../../../Shapes.bmp");
    BMP filteredBmp(bmp.bmp_info_header.width, bmp.bmp_info_header.height, bmp.bmp_info_header.bit_count == 32);

    // Initialize OpenCL device.
    initializeDevice();

    FilterImageGPU(bmp, filteredBmp);
    filteredBmp.write("../../../Shapes_filt.bmp");

	std::cout << "Hello OpenCL." << std::endl;
	return 0;
}
