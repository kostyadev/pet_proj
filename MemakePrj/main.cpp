#include "Memake/Memake.h"
#include <CL/cl.hpp>
#include <math.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include "mat2x2.h"

using namespace std;

Memake mmk(1024, 900, "memake");

cl::Program program;  // The program that will run on the device.    
cl::Context context;                // The context which holds the device.    
cl::Device device;                  // The device where the kernel will run.

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
    auto err = program.build(ss.str().c_str());
    if (err != CL_BUILD_SUCCESS)
    {
        std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
            << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }
}


struct Ball 
{
public:
    float r;
    Point2f pos;
    Point2f f = { 0.f, 0.01f };

    Ball(float _x = 0.f, float _y = 0.f, float _r = 0.f, float angle = 0.f) 
    {
        pos.x = _x;
        pos.y = _y;
        r = _r;
        f = Mat2x2f().rot(k_PI * 2.f * (angle / 12.f)) * f;
    }

    void update(double frameTimeMs) 
    {
        move(frameTimeMs);
    }

    void draw() 
    {
        mmk.drawCircle(pos.x, pos.y, r, Colmake.beige);
    }

    void checkCollision(const Ball& b) 
    {
        float dist = pos.distanceTo(b.pos);
        if (dist < r + b.r) 
        {
            // direction to other ball
            Point2f toB = b.pos - pos;
            // calculate dot product
            float dotProd = f.dotProduct(toB);
            // if dot product is negative then force directed away from B ball
            // and we do nothing
            if (dotProd > 0) 
            {
                // angle between normal and force (moving) vectors
                float angle = f.angleTo(toB);
                // the angle of incidence is equal to the angle of reflection
                f = Mat2x2f().rot(angle * 2) * f;
                f = f * (-1.f);
            }
        }
    }

    void move(double frameTimeMs)
    {
        pos.x += f.x * frameTimeMs;
        pos.y += f.y * frameTimeMs;
    }

    void checkBorders() 
    {
        if (pos.x <= 0 || pos.x >= mmk.getScreenW()) 
        {
            f.x *= -1;
        }
        if (pos.y <= 0 || pos.y >= mmk.getScreenH()) 
        {
            f.y *= -1;
        }
    }

};

void colladeAndUpdateCPU(Ball* b, Ball* tmpB, const int numOfBall, const double frameTimeMs)
{
    for (int i = 0; i < numOfBall; i++)
    {
        Ball tmpBall = b[i];
        for (int j = 0; j < numOfBall; j++)
        {
            // check collision between one ball to others, but don't check collision to itself
            if (j != i)
            {
                tmpBall.checkCollision(b[j]);
            }
        }
        tmpBall.checkBorders();
        tmpBall.update(frameTimeMs);  // update/move every ball
        tmpB[i] = tmpBall;
    }
}

void colladeAndUpdateGPU(Ball* b, Ball* tmpB, const int numOfBall, const double frameTimeMs)
{
    cl::Buffer inB(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, numOfBall * sizeof(Ball), (void*)b);
    cl::Buffer outB(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, numOfBall * sizeof(Ball), (void*)tmpB);

    cl::Kernel kern(program, "collideAndUpdate");
    kern.setArg(0, inB);
    kern.setArg(1, outB);
    kern.setArg(2, numOfBall);
    kern.setArg(3, mmk.getScreenW());
    kern.setArg(4, mmk.getScreenH());
    kern.setArg(5, frameTimeMs);

    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(numOfBall, 1));
    queue.enqueueReadBuffer(outB, CL_TRUE, 0, numOfBall * sizeof(Ball), tmpB);
}

int main()
{
    const int numOfBall = 20000;
    vector<Ball> b1;
    b1.reserve(numOfBall);
    for (int i = 0; i < numOfBall; i++)
    {
        b1.push_back(Ball(random(0, mmk.getScreenW()), random(0, mmk.getScreenH()), 1, random_f(0.f, k_PI * 2.f)));
    }

    // Initialize OpenCL device.
    initializeDevice();

    long long frameCnt = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    auto curTime = t_start;
    double frameTimeMs = 0;
    vector<Ball> b2(numOfBall);
    Ball* b = &b1[0];
    Ball* tmpB = &b2[0];
    mmk.update( [&]() 
    {
        colladeAndUpdateGPU(b, tmpB, numOfBall, frameTimeMs);

        for (int i = 0; i < numOfBall; ++i)
        {
            tmpB[i].draw();
        }

        swap(b, tmpB);
        
        auto oldTime = curTime;
        curTime = std::chrono::high_resolution_clock::now();
        frameTimeMs = std::chrono::duration<double, std::milli>(curTime - oldTime).count();

        frameCnt++;
        //mmk.delay(2);
    });

    auto t_end = std::chrono::high_resolution_clock::now();
    auto timeMs = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    auto fps = frameCnt * 1000 / timeMs;
    std::cout << "fps: " << fps << std::endl;

    return 0;
}