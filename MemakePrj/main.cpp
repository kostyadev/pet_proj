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

struct BorderLine
{
public:
    Point2f p1, p2;

    void draw() const
    {
        mmk.drawLine(p1.x, p1.y, p2.x, p2.y, Colmake.white);
    }

    Point2f getXYMin() const
    {
        return { p1.x < p2.x ? p1.x : p2.x, p1.y < p2.y ? p1.y : p2.y };
    }

    Point2f getXYMax() const
    {
        return { p1.x > p2.x ? p1.x : p2.x, p1.y > p2.y ? p1.y : p2.y };
    }
};

struct Ball 
{
public:
    int id; // needs for debug purposes
    float r;
    Point2f pos;
    Point2f f = { 0.f, 0.05f };

    Ball(float _x = 0.f, float _y = 0.f, float _r = 0.f, float angle = 0.f, float _id = 0) 
    {
        pos.x = _x;
        pos.y = _y;
        r = _r;
        f = Mat2x2f().rot(angle) * f;
        id = _id;
    }

    void update(double frameTimeMs) 
    {
        move(frameTimeMs);
    }

    void draw() 
    {
        mmk.drawCircle(pos.x, pos.y, r, Colmake.beige);
    }

    void pulseColl(const Ball& b2)
    {
        // direction to other ball
        Point2f toB2 = b2.pos - pos;
        Point2f toB2Unit = toB2.unitVector();
        // calculate dot product
        float dotProdB = f.dotProduct(toB2Unit);
        // calculate speed v for both balls
        // that is projection to hit axis
        float v1 = dotProdB;
        float dotProdB2 = b2.f.dotProduct(toB2Unit);
        float v2 = dotProdB2;
        // if move projection to hit axis 
        // for both balls are directed to move away 
        // one from other then do nothing
        if (v1 <= 0 && v2 >= 0)
        {
            return;
        }
        // mass is equal to square of 2d ball
        float m1 = r * r;
        float m2 = b2.r * b2.r;
        // speed of this ball after collision 
        float v1new = (2 * m2 * v2 + v1 * (m1 - m2)) / (m1 + m2);
        // "to" move component
        Point2f tmpTo = toB2Unit * v1new;
        // "tangent" move component
        Point2f tanUnit = Mat2x2f().rot(k_PI / 2) * toB2Unit;
        float dotProdTan1 = f.dotProduct(tanUnit);
        Point2f tmpTan = tanUnit * dotProdTan1;
        // new move vector
        Point2f newF = tmpTo + tmpTan;
        f = newF;
    }

    void opticColl(const Ball& b)
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

    void checkCollision(const Ball& b) 
    {
        float dist = pos.distanceTo(b.pos);
        if (dist < r + b.r) 
        {
            pulseColl(b);
        }
    }

    void opticCollPoint(const Point2f contactPoint)
    {
        // direction to other contact point
        Point2f toContPoint = contactPoint - pos;
        // calculate dot product
        float dotProd = f.dotProduct(toContPoint);
        // if dot product is negative then force directed away 
        // from contact point and we do nothing
        if (dotProd > 0)
        {
            // angle between normal and force (moving) vectors
            float angle = f.angleTo(toContPoint);
            // the angle of incidence is equal to the angle of reflection
            f = Mat2x2f().rot(angle * 2) * f;
            f = f * (-1.f);
        }
    }

    void checkCollision(const BorderLine& bl)
    {
        // rectangle around line
        Point2f xyMin = bl.getXYMin();
        xyMin = {xyMin.x - r, xyMin.y - r};
        Point2f xyMax = bl.getXYMax();
        xyMax = { xyMax.x + r, xyMax.y + r };
        if (pos.x >= xyMin.x && pos.y >= xyMin.y && pos.x <= xyMax.x && pos.y <= xyMax.y)
        {
            Point2f a = pos;
            // sides of triangle
            float p1a = bl.p1.distanceTo(a);
            float p2a = bl.p2.distanceTo(a);
            float p1p2 = bl.p1.distanceTo(bl.p2);
            // angles
            float ang1 = acos((p1a * p1a + p1p2 * p1p2 - p2a * p2a) / (2 * p1a * p1p2));
            float ang2 = acos((p2a * p2a + p1p2 * p1p2 - p1a * p1a) / (2 * p2a * p1p2));
            // if both angles are sharp then calculate h (distanse from a to line)
            if (ang1 <= k_PI / 2 && ang2 <= k_PI / 2)
            {
                float h = sin(ang1) * p1a;
                if (h <= r)
                {
                    // calculate contact point
                    float cath1 = cos(ang1) * p1a;
                    Point2f v = (bl.p2 - bl.p1).unitVector() * cath1;
                    Point2f contactPoint = bl.p1 + v;
                    opticCollPoint(contactPoint);
                }
            }
            else if (p1a <= r) // first line end
            {
                opticCollPoint(bl.p1);
            }
            else if (p2a <= r) // second line end
            {
                opticCollPoint(bl.p2);
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
        if ((pos.x - r) <= 0 && f.x < 0)
        {
            f.x = abs(f.x);
        }
        if ((pos.x + r) >= mmk.getScreenW() && f.x > 0)
        {
            f.x = -abs(f.x);
        }
        if ((pos.y - r) <= 0 && f.y < 0) 
        {
            f.y = abs(f.y);
        }
        if ((pos.y + r) >= mmk.getScreenH() && f.y > 0)
        {
            f.y = -abs(f.y);
        }
    }

};

void colladeAndUpdateCPU(Ball* b, Ball* tmpB, const int numOfBall, BorderLine* bordLine, const int bordLineCnt, const double frameTimeMs)
{
    for (int i = 0; i < numOfBall; i++)
    {
        Ball ball = b[i];
        for (int j = 0; j < numOfBall; j++)
        {
            // check collision between one ball to others, but don't check collision to itself
            if (j != i)
            {
                ball.checkCollision(b[j]);
            }
        }

        for (int j = 0; j < bordLineCnt; ++j)
        {
            ball.checkCollision(bordLine[j]);
        }

        ball.checkBorders();
        ball.update(frameTimeMs);  // update/move every ball
        tmpB[i] = ball;
    }
}

void colladeAndUpdateGPU(Ball* b, Ball* tmpB, const int numOfBall, BorderLine* bl, int blCnt, const double frameTimeMs)
{
    cl::Buffer inB(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, numOfBall * sizeof(Ball), (void*)b);
    cl::Buffer outB(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, numOfBall * sizeof(Ball), (void*)tmpB);
    cl::Buffer inBl(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, blCnt * sizeof(BorderLine), (void*)bl);

    cl::Kernel kern(program, "collideAndUpdate");
    kern.setArg(0, inB);
    kern.setArg(1, outB);
    kern.setArg(2, numOfBall);
    kern.setArg(3, inBl);
    kern.setArg(4, blCnt);
    kern.setArg(5, mmk.getScreenW());
    kern.setArg(6, mmk.getScreenH());
    kern.setArg(7, frameTimeMs);

    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(numOfBall, 1));
    queue.enqueueReadBuffer(outB, CL_TRUE, 0, numOfBall * sizeof(Ball), tmpB);
}

int main()
{
    const int numOfBall = 4000;
    vector<Ball> b1;
    b1.reserve(numOfBall);
    int sW = mmk.getScreenW();
    int sH = mmk.getScreenH();
    float r = 3;
    float d = 2 * r;
    int col = 0;
    int row = 0;
    for (int i = 0; i < numOfBall; i++)
    {
        int x = col * 1.5 * d + 0.75 * d;
        int y = row * 1.5 * d + 0.75 * d;
        b1.push_back(Ball(x, y, r, random_f(0.f, k_PI * 2.f), i));
        if ((x + 1.5 * d) > sW)
        {
            col = 0;
            row++;
        }
        else
        {
            col++;
        }
    }

    std::vector<BorderLine> lines;
    lines.push_back({ {0, 400}, {300, 500} });
    lines.push_back({ {300, 500}, {400, 600} });
    lines.push_back({ {400, 600}, {450, 700} });
    lines.push_back({ {1023, 400}, {723, 500} });
    lines.push_back({ {723, 500}, {623, 600} });
    lines.push_back({ {623, 600}, {573, 700} });

    // Initialize OpenCL device.
    initializeDevice();

    long long frameCnt = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    auto curTime = t_start;
    double frameTimeMs = 0;
    vector<Ball> b2(numOfBall);
    Ball* b = &b1[0];
    Ball* tmpB = &b2[0];
    BorderLine* bLine = &lines[0];
    mmk.update( [&]() 
    {
        colladeAndUpdateGPU(b, tmpB, numOfBall, bLine, lines.size(), frameTimeMs);

        for (int i = 0; i < numOfBall; ++i)
        {
            tmpB[i].draw();
        }

        for (int i = 0; i < lines.size(); ++i)
        {
            bLine[i].draw();
        }

        swap(b, tmpB);
        
        auto oldTime = curTime;
        curTime = std::chrono::high_resolution_clock::now();
        frameTimeMs = std::chrono::duration<double, std::milli>(curTime - oldTime).count();

        frameCnt++;
        //mmk.delay(100);
    });

    auto t_end = std::chrono::high_resolution_clock::now();
    auto timeMs = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    auto fps = frameCnt * 1000 / timeMs;
    std::cout << "fps: " << fps << std::endl;

    return 0;
}