#include "Memake/Memake.h"
#include <math.h>
#include <chrono>
#include <vector>
#include "mat2x2.h"

using namespace std;

Memake mmk(800, 800, "memake");

class Ball 
{
public:
    float r;
    Point2f pos;
    Point2f f = { 0.f, 0.1f };

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

    void checkCollision(Ball& b) 
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

int main()
{
    const int numOfBall = 25;
    vector<Ball> b;
    b.reserve(numOfBall);
    for (int i = 0; i < numOfBall; i++)
    {
        b.push_back(Ball(random(0, 800), random(0, 800), 5, random_f(0.f, k_PI * 2.f)));
    }

    long long frameCnt = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    auto curTime = std::chrono::high_resolution_clock::now();
    double frameTimeMs = 0;
    mmk.update( [&]() 
    {
        for (int i = 0; i < numOfBall; i++) 
        {
            for (int j = 0; j < numOfBall; j++) 
            {
                // check collision between one ball to others, but don't check collision to itself
                if (j != i) 
                {
                    b[i].checkCollision(b[j]);
                }
            }
            b[i].checkBorders();
            b[i].update(frameTimeMs);  // update/move every ball
            b[i].draw();
        }
        
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