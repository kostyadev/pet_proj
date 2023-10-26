#define _USE_MATH_DEFINES

#include <thread>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>

const size_t vectSize = 300 * 1000 * 1000;

float RandomF(float minf, float maxf)
{
	return ((float(rand()) / float(RAND_MAX)) * (minf - maxf)) + minf;
}

struct Point3f
{
	float x;
	float y;
	float z;
};

struct Mat3x3f
{
	Point3f values[3];

	Mat3x3f& rot(const float angleX, const float angleY, const float angleZ)
	{
		values[0] = { cosf(angleY)*cosf(angleZ), sinf(angleX)*sinf(angleY)*cosf(angleZ) - cosf(angleX) * sinf(angleZ), cosf(angleX)*sinf(angleY)*cosf(angleZ) + sinf(angleX)*sinf(angleZ) };
		values[1] = { cosf(angleY)*sinf(angleZ), sinf(angleX)*sinf(angleY)*sinf(angleZ) + cosf(angleX) * sinf(angleZ), cosf(angleX)*sinf(angleY)*sinf(angleZ) - sinf(angleX)*cosf(angleZ) };
		values[2] = { -sinf(angleY), sinf(angleX)*cosf(angleY), cosf(angleX)*cosf(angleY)};
		return *this;
	}

	Point3f operator*(const Point3f& p) const
	{
		return 
		{ 
			p.x*values[0].x + p.y*values[0].y + p.z*values[0].z, 
			p.x*values[1].x + p.y*values[1].y + p.z*values[1].z,
			p.x*values[2].x + p.y*values[2].y + p.z*values[2].z
		};
	}
};

typedef std::function<void(Point3f& p)> PointFunc;

void applyToPoints(std::vector<Point3f>& points, const int elemCnt, PointFunc func, const unsigned int thrCnt = std::thread::hardware_concurrency())
{
	auto start = std::chrono::steady_clock::now();
	std::vector<std::thread> workers;
	for (int thrId = 0; thrId < thrCnt; ++thrId)
	{
		workers.push_back(std::thread([thrId, thrCnt, elemCnt, &points, &func]()
			{
				for (int i = thrId * elemCnt; i < vectSize; i += thrCnt * elemCnt)
				{
					int blockEndIdx = std::min(i + elemCnt, (int)vectSize);
					for (int j = i; j < blockEndIdx; ++j)
					{
						func(points[j]);
					}
				}
			}));
	}

	for (std::thread& t : workers)
	{
		t.join();
	}
	workers.clear();

	auto end = std::chrono::steady_clock::now();
	const std::chrono::duration<double> elemByThreadTime = end - start;
	std::cout << "threads: " << std::setw(2) << thrCnt << ", block size: " << std::setw(10) << sizeof(points.back()) * elemCnt << " bytes, time: " << elemByThreadTime.count() << std::endl;
}

void applyToPoints2ndWay(std::vector<Point3f>& points, PointFunc func, const unsigned int thrCnt = std::thread::hardware_concurrency())
{
	auto start = std::chrono::steady_clock::now();
	std::vector<std::thread> workers;
	size_t rod = points.size() % thrCnt;
	auto thrItemCnt = (points.size() - rod) / thrCnt;
	for (int thrId = 0; thrId < thrCnt; ++thrId)
	{
		workers.push_back(std::thread([thrId, thrCnt, thrItemCnt, &points, &func]()
			{
				auto begIdx = thrId * thrItemCnt;
				bool isLastThread = (thrId == thrCnt - 1);
				auto endIdx = isLastThread ? points.size() : begIdx + thrItemCnt;
				for (auto i = begIdx; i < endIdx; ++i)
				{
					func(points[i]);
				}
			}));
	}

	for (std::thread& t : workers)
	{
		t.join();
	}
	workers.clear();

	auto end = std::chrono::steady_clock::now();
	const std::chrono::duration<double> elemByThreadTime = end - start;
	std::cout << "threads: " << std::setw(2) << thrCnt << ", block size: " << std::setw(10) << sizeof(points.back()) * thrItemCnt << " bytes, time: " << elemByThreadTime.count() << std::endl;
}

int main()
{
	std::cout << "-- create array.. " << std::endl;
	std::vector<Point3f> points(vectSize);
	std::cout << "-- create array done! " << std::endl;
	std::cout << "-- init array: " << std::endl;
	PointFunc initFunc = [](Point3f& p) { p = { RandomF(0.f, 5.f), RandomF(0.f, 5.f), RandomF(0.f, 5.f) }; };
	applyToPoints2ndWay(points, initFunc);
	applyToPoints2ndWay(points, initFunc, 8);
	std::cout << "-- init array done! " << std::endl;

	auto rotMat = Mat3x3f().rot(M_PI / 4.f, M_PI / 4.f, M_PI / 2.f);
	PointFunc matFunc = [&rotMat](Point3f& p) { p = rotMat * p; };

	std::cout << "-- 1st way: many small blocks: " << std::endl;
	applyToPoints(points, 1, matFunc, 1);
	applyToPoints(points, 1, matFunc);
	applyToPoints(points, 8, matFunc);
	applyToPoints(points, 128, matFunc);
	applyToPoints(points, 1024, matFunc);
	applyToPoints(points, 10*1024, matFunc);

	std::cout << "-- 2st way: the biggest blocks: " << std::endl;
	applyToPoints2ndWay(points, matFunc);
	applyToPoints2ndWay(points, matFunc, 8);
	applyToPoints2ndWay(points, matFunc, 4);
	applyToPoints2ndWay(points, matFunc, 2);
}
