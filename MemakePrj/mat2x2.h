#pragma once
#include <cmath>

static const float k_PI = 3.1415926536f;

struct Point2f
{
	float x, y;

	Point2f operator+(const Point2f& p) const
	{
		return Point2f({ x + p.x, y + p.y });
	}

	Point2f operator-(const Point2f& p) const
	{
		return Point2f({x - p.x, y - p.y});
	}

	Point2f& operator+=(const Point2f& p)
	{
		x += p.x;
		y += p.y;
		return *this;
	}

	Point2f operator*(const float& k) const
	{
		Point2f p = *this;
		p.x *= k;
		p.y *= k;
		return p;
	}

	float distanceTo(const Point2f& p) const
	{
		Point2f v = p - *this;
		const float dist = sqrt(v.x*v.x + v.y*v.y);
		return dist;
	}

	float lenght() const
	{
		Point2f p = {0.f, 0.f};
		return p.distanceTo(*this);
	}

	float dotProduct(const Point2f& v) const
	{
		return x*v.x + y*v.y;
	}

	float crossProduct(const Point2f& v) const
	{
		return x*v.y - y*v.x;
	}

	float angleTo(const Point2f& v) const
	{
		return asinf( crossProduct(v) / (lenght() * v.lenght()) );
	}

	Point2f unitVector() const
	{
		float len = lenght();
		return {x / len, y / len};
	}
};

struct Mat2x2f
{
	Point2f values[2];

	Mat2x2f& rot(const float angle)
	{
		const float c = cosf(angle);
		const float s = sinf(angle);
		values[0] = { c, -s };
		values[1] = { s,  c };
		return *this;
	}

	Point2f operator*(const Point2f& p) const
	{
		Point2f res =
		{
			p.x * values[0].x + p.y * values[0].y,
			p.x * values[1].x + p.y * values[1].y
		};
		return res;
	}
};