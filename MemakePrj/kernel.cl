typedef struct Point2f
{
    float x, y;
} Point2f;

typedef struct Ball
{
    float r;
    struct Point2f pos;
    struct Point2f f;
} Ball;

float getDistanceBetween(float2 p1, float2 p2)
{
    float2 p1p2 = p2 - p1;
    float dist = native_sqrt(p1p2[0] * p1p2[0] + p1p2[1] * p1p2[1]);
    return dist;
}

float getLen(float2 p)
{
    float len = getDistanceBetween((float2)(0, 0), p);
    return len;
}

float getCrossProd(float2 p1, float2 p2)
{
    float crossProd = p1[0] * p2[1] - p1[1] * p2[0];
    return crossProd;
}

float getAngleTo(float2 p1, float2 p2) 
{
    return asin(getCrossProd(p1, p2) / (getLen(p1) * getLen(p2)));
}

float2 rotVect(float2 v, float angle)
{
    // first of all create rotation  matrix
    float c = cos(angle);
    float s = sin(angle);
    float2 mr0 = { c, -s }; // first row
    float2 mr1 = { s,  c }; // second row
    // get rotated vector by multiply matrix to vector
    float2 tmp;
    tmp[0] = (v[0] * mr0[0] + v[1] * mr0[1]);
    tmp[1] = (v[0] * mr1[0] + v[1] * mr1[1]);
    return tmp;
}

Ball checkCollision(Ball b1, const Ball b2)
{
    float2 p1 = (float2)(b1.pos.x, b1.pos.y);
    float2 p2 = (float2)(b2.pos.x, b2.pos.y);
    const float dist = getDistanceBetween(p1, p2);
    if (dist < b1.r + b2.r)
    {
        // direction to other ball
        float2 to2 = p2 - p1;
        float2 f = (float2)(b1.f.x, b1.f.y);
        // calculate dot product
        float dotProd = dot(f, to2);
        // if dot product is negative then force directed away from B ball
        // and we do nothing
        if (dotProd > 0)
        {
            // angle between normal and force (moving) vectors
            float angle = getAngleTo(f, to2);
            // the angle of incidence is equal to the angle of reflection
            f = rotVect(f, angle * 2);
            f = f * (-1);
            b1.f.x = f.x;
            b1.f.y = f.y;
        }
    }
    return b1;
}

__kernel void collideAndUpdate(
    __global Ball* b1,
    __global Ball* b2,
    const int ballCnt,
    const int scrW,
    const int scrH,
    const double frameTimeMs)
{
    // Get work-item identifiers.
    int i = get_global_id(0);
    Ball tmp = b1[i];
    for (int j = 0; j < ballCnt; j++)
    {
        // check collision between one ball to others, but don't check collision to itself
        if (j != i)
        {
            Ball tmp2 = b1[j];
            tmp = checkCollision(tmp, tmp2);
        }
    }

    // check borders
    if (tmp.pos.x <= 0 || tmp.pos.x >= scrW)
    {
        tmp.f.x *= -1;
    }
    if (tmp.pos.y <= 0 || tmp.pos.y >= scrH)
    {
        tmp.f.y *= -1;
    }

    // update positions
    tmp.pos.x += tmp.f.x * frameTimeMs;
    tmp.pos.y += tmp.f.y * frameTimeMs;

    b2[i] = tmp;
}