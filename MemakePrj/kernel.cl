typedef struct Point2f
{
    float x, y;
} Point2f;

typedef struct Ball
{
    int id;
    float r;
    struct Point2f pos;
    struct Point2f f;
} Ball;

typedef struct BorderLine
{
    struct Point2f p1;
    struct Point2f p2;
} BorderLine;

Point2f getBLXYMin(BorderLine bl)
{
    Point2f xyMin;
    xyMin.x = bl.p1.x < bl.p2.x ? bl.p1.x : bl.p2.x;
    xyMin.y = bl.p1.y < bl.p2.y ? bl.p1.y : bl.p2.y;
    return xyMin;
}

Point2f getBLXYMax(BorderLine bl)
{
    Point2f xyMax;
    xyMax.x = bl.p1.x > bl.p2.x ? bl.p1.x : bl.p2.x;
    xyMax.y = bl.p1.y > bl.p2.y ? bl.p1.y : bl.p2.y;
    return xyMax;
}

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

float2 getUnitVector(float2 v)
{
    float vectLen = getLen(v);
    return (float2)( v[0] / vectLen, v[1] / vectLen );
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

Ball opticColl(Ball b1, const Ball b2)
{
    // direction to other ball
    float2 p1 = (float2)(b1.pos.x, b1.pos.y);
    float2 p2 = (float2)(b2.pos.x, b2.pos.y);
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
    return b1;
}

Ball pulseColl(Ball b1, const Ball b2)
{
    // direction to other ball
    float2 p1 = (float2)(b1.pos.x, b1.pos.y);
    float2 p2 = (float2)(b2.pos.x, b2.pos.y);
    float2 toB2 = p2 - p1;
    float2 toB2Unit = getUnitVector(toB2);
    // calculate dot product
    float2 f1 = (float2)(b1.f.x, b1.f.y);
    float dotProdB1 = dot(f1, toB2Unit);
    // calculate speed v for both balls
    // that is projection to hit axis
    float v1 = dotProdB1;
    float2 f2 = (float2)(b2.f.x, b2.f.y);
    float dotProdB2 = dot(f2, toB2Unit);
    float v2 = dotProdB2;
    // if move projection to hit axis 
    // for both balls are directed to move away 
    // one from other then do nothing
    if (v1 <= 0 && v2 >= 0)
    {
        return b1;
    }
    // mass is equal to square of 2d ball
    float m1 = b1.r * b1.r;
    float m2 = b2.r * b2.r;
    // speed of this ball after collision
    float v1new = (2 * m2 * v2 + v1 * (m1 - m2)) / (m1 + m2);
    // "to" move component
    float2 tmpTo = toB2Unit * v1new;
    // "tangent" move component
    float2 tanUnit = rotVect(toB2Unit, M_PI_2_F);
    float dotProdTan1 = dot(f1, tanUnit);
    float2 tmpTan = tanUnit * dotProdTan1;
    // new move vector
    float2 newF = tmpTo + tmpTan;
    b1.f.x = newF[0];
    b1.f.y = newF[1];
    return b1;
}

Ball opticCollPoint(Ball b, float2 contactPoint)
{
    float2 pos = (float2)(b.pos.x, b.pos.y);
    // direction to other contact point
    float2 toContPoint = contactPoint - pos;
    // calculate dot product
    float2 f = (float2)(b.f.x, b.f.y);
    float dotProd = dot(f, toContPoint);
    // if dot product is negative then force directed away 
    // from contact point and we do nothing
    if (dotProd > 0)
    {
        // angle between normal and force (moving) vectors
        float angle = getAngleTo(f, toContPoint);
        // the angle of incidence is equal to the angle of reflection
        f = rotVect(f, angle * 2);
        f = f * (-1);
        b.f.x = f.x;
        b.f.y = f.y;
    }
    return b;
}

Ball checkCollision(Ball b1, const Ball b2)
{
    float2 p1 = (float2)(b1.pos.x, b1.pos.y);
    float2 p2 = (float2)(b2.pos.x, b2.pos.y);
    const float dist = getDistanceBetween(p1, p2);
    if (dist < b1.r + b2.r)
    {
        b1 = pulseColl(b1, b2);
    }
    return b1;
}

Ball checkCollisionBL(Ball b, BorderLine bl)
{
    // rectangle around line
    Point2f xyMin = getBLXYMin(bl);
    xyMin.x = xyMin.x - b.r;
    xyMin.y = xyMin.y - b.r;
    Point2f xyMax = getBLXYMax(bl);
    xyMax.x = xyMax.x + b.r;
    xyMax.y = xyMax.y + b.r;
    if (b.pos.x >= xyMin.x && b.pos.y >= xyMin.y && b.pos.x <= xyMax.x && b.pos.y <= xyMax.y)
    {
        float2 a = (float2)(b.pos.x, b.pos.y);
        // sides of triangle
        float2 blp1 = (float2)(bl.p1.x, bl.p1.y);
        float2 blp2 = (float2)(bl.p2.x, bl.p2.y);
        float p1a = getDistanceBetween(blp1, a);
        float p2a = getDistanceBetween(blp2, a);
        float p1p2 = getDistanceBetween(blp1, blp2);
        // angles
        float ang1 = acos((p1a * p1a + p1p2 * p1p2 - p2a * p2a) / (2 * p1a * p1p2));
        float ang2 = acos((p2a * p2a + p1p2 * p1p2 - p1a * p1a) / (2 * p2a * p1p2));
        // if both angles are sharp then calculate h (distanse from a to line)
        if (ang1 <= M_PI_2_F && ang2 <= M_PI_2_F)
        {
            float h = sin(ang1) * p1a;
            if (h <= b.r)
            {
                // calculate contact point
                float cath1 = cos(ang1) * p1a;
                float2 v = (blp2 - blp1);
                v = getUnitVector(v) * cath1;
                float2 contactPoint = blp1 + v;
                b = opticCollPoint(b, contactPoint);
            }
        }
        else if (p1a <= b.r) // first line end
        {
            b = opticCollPoint(b, blp1);
        }
        else if (p2a <= b.r) // second line end
        {
            b = opticCollPoint(b, blp2);
        }
    }
    return b;
}

Ball checkBorders(Ball b, const int scrW, const int scrH)
{
    // check borders
    if ((b.pos.x - b.r) <= 0 && b.f.x < 0)
    {
        b.f.x = fabs(b.f.x);
    }
    if ((b.pos.x + b.r) >= scrW && b.f.x > 0)
    {
        b.f.x = -fabs(b.f.x);
    }
    if ((b.pos.y - b.r) <= 0 && b.f.y < 0)
    {
        b.f.y = fabs(b.f.y);
    }
    if ((b.pos.y + b.r) >= scrH && b.f.y > 0)
    {
        b.f.y = -fabs(b.f.y);
    }
    return b;
}

__kernel void collideAndUpdate(
    __global Ball* b1,
    __global Ball* b2,
    const int ballCnt,
    __global BorderLine* bl,
    const int blCnt,
    const int scrW,
    const int scrH,
    const double frameTimeMs)
{
    // Get work-item identifiers.
    int i = get_global_id(0);
    Ball ball = b1[i];
    for (int j = 0; j < ballCnt; j++)
    {
        // check collision between one ball to others, but don't check collision to itself
        if (j != i)
        {
            ball = checkCollision(ball, b1[j]);
        }
    }

    for (int j = 0; j < blCnt; ++j)
    {
        ball = checkCollisionBL(ball, bl[j]);
    }

    ball = checkBorders(ball, scrW, scrH);

    // update positions
    ball.pos.x += ball.f.x * frameTimeMs;
    ball.pos.y += ball.f.y * frameTimeMs;

    b2[i] = ball;
}