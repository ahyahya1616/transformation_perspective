#include "utils.hpp"
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

vector<Point2f> sortCorners(vector<Point2f> pts)
{
    Point2f center(0, 0);
    for (auto& p : pts) center += p;
    center *= (1.0f / pts.size());

    vector<Point2f> tl, tr, br, bl;
    for (auto& p : pts) {
        if      (p.x < center.x && p.y < center.y) tl.push_back(p);
        else if (p.x > center.x && p.y < center.y) tr.push_back(p);
        else if (p.x > center.x && p.y > center.y) br.push_back(p);
        else                                         bl.push_back(p);
    }

    auto best = [](vector<Point2f>& v, Point2f ref) -> Point2f {
        if (v.empty()) return ref;
        return *min_element(v.begin(), v.end(), [&](const Point2f& a, const Point2f& b){
            return norm(a - ref) < norm(b - ref);
        });
    };

    return { best(tl, {0,0}),
             best(tr, {1e9f, 0}),
             best(br, {1e9f, 1e9f}),
             best(bl, {0,    1e9f}) };
}

float dist(Point2f a, Point2f b)
{
    return (float)norm(a - b);
}

bool lineIntersect(Point2f p1, Point2f p2,
                   Point2f p3, Point2f p4,
                   Point2f& out)
{
    float d = (p1.x-p2.x)*(p3.y-p4.y) - (p1.y-p2.y)*(p3.x-p4.x);
    if (fabsf(d) < 1e-6f) return false;
    float t = ((p1.x-p3.x)*(p3.y-p4.y) - (p1.y-p3.y)*(p3.x-p4.x)) / d;
    out.x = p1.x + t*(p2.x-p1.x);
    out.y = p1.y + t*(p2.y-p1.y);
    return true;
}

