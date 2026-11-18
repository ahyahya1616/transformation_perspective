#include "perspective.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

Mat applyPerspective(const Mat& src, const vector<Point2f>& corners)
{
    // Taille de sortie = max des longueurs opposées
    float W = max(dist(corners[0], corners[1]),   // haut
                  dist(corners[3], corners[2]));  // bas
    float H = max(dist(corners[0], corners[3]),   // gauche
                  dist(corners[1], corners[2]));  // droite

    vector<Point2f> dst = {
        {0.f,   0.f  },
        {W-1,   0.f  },
        {W-1,   H-1  },
        {0.f,   H-1  }
    };

    Mat M = getPerspectiveTransform(corners, dst);
    Mat warped;
    warpPerspective(src, warped, M, Size((int)W, (int)H));
    return warped;
}
