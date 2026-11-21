#include "detection.hpp"
#include "utils.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// ─────────────────────────────────────────────────────────────
//  Prétraitement : CLAHE pour améliorer le contraste local
// ─────────────────────────────────────────────────────────────
static Mat preprocess(const Mat& img)
{
    Mat gray, out;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(gray, out);
    return out;
}

// ─────────────────────────────────────────────────────────────
//  Détection automatique du type de fond
// ─────────────────────────────────────────────────────────────
bool isTexturedBackground(const Mat& img)
{
    Mat gray, lap;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);
    Scalar mean, stddev;
    meanStdDev(lap, mean, stddev);
    double variance = stddev[0] * stddev[0];
    cout << "  Variance Laplacien = " << (int)variance << "\n";
    return variance > 300.0;
}

// ─────────────────────────────────────────────────────────────
//  Score d'un quadrilatère
// ─────────────────────────────────────────────────────────────
static bool isCameraFrame(const vector<Point>& quad, int w, int h)
{
    int touches = 0;
    for (auto& p : quad) {
        if ((p.x <= 5 || p.x >= w - 5) && (p.y <= 5 || p.y >= h - 5)) {
            touches++;
        }
    }
    return touches >= 3;
}

static double scoreQuad(const vector<Point>& quad, const Mat& img)
{
    if (quad.size() != 4) return 0;

    double area    = contourArea(quad);
    double imgArea = (double)img.rows * img.cols;
    double areaRatio = area / imgArea;

    if (areaRatio < 0.05 || areaRatio > 0.95) return 0;
    if (isCameraFrame(quad, img.cols, img.rows)) return 0;

    double angleScore = 0;
    for (int i = 0; i < 4; i++) {
        Point2f a = quad[(i+3)%4], b = quad[i], c = quad[(i+1)%4];
        Point2f v1 = a-b, v2 = c-b;
        double cosA = v1.dot(v2) / (norm(v1)*norm(v2) + 1e-6);
        angleScore += (1.0 - fabs(cosA));
    }
    angleScore /= 4.0;

    Rect br = boundingRect(quad);
    double fillScore = area / (br.area() + 1e-6);

    return areaRatio * 0.40 + angleScore * 0.40 + fillScore * 0.20;
}

static vector<Point2f> bestQuadFromBinary(const Mat& binary, const Mat& img)
{
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b){
        return contourArea(a) > contourArea(b);
    });
    if (contours.size() > 20) contours.resize(20);

    double bestScore = 0;
    vector<Point2f> bestCorners;

    for (auto& c : contours) {
        for (double eps : {0.01, 0.02, 0.03, 0.04, 0.05}) {
            vector<Point> approx;
            approxPolyDP(c, approx, eps * arcLength(c, true), true);
            if (approx.size() == 4 && isContourConvex(approx)) {
                double score = scoreQuad(approx, img);
                if (score > bestScore) {
                    bestScore = score;
                    vector<Point2f> pts;
                    for (auto& p : approx) pts.push_back(Point2f(p));
                    bestCorners = sortCorners(pts);
                }
            }
        }
    }
    return (bestScore > 0.08) ? bestCorners : vector<Point2f>{};
}

// ─────────────────────────────────────────────────────────────
//  MÉTHODE 1 : Binarisation Otsu + Canny (Fond Simple)
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectSimpleBackground(const Mat& img)
{
    Mat gray = preprocess(img);
    Mat blur, thresh, edges;
    GaussianBlur(gray, blur, Size(5, 5), 0);

    threshold(blur, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
    Mat k7 = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(thresh, thresh, MORPH_CLOSE, k7);
    auto resOtsu = bestQuadFromBinary(thresh, img);
    if (!resOtsu.empty()) return resOtsu;

    Mat sorted; blur.reshape(1,1).copyTo(sorted);
    cv::sort(sorted, sorted, SORT_EVERY_ROW);
    double med = sorted.at<uchar>(0, sorted.cols/2);
    Canny(blur, edges, max(0.0, 0.5*med), min(255.0, 1.5*med));
    morphologyEx(edges, edges, MORPH_CLOSE, k7);
    auto resCanny = bestQuadFromBinary(edges, img);
    if (!resCanny.empty()) return resCanny;

    return {};
}

// ─────────────────────────────────────────────────────────────
//  MÉTHODE 2 : Analyse locale & Lignes de Hough (Fond Texturé)
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectTexturedBackground(const Mat& img)
{
    Mat gray, blur;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(gray, gray);
    bilateralFilter(gray, blur, 11, 90, 90);

    Mat edges;
    Canny(blur, edges, 20, 80);
    Mat k5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(edges, edges, MORPH_CLOSE, k5);

    int minDim = min(img.cols, img.rows);
    struct HP { int vote, len, gap; };
    for (auto& p : vector<HP>{{80,minDim/4,20},{60,minDim/6,30},{40,minDim/8,40},{30,minDim/10,50}}) {
        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI/180, p.vote, p.len, p.gap);
        if (lines.size() < 4) continue;

        vector<Vec4i> horiz, vert;
        for (auto& l : lines) {
            double a = atan2(abs(l[3]-l[1]), abs(l[2]-l[0])) * 180.0 / CV_PI;
            if (a < 25)      horiz.push_back(l);
            else if (a > 65) vert.push_back(l);
        }
        if (horiz.size() < 2 || vert.size() < 2) continue;

        // Fusion des lignes proches
        auto merge = [](vector<Vec4i>& v, bool h) {
            vector<pair<int,Vec4i>> g;
            for (auto& l : v) {
                int pos = h ? (l[1]+l[3])/2 : (l[0]+l[2])/2;
                bool done = false;
                for (auto& x : g) if (abs(x.first-pos) < 50) {
                    Vec4i& gl = x.second;
                    if (h) { gl[0]=min(gl[0],min(l[0],l[2])); gl[2]=max(gl[2],max(l[0],l[2])); }
                    else   { gl[1]=min(gl[1],min(l[1],l[3])); gl[3]=max(gl[3],max(l[1],l[3])); }
                    x.first = pos; done = true; break;
                }
                if (!done) g.push_back({pos, l});
            }
            v.clear(); for (auto& x : g) v.push_back(x.second);
        };
        merge(horiz, true); merge(vert, false);
        if (horiz.size() < 2 || vert.size() < 2) continue;

        sort(horiz.begin(), horiz.end(), [](const Vec4i& a, const Vec4i& b){ return (a[1]+a[3]) < (b[1]+b[3]); });
        sort(vert.begin(),  vert.end(),  [](const Vec4i& a, const Vec4i& b){ return (a[0]+a[2]) < (b[0]+b[2]); });

        auto L = [](const Vec4i& l) { return make_pair(Point2f(l[0],l[1]), Point2f(l[2],l[3])); };
        auto [t1,t2] = L(horiz.front()); auto [b1,b2] = L(horiz.back());
        auto [l1,l2] = L(vert.front());  auto [r1,r2] = L(vert.back());

        Point2f tl,tr,br,bl;
        if (!lineIntersect(t1,t2,l1,l2,tl)) continue;
        if (!lineIntersect(t1,t2,r1,r2,tr)) continue;
        if (!lineIntersect(b1,b2,r1,r2,br)) continue;
        if (!lineIntersect(b1,b2,l1,l2,bl)) continue;

        auto inside = [&](Point2f pt) {
            return pt.x >= 0 && pt.x < img.cols && pt.y >= 0 && pt.y < img.rows;
        };
        if (!inside(tl)||!inside(tr)||!inside(br)||!inside(bl)) continue;

        vector<Point> quad = {Point(tl),Point(tr),Point(br),Point(bl)};
        if (contourArea(quad) < 0.05 * img.cols * img.rows) continue;

        return sortCorners({tl, tr, br, bl});
    }

    return {};
}

// ─────────────────────────────────────────────────────────────
//  Wrapper principal
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectCorners(const Mat& img, bool textured)
{
    cout << "Méthode demandée : "
         << (textured ? "Structure & Hough (fond texturé)" : "Otsu & Contours (fond simple)") << "\n";

    vector<Point2f> corners;
    if (textured) {
        corners = detectTexturedBackground(img);
        if (corners.empty()) {
            cout << "  ❌ Méthode texture n'a pas trouvé de quadrilatère.\n";
        } else {
            cout << "  ✅ Méthode texture a détecté le document avec succès !\n";
            return corners;
        }
    } else {
        corners = detectSimpleBackground(img);
        if (corners.empty()) {
            cout << "  ❌ Méthode simple (Otsu) a échoué (contraste insuffisant ou fond trop complexe).\n";
        } else {
            cout << "  ✅ Méthode simple a détecté le document avec succès !\n";
            return corners;
        }
    }

    // Si la méthode demandée a échoué, tentative de secours avec l'autre méthode
    cout << "  → Tentative de secours avec l'autre méthode...\n";
    if (textured) {
        corners = detectSimpleBackground(img);
    } else {
        corners = detectTexturedBackground(img);
    }

    if (!corners.empty()) {
        cout << "  ✅ La méthode de secours a réussi à trouver les coins !\n";
        return corners;
    }

    cout << "  ⚠️ Aucune méthode automatique n'a pu segmenter l'image.\n";
    cout << "  → Placement de la boîte initiale par défaut (marge 20%) pour l'étape interactive.\n";
    float mx = img.cols * 0.20f, my = img.rows * 0.20f;
    float W  = img.cols - 1,     H  = img.rows - 1;
    return { {mx,my}, {W-mx,my}, {W-mx,H-my}, {mx,H-my} };
}
