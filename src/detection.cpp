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
    return variance > 500.0;
}

// ─────────────────────────────────────────────────────────────
//  Score d'un quadrilatère
// ─────────────────────────────────────────────────────────────
static double scoreQuad(const vector<Point>& quad, const Mat& img)
{
    if (quad.size() != 4) return 0;

    double area    = contourArea(quad);
    double imgArea = (double)img.rows * img.cols;
    double areaRatio = area / imgArea;
    if (areaRatio < 0.05 || areaRatio > 0.98) return 0;

    // Vérifier que tous les coins sont DANS l'image
    for (auto& p : quad) {
        if (p.x < 0 || p.x >= img.cols || p.y < 0 || p.y >= img.rows)
            return 0;
    }

    double angleScore = 0;
    for (int i = 0; i < 4; i++) {
        Point2f a = quad[(i+3)%4], b = quad[i], c = quad[(i+1)%4];
        Point2f v1 = a-b, v2 = c-b;
        double cosA = v1.dot(v2) / (norm(v1)*norm(v2) + 1e-6);
        angleScore += (1.0 - fabs(cosA));
    }
    angleScore /= 4.0;

    Rect  br       = boundingRect(quad);
    double fillScore = area / (br.area() + 1e-6);

    return areaRatio * 0.4 + angleScore * 0.4 + fillScore * 0.2;
}

// Chercher le meilleur quad convexe à 4 côtés dans une image binaire
static vector<Point2f> bestQuadFromBinary(const Mat& binary, const Mat& img)
{
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Trier par aire décroissante
    sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b){
        return contourArea(a) > contourArea(b);
    });
    if (contours.size() > 15) contours.resize(15);

    double bestScore = 0;
    vector<Point2f> bestCorners;

    for (auto& c : contours) {
        for (double eps : {0.01, 0.02, 0.03, 0.05, 0.08}) {
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
//  MÉTHODE A : Binarisation Otsu  (fond simple)
//
//  JUSTIFICATION : Fort contraste doc/fond → Otsu trouve le
//  seuil global optimal pour séparer les deux régions.
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectSimpleBackground(const Mat& img)
{
    Mat gray = preprocess(img);
    Mat blur, thresh, edges;
    GaussianBlur(gray, blur, Size(5, 5), 0);

    // Otsu
    threshold(blur, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
    Mat k7 = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(thresh, thresh, MORPH_CLOSE, k7);
    morphologyEx(thresh, thresh, MORPH_OPEN,  k7);

    // Canny adaptatif (médiane)
    Mat sorted; blur.reshape(1,1).copyTo(sorted);
    cv::sort(sorted, sorted, SORT_EVERY_ROW);
    double med = sorted.at<uchar>(0, sorted.cols/2);
    Canny(blur, edges, max(0.0, 0.5*med), min(255.0, 1.5*med));
    morphologyEx(edges, edges, MORPH_CLOSE, k7);

    // Meilleur quad sur les 2 sources
    for (auto& src : {thresh, edges}) {
        auto result = bestQuadFromBinary(src, img);
        if (result.size() == 4) return result;
    }
    return {};
}

// ─────────────────────────────────────────────────────────────
//  MÉTHODE B : Seuillage ADAPTATIF  (fond texturé)
//
//  JUSTIFICATION : Le seuillage adaptatif calcule un seuil
//  différent pour chaque région de l'image (fenêtre locale),
//  ce qui le rend robuste aux variations d'éclairage et aux
//  fonds non uniformes. Il isole le document même quand l'Otsu
//  global échoue à cause du fond complexe.
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectTexturedBackground(const Mat& img)
{
    Mat gray = preprocess(img);
    Mat blur, thresh;

    // Filtre bilatéral : lisse le bruit en préservant les bords nets
    bilateralFilter(gray, blur, 11, 90, 90);

    // Essai 1 : seuillage adaptatif gaussien
    adaptiveThreshold(blur, thresh, 255,
                      ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 5);
    Mat k5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat k9 = getStructuringElement(MORPH_RECT, Size(9, 9));
    morphologyEx(thresh, thresh, MORPH_CLOSE, k9);
    morphologyEx(thresh, thresh, MORPH_OPEN,  k5);

    auto result = bestQuadFromBinary(thresh, img);
    if (result.size() == 4) return result;

    // Essai 2 : seuillage adaptatif avec taille de fenêtre plus grande
    adaptiveThreshold(blur, thresh, 255,
                      ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 51, 8);
    morphologyEx(thresh, thresh, MORPH_CLOSE, k9);
    morphologyEx(thresh, thresh, MORPH_OPEN,  k5);

    result = bestQuadFromBinary(thresh, img);
    if (result.size() == 4) return result;

    // Essai 3 : Hough sur bords Canny (approche originale en fallback)
    Mat edges;
    Canny(blur, edges, 20, 80);
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

        // Strict : tous les coins DANS l'image
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
//  Wrapper : essaie toutes les méthodes dans l'ordre
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectCorners(const Mat& img, bool textured)
{
    cout << "Méthode principale : "
         << (textured ? "Adaptatif (fond texturé)" : "Otsu+Canny (fond simple)") << "\n";

    // 1. Méthode choisie selon le type de fond
    vector<Point2f> corners = textured
        ? detectTexturedBackground(img)
        : detectSimpleBackground(img);

    // 2. Fallback : méthode alternative
    if (corners.size() != 4) {
        cout << "  → Essai de la méthode alternative...\n";
        corners = textured
            ? detectSimpleBackground(img)
            : detectTexturedBackground(img);
    }

    // 3. Fallback ultime : coins à 10% dans l'image
    if (corners.size() != 4) {
        cout << "  → Détection échouée — coins par défaut (marge 10%).\n";
        float mx = img.cols * 0.10f, my = img.rows * 0.10f;
        float W  = img.cols - 1,     H  = img.rows - 1;
        corners  = { {mx,my}, {W-mx,my}, {W-mx,H-my}, {mx,H-my} };
    }

    return corners;
}
