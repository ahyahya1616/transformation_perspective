#include "detection.hpp"
#include "utils.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// ── Analyse du fond ───────────────────────────────────────────

bool isTexturedBackground(const Mat& img)
{
    Mat gray, lap;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);

    Scalar mean, stddev;
    meanStdDev(lap, mean, stddev);
    double variance = stddev[0] * stddev[0];

    cout << "  Variance Laplacien = " << variance << "\n";
    return variance > 500.0;
}

// ── Méthode 1 : Fond simple ───────────────────────────────────

vector<Point2f> detectSimpleBackground(const Mat& img)
{
    Mat gray, blur, edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 0);

    // Seuil de Canny adaptatif basé sur la médiane
    Mat sorted;
    blur.reshape(1, 1).copyTo(sorted);
    cv::sort(sorted, sorted, SORT_EVERY_ROW);
    double med = sorted.at<uchar>(0, sorted.cols / 2);
    Canny(blur, edges, max(0.0, 0.66 * med), min(255.0, 1.33 * med));

    // Fermeture pour boucher les trous dans les contours
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);

    // Trouver le plus grand contour quadrilatère convexe
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b){
        return contourArea(a) > contourArea(b);
    });

    for (auto& c : contours) {
        vector<Point> approx;
        approxPolyDP(c, approx, 0.02 * arcLength(c, true), true);

        if (approx.size() == 4 && isContourConvex(approx) && contourArea(approx) > 10000) {
            vector<Point2f> pts;
            for (auto& p : approx) pts.push_back(Point2f(p));
            return sortCorners(pts);
        }
    }
    return {};
}

// ── Méthode 2 : Fond texturé ──────────────────────────────────

vector<Point2f> detectTexturedBackground(const Mat& img)
{
    Mat gray, blur, edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(7, 7), 0);
    Canny(blur, edges, 50, 150);

    // Hough probabiliste : longues lignes seulement
    int minLen = max(img.cols, img.rows) / 5;
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 80, minLen, 20);

    if (lines.size() < 4) return {};

    // Séparer horizontales et verticales (±30°)
    vector<Vec4i> horiz, vert;
    for (auto& l : lines) {
        double angle = atan2(l[3]-l[1], l[2]-l[0]) * 180.0 / CV_PI;
        if (fabs(angle) < 30 || fabs(angle) > 150) horiz.push_back(l);
        else                                         vert.push_back(l);
    }
    if (horiz.size() < 2 || vert.size() < 2) return {};

    // Garder les lignes les plus éloignées (haut/bas, gauche/droite)
    sort(horiz.begin(), horiz.end(), [](const Vec4i& a, const Vec4i& b){
        return (a[1]+a[3]) < (b[1]+b[3]);
    });
    sort(vert.begin(), vert.end(), [](const Vec4i& a, const Vec4i& b){
        return (a[0]+a[2]) < (b[0]+b[2]);
    });

    Vec4i top = horiz.front(), bottom = horiz.back();
    Vec4i left = vert.front(), right = vert.back();

    auto L = [](const Vec4i& l) -> pair<Point2f, Point2f> {
        return { {(float)l[0],(float)l[1]}, {(float)l[2],(float)l[3]} };
    };

    auto [t1,t2] = L(top);    auto [b1,b2] = L(bottom);
    auto [l1,l2] = L(left);   auto [r1,r2] = L(right);

    Point2f tl, tr, br, bl;
    if (!lineIntersect(t1,t2,l1,l2,tl)) return {};
    if (!lineIntersect(t1,t2,r1,r2,tr)) return {};
    if (!lineIntersect(b1,b2,r1,r2,br)) return {};
    if (!lineIntersect(b1,b2,l1,l2,bl)) return {};

    return sortCorners({tl, tr, br, bl});
}

// ── Wrapper avec fallback ─────────────────────────────────────

vector<Point2f> detectCorners(const Mat& img, bool textured)
{
    cout << "Méthode : " << (textured ? "HoughLinesP (fond texturé)" : "Contours Canny (fond simple)") << "\n";

    vector<Point2f> corners = textured
        ? detectTexturedBackground(img)
        : detectSimpleBackground(img);

    if (corners.size() != 4) {
        cout << "  Méthode principale échouée → essai de l'alternative...\n";
        corners = textured
            ? detectSimpleBackground(img)
            : detectTexturedBackground(img);
    }

    if (corners.size() != 4) {
        cout << "  Détection échouée → coins de l'image utilisés.\n";
        float W = img.cols - 1, H = img.rows - 1;
        corners = { {0,0}, {W,0}, {W,H}, {0,H} };
    }

    return corners;
}
