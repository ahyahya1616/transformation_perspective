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
//  Rejette catégoriquement le cadre extérieur de l'image
// ─────────────────────────────────────────────────────────────
static double scoreQuad(const vector<Point>& quad, const Mat& img)
{
    if (quad.size() != 4) return 0;

    double area    = contourArea(quad);
    double imgArea = (double)img.rows * img.cols;
    double areaRatio = area / imgArea;

    // Un document réel dans l'image ne prend ni < 5% ni > 80% de l'écran entier
    if (areaRatio < 0.05 || areaRatio > 0.80) return 0;

    // REJET DU CADRE EXTÉRIEUR : Si les coins touchent le bord de l'image (marge de 12px)
    int borderMargin = 12;
    int borderTouches = 0;
    for (auto& p : quad) {
        if (p.x < 0 || p.x >= img.cols || p.y < 0 || p.y >= img.rows) return 0;
        if (p.x <= borderMargin || p.x >= img.cols - borderMargin ||
            p.y <= borderMargin || p.y >= img.rows - borderMargin) {
            borderTouches++;
        }
    }
    // Si 2 coins ou plus touchent le bord extrême de la photo, c'est le cadre de l'image, pas le doc
    if (borderTouches >= 2) return 0;

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

    return areaRatio * 0.35 + angleScore * 0.45 + fillScore * 0.20;
}

// Chercher le meilleur quadrilatère dans une image binaire
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
//  MÉTHODE A : Binarisation Otsu + Canny (Fond Simple)
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

    // Canny adaptatif
    Mat sorted; blur.reshape(1,1).copyTo(sorted);
    cv::sort(sorted, sorted, SORT_EVERY_ROW);
    double med = sorted.at<uchar>(0, sorted.cols/2);
    Canny(blur, edges, max(0.0, 0.5*med), min(255.0, 1.5*med));
    morphologyEx(edges, edges, MORPH_CLOSE, k7);

    for (auto& src : {thresh, edges}) {
        auto result = bestQuadFromBinary(src, img);
        if (result.size() == 4) return result;
    }
    return {};
}

// ─────────────────────────────────────────────────────────────
//  MÉTHODE B : Analyse de structure & Hough Lines (Fond Texturé)
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectTexturedBackground(const Mat& img)
{
    Mat gray = preprocess(img);
    Mat blur, thresh;

    // 1. Filtre bilatéral + seuillage adaptatif local
    bilateralFilter(gray, blur, 9, 75, 75);
    for (int bs : {25, 45, 65}) {
        adaptiveThreshold(blur, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, bs, 6);
        Mat k5 = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(thresh, thresh, MORPH_CLOSE, k5);
        auto result = bestQuadFromBinary(thresh, img);
        if (result.size() == 4) return result;
    }

    // 2. Hough Lines : chercher les arêtes rectilignes internes du document
    Mat edges;
    Canny(blur, edges, 15, 50);
    Mat k3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(edges, edges, k3);

    int minDim = min(img.cols, img.rows);
    int margin = 20; // Ignorer les lignes du cadre de la caméra

    for (int minLen : {minDim / 6, minDim / 8, minDim / 10}) {
        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI / 180, 35, minLen, 25);
        if (lines.size() < 4) continue;

        vector<Vec4i> horiz, vert;
        for (auto& l : lines) {
            // Exclure les bords extérieurs de l'image
            if ((l[0] < margin && l[2] < margin) ||
                (l[0] > img.cols - margin && l[2] > img.cols - margin) ||
                (l[1] < margin && l[3] < margin) ||
                (l[1] > img.rows - margin && l[3] > img.rows - margin))
                continue;

            double a = atan2(abs(l[3]-l[1]), abs(l[2]-l[0])) * 180.0 / CV_PI;
            if (a < 20)      horiz.push_back(l);
            else if (a > 70) vert.push_back(l);
        }
        if (horiz.size() < 2 || vert.size() < 2) continue;

        // Regrouper par position pour trouver les bords gauche, droite, haut, bas
        vector<int> hPositions, vPositions;
        for (auto& l : horiz) hPositions.push_back((l[1] + l[3]) / 2);
        for (auto& l : vert)  vPositions.push_back((l[0] + l[2]) / 2);

        sort(hPositions.begin(), hPositions.end());
        sort(vPositions.begin(), vPositions.end());

        // Prendre les clusters représentatifs éloignés
        int topY = hPositions[hPositions.size() * 0.15];
        int botY = hPositions[hPositions.size() * 0.85];
        int leftX = vPositions[vPositions.size() * 0.15];
        int rightX = vPositions[vPositions.size() * 0.85];

        if ((botY - topY) > 0.25 * img.rows && (rightX - leftX) > 0.25 * img.cols) {
            Point2f tl((float)leftX, (float)topY);
            Point2f tr((float)rightX, (float)topY);
            Point2f br((float)rightX, (float)botY);
            Point2f bl((float)leftX, (float)botY);
            return sortCorners({tl, tr, br, bl});
        }
    }

    return {};
}

// ─────────────────────────────────────────────────────────────
//  Wrapper avec fallback
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectCorners(const Mat& img, bool textured)
{
    cout << "Méthode : "
         << (textured ? "Structure & Hough (fond texturé)" : "Otsu & Contours (fond simple)") << "\n";

    vector<Point2f> corners = textured
        ? detectTexturedBackground(img)
        : detectSimpleBackground(img);

    if (corners.size() != 4) {
        cout << "  → Essai de la méthode alternative...\n";
        corners = textured
            ? detectSimpleBackground(img)
            : detectTexturedBackground(img);
    }

    if (corners.size() != 4) {
        cout << "  → Détection automatique incertaine : placement initial au centre (marge 20%).\n";
        float mx = img.cols * 0.20f, my = img.rows * 0.20f;
        float W  = img.cols - 1,     H  = img.rows - 1;
        corners  = { {mx,my}, {W-mx,my}, {W-mx,H-my}, {mx,H-my} };
    }

    return corners;
}
