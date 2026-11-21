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
static double scoreQuad(const vector<Point>& quad, const Mat& img)
{
    if (quad.size() != 4) return 0;

    double area    = contourArea(quad);
    double imgArea = (double)img.rows * img.cols;
    double areaRatio = area / imgArea;

    // Un document réel ne prend pas < 5% ni > 80% de l'image
    if (areaRatio < 0.05 || areaRatio > 0.80) return 0;

    // Rejeter le cadre extérieur de l'image
    int bm = 12;
    int borderTouches = 0;
    for (auto& p : quad) {
        if (p.x < 0 || p.x >= img.cols || p.y < 0 || p.y >= img.rows) return 0;
        if (p.x <= bm || p.x >= img.cols - bm || p.y <= bm || p.y >= img.rows - bm) {
            borderTouches++;
        }
    }
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
//  MÉTHODE 1 : Binarisation Otsu + Canny (Fond Simple)
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectSimpleBackground(const Mat& img)
{
    Mat gray = preprocess(img);
    Mat blur, thresh, edges;
    GaussianBlur(gray, blur, Size(5, 5), 0);

    // Essai A : Binarisation globale Otsu
    threshold(blur, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
    Mat k7 = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(thresh, thresh, MORPH_CLOSE, k7);
    morphologyEx(thresh, thresh, MORPH_OPEN,  k7);
    auto resOtsu = bestQuadFromBinary(thresh, img);
    if (!resOtsu.empty()) return resOtsu;

    // Essai B : Canny standard
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
    Mat gray = preprocess(img);
    Mat blur, thresh;

    // Essai A : Filtre bilatéral + seuillage adaptatif local (fenêtre variable)
    bilateralFilter(gray, blur, 7, 50, 50);
    for (int bs : {25, 45, 65}) {
        adaptiveThreshold(blur, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, bs, 6);
        Mat k5 = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(thresh, thresh, MORPH_CLOSE, k5);
        auto result = bestQuadFromBinary(thresh, img);
        if (result.size() == 4) return result;
    }

    // Essai B : Lignes de Hough pour reconstituer le quadrilatère sur fond bruité
    Mat edges;
    Canny(blur, edges, 15, 50);

    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 35, 70, 25);

    int margin = 20;
    vector<Vec4i> horiz, vert;
    for (auto& l : lines) {
        if ((l[0] < margin && l[2] < margin) || (l[0] > img.cols - margin && l[2] > img.cols - margin) ||
            (l[1] < margin && l[3] < margin) || (l[1] > img.rows - margin && l[3] > img.rows - margin))
            continue;

        double a = atan2(abs(l[3]-l[1]), abs(l[2]-l[0])) * 180.0 / CV_PI;
        if (a < 20)      horiz.push_back(l);
        else if (a > 70) vert.push_back(l);
    }

    if (horiz.size() >= 2 && vert.size() >= 2) {
        sort(horiz.begin(), horiz.end(), [](const Vec4i& a, const Vec4i& b){
            return (a[1]+a[3]) < (b[1]+b[3]);
        });
        sort(vert.begin(), vert.end(), [](const Vec4i& a, const Vec4i& b){
            return (a[0]+a[2]) < (b[0]+b[2]);
        });

        int bestTopY = -1, bestBotY = -1;
        float maxScoreH = 0;
        for (size_t i = 0; i < min((size_t)8, horiz.size()); i++) {
            int y1 = (horiz[i][1] + horiz[i][3]) / 2;
            float len1 = hypot(horiz[i][2]-horiz[i][0], horiz[i][3]-horiz[i][1]);
            for (int j = (int)horiz.size() - 1; j >= max(0, (int)horiz.size() - 8); j--) {
                int y2 = (horiz[j][1] + horiz[j][3]) / 2;
                float len2 = hypot(horiz[j][2]-horiz[j][0], horiz[j][3]-horiz[j][1]);
                int dy = y2 - y1;
                if (dy > 0.25 * img.rows && dy < 0.90 * img.rows) {
                    float score = (len1 + len2) * dy;
                    if (score > maxScoreH) {
                        maxScoreH = score;
                        bestTopY = y1;
                        bestBotY = y2;
                    }
                }
            }
        }

        int bestLeftX = -1, bestRightX = -1;
        float maxScoreV = 0;
        for (size_t i = 0; i < min((size_t)12, vert.size()); i++) {
            int x1 = (vert[i][0] + vert[i][2]) / 2;
            float len1 = hypot(vert[i][2]-vert[i][0], vert[i][3]-vert[i][1]);
            for (int j = (int)vert.size() - 1; j >= max(0, (int)vert.size() - 12); j--) {
                int x2 = (vert[j][0] + vert[j][2]) / 2;
                float len2 = hypot(vert[j][2]-vert[j][0], vert[j][3]-vert[j][1]);
                int dx = x2 - x1;
                if (dx > 0.20 * img.cols && dx < 0.90 * img.cols) {
                    float score = (len1 + len2) * dx;
                    if (score > maxScoreV) {
                        maxScoreV = score;
                        bestLeftX = x1;
                        bestRightX = x2;
                    }
                }
            }
        }

        if (bestTopY > 0 && bestBotY > 0 && bestLeftX > 0 && bestRightX > 0) {
            return {
                Point2f((float)bestLeftX,  (float)bestTopY),
                Point2f((float)bestRightX, (float)bestTopY),
                Point2f((float)bestRightX, (float)bestBotY),
                Point2f((float)bestLeftX,  (float)bestBotY)
            };
        }
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

    // Si la méthode demandée a échoué, on essaie l'autre en secours :
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
