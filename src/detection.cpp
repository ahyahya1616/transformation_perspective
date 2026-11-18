#include "detection.hpp"
#include "utils.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// ── Prétraitement commun ──────────────────────────────────────
// CLAHE : améliore le contraste local avant toute détection
static Mat preprocess(const Mat& img)
{
    Mat gray, clahe_out;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(gray, clahe_out);
    return clahe_out;
}

// ── Analyse du fond ───────────────────────────────────────────
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

// ── Score d'un quadrilatère (0=mauvais, 1=parfait) ───────────
// Critères : grande aire + angles proches de 90°
static double scoreQuad(const vector<Point>& quad, const Mat& img)
{
    if (quad.size() != 4) return 0;

    // 1. Aire relative
    double area = contourArea(quad);
    double imgArea = img.rows * img.cols;
    double areaScore = area / imgArea;
    if (areaScore < 0.05 || areaScore > 0.98) return 0;

    // 2. Score angulaire : les 4 angles doivent être ~90°
    double angleScore = 0;
    for (int i = 0; i < 4; i++) {
        Point2f a = quad[(i + 3) % 4];
        Point2f b = quad[i];
        Point2f c = quad[(i + 1) % 4];
        Point2f v1 = a - b, v2 = c - b;
        double cosA = v1.dot(v2) / (norm(v1) * norm(v2) + 1e-6);
        // cos(90°)=0 → score maximal
        angleScore += (1.0 - fabs(cosA));
    }
    angleScore /= 4.0;

    // 3. Ratio de remplissage (aire / bounding rect)
    Rect br = boundingRect(quad);
    double fillScore = area / (br.area() + 1e-6);

    return areaScore * 0.4 + angleScore * 0.4 + fillScore * 0.2;
}

// ── Méthode 1 : Fond simple ───────────────────────────────────
// Canny + Otsu + findContours + approxPolyDP
// Justification : fort contraste doc/fond → gradient net sur les bords
vector<Point2f> detectSimpleBackground(const Mat& img)
{
    Mat gray = preprocess(img);
    Mat blur, edges, thresh;
    GaussianBlur(gray, blur, Size(5, 5), 0);

    // Essai 1 : binarisation Otsu (très efficace sur fond uni)
    threshold(blur, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);
    morphologyEx(thresh, thresh, MORPH_OPEN,  kernel);

    // Essai 2 : Canny avec seuil adaptatif (médiane)
    Mat sorted; blur.reshape(1, 1).copyTo(sorted);
    cv::sort(sorted, sorted, SORT_EVERY_ROW);
    double med = sorted.at<uchar>(0, sorted.cols / 2);
    Canny(blur, edges, max(0.0, 0.5 * med), min(255.0, 1.5 * med));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);

    // Chercher le meilleur quad dans les deux sources
    vector<Mat> sources = { thresh, edges };
    double bestScore = 0;
    vector<Point2f> bestCorners;

    for (auto& src : sources) {
        vector<vector<Point>> contours;
        findContours(src, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

        // Trier par aire décroissante, ne garder que les 10 plus grands
        sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b){
            return contourArea(a) > contourArea(b);
        });
        if (contours.size() > 10) contours.resize(10);

        for (auto& c : contours) {
            // Tester plusieurs valeurs d'epsilon pour approxPolyDP
            for (double eps : {0.01, 0.02, 0.03, 0.05}) {
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
    }

    if (bestScore > 0.1) return bestCorners;
    return {};
}

// ── Méthode 2 : Fond texturé ──────────────────────────────────
// HoughLinesP avec filtrage et fusion des lignes proches
// Justification : Canny seul donne trop de bruit sur fond complexe.
// HoughLinesP filtre les longues lignes formant les bords du document.
static vector<Vec4i> mergeCloseLines(const vector<Vec4i>& lines, bool horizontal)
{
    // Regrouper les lignes parallèles proches (même position)
    vector<pair<int, Vec4i>> groups; // (position moyenne, ligne fusionnée)
    for (auto& l : lines) {
        int pos = horizontal ? (l[1] + l[3]) / 2 : (l[0] + l[2]) / 2;
        bool merged = false;
        for (auto& g : groups) {
            if (abs(g.first - pos) < 40) {
                // Fusionner : étendre la ligne existante
                Vec4i& gl = g.second;
                if (horizontal) {
                    gl[0] = min(gl[0], min(l[0], l[2]));
                    gl[2] = max(gl[2], max(l[0], l[2]));
                } else {
                    gl[1] = min(gl[1], min(l[1], l[3]));
                    gl[3] = max(gl[3], max(l[1], l[3]));
                }
                g.first = pos;
                merged = true;
                break;
            }
        }
        if (!merged) groups.push_back({pos, l});
    }
    vector<Vec4i> result;
    for (auto& g : groups) result.push_back(g.second);
    return result;
}

vector<Point2f> detectTexturedBackground(const Mat& img)
{
    Mat gray = preprocess(img);
    Mat blur, edges;
    // Filtre bilatéral : lisse le bruit mais préserve les bords nets
    bilateralFilter(gray, blur, 9, 75, 75);
    Canny(blur, edges, 30, 100);

    // Hough probabiliste : longues lignes seulement
    int minLen = max(img.cols, img.rows) / 6;
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 60, minLen, 30);

    if (lines.size() < 4) {
        // Réessayer avec seuil plus bas
        HoughLinesP(edges, lines, 1, CV_PI / 180, 40, minLen / 2, 40);
    }
    if (lines.size() < 4) return {};

    // Séparer horizontales et verticales
    vector<Vec4i> horiz, vert;
    for (auto& l : lines) {
        double angle = atan2(abs(l[3]-l[1]), abs(l[2]-l[0])) * 180.0 / CV_PI;
        if (angle < 30)        horiz.push_back(l);  // quasi-horizontale
        else if (angle > 60)   vert.push_back(l);   // quasi-verticale
    }
    if (horiz.size() < 2 || vert.size() < 2) return {};

    // Fusionner les lignes proches
    horiz = mergeCloseLines(horiz, true);
    vert  = mergeCloseLines(vert,  false);

    if (horiz.size() < 2 || vert.size() < 2) return {};

    // Garder les 2 lignes les plus éloignées (haut/bas, gauche/droite)
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
    auto [t1,t2] = L(top);  auto [b1,b2] = L(bottom);
    auto [l1,l2] = L(left); auto [r1,r2] = L(right);

    Point2f tl, tr, br, bl;
    if (!lineIntersect(t1,t2,l1,l2,tl)) return {};
    if (!lineIntersect(t1,t2,r1,r2,tr)) return {};
    if (!lineIntersect(b1,b2,r1,r2,br)) return {};
    if (!lineIntersect(b1,b2,l1,l2,bl)) return {};

    // Valider que les coins sont dans ou près de l'image
    float margin = 0.2f * max(img.cols, img.rows);
    auto inBounds = [&](Point2f p) {
        return p.x > -margin && p.x < img.cols + margin &&
               p.y > -margin && p.y < img.rows + margin;
    };
    if (!inBounds(tl) || !inBounds(tr) || !inBounds(br) || !inBounds(bl)) return {};

    return sortCorners({tl, tr, br, bl});
}

// ── Wrapper avec fallback ─────────────────────────────────────
vector<Point2f> detectCorners(const Mat& img, bool textured)
{
    cout << "Méthode : "
         << (textured ? "HoughLinesP (fond texturé)" : "Otsu+Canny (fond simple)") << "\n";

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
        // Fallback : coins à 10% à l'intérieur de l'image (mieux que les bords exacts)
        cout << "  Détection échouée → coins par défaut (marge 10%).\n";
        float mx = img.cols * 0.1f, my = img.rows * 0.1f;
        float W = img.cols - 1, H = img.rows - 1;
        corners = { {mx, my}, {W-mx, my}, {W-mx, H-my}, {mx, H-my} };
    }

    return corners;
}
