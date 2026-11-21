#include "detection.hpp"
#include "utils.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// ─────────────────────────────────────────────────────────────
//  Prétraitement commun : CLAHE pour améliorer le contraste local
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
//  Détection automatique du type de fond (variance Laplacien)
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
//  Score d'un quadrilatère (0 = mauvais, 1 = parfait)
//  Critères : aire + angles proches de 90° + remplissage
// ─────────────────────────────────────────────────────────────
static double scoreQuad(const vector<Point>& quad, const Mat& img)
{
    if (quad.size() != 4) return 0;

    double area    = contourArea(quad);
    double imgArea = img.rows * img.cols;
    double areaRatio = area / imgArea;
    if (areaRatio < 0.05 || areaRatio > 0.98) return 0;

    double angleScore = 0;
    for (int i = 0; i < 4; i++) {
        Point2f a = quad[(i+3)%4], b = quad[i], c = quad[(i+1)%4];
        Point2f v1 = a-b, v2 = c-b;
        double cosA = v1.dot(v2) / (norm(v1)*norm(v2) + 1e-6);
        angleScore += (1.0 - fabs(cosA)); // cos(90°)=0 → score max
    }
    angleScore /= 4.0;

    Rect  br = boundingRect(quad);
    double fillScore = area / (br.area() + 1e-6);

    return areaRatio * 0.4 + angleScore * 0.4 + fillScore * 0.2;
}

// ─────────────────────────────────────────────────────────────
//  MÉTHODE 1 — Fond SIMPLE
//
//  Pipeline : CLAHE → Otsu + Canny → findContours → approxPolyDP
//
//  JUSTIFICATION :
//  Sur un fond uniforme, le document crée un fort contraste avec
//  le fond. La binarisation d'Otsu sépare efficacement les deux
//  régions car leurs niveaux de gris sont bien distincts.
//  approxPolyDP extrait directement le quadrilatère sans ambiguïté.
// ─────────────────────────────────────────────────────────────
vector<Point2f> detectSimpleBackground(const Mat& img)
{
    Mat gray = preprocess(img);
    Mat blur, edges, thresh;
    GaussianBlur(gray, blur, Size(5, 5), 0);

    // Binarisation Otsu : trouve le seuil optimal automatiquement
    threshold(blur, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel); // boucher les trous
    morphologyEx(thresh, thresh, MORPH_OPEN,  kernel); // supprimer le bruit

    // Canny adaptatif (seuil basé sur la médiane de l'image)
    Mat sorted; blur.reshape(1,1).copyTo(sorted);
    cv::sort(sorted, sorted, SORT_EVERY_ROW);
    double med = sorted.at<uchar>(0, sorted.cols/2);
    Canny(blur, edges, max(0.0, 0.5*med), min(255.0, 1.5*med));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);

    // Chercher le meilleur quad dans les deux sources (Otsu ET Canny)
    vector<Mat> sources = { thresh, edges };
    double bestScore = 0;
    vector<Point2f> bestCorners;

    for (auto& src : sources) {
        vector<vector<Point>> contours;
        findContours(src, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
        sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b){
            return contourArea(a) > contourArea(b);
        });
        if (contours.size() > 10) contours.resize(10);

        for (auto& c : contours) {
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

// ─────────────────────────────────────────────────────────────
//  MÉTHODE 2 — Fond TEXTURÉ
//
//  Pipeline : CLAHE → Flou bilateral → Canny → HoughLinesP
//             → fusion des lignes proches → 4 intersections
//
//  JUSTIFICATION :
//  Sur un fond complexe, findContours génère des centaines de
//  contours parasites. HoughLinesP ne retient que les LONGUES
//  lignes droites continues (= bords du document) et ignore les
//  courtes lignes de texture. Le filtre bilatéral lisse le bruit
//  sans détruire les bords nets du document.
//
//  AMÉLIORATIONS :
//  - On essaie plusieurs seuils Hough automatiquement
//  - On choisit les meilleures lignes par longueur (pas juste position)
//  - Validation géométrique du quadrilatère final
// ─────────────────────────────────────────────────────────────

// Fusion des lignes parallèles proches (évite les doublons)
static vector<Vec4i> mergeCloseLines(const vector<Vec4i>& lines, bool horizontal)
{
    vector<pair<int, Vec4i>> groups;
    for (auto& l : lines) {
        int pos = horizontal ? (l[1]+l[3])/2 : (l[0]+l[2])/2;
        bool merged = false;
        for (auto& g : groups) {
            if (abs(g.first - pos) < 50) {
                Vec4i& gl = g.second;
                if (horizontal) {
                    gl[0] = min(gl[0], min(l[0], l[2]));
                    gl[2] = max(gl[2], max(l[0], l[2]));
                } else {
                    gl[1] = min(gl[1], min(l[1], l[3]));
                    gl[3] = max(gl[3], max(l[1], l[3]));
                }
                g.first = pos;
                merged  = true;
                break;
            }
        }
        if (!merged) groups.push_back({pos, l});
    }
    vector<Vec4i> result;
    for (auto& g : groups) result.push_back(g.second);
    return result;
}

// Longueur d'un segment Vec4i
static float lineLen(const Vec4i& l)
{
    return hypot(l[2]-l[0], l[3]-l[1]);
}

vector<Point2f> detectTexturedBackground(const Mat& img)
{
    Mat gray = preprocess(img);
    Mat blur, edges;

    // Filtre bilatéral : lisse le bruit en préservant les bords nets
    bilateralFilter(gray, blur, 11, 90, 90);
    Canny(blur, edges, 20, 80);

    // Fermeture pour connecter les bords brisés du document
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);

    // Essayer plusieurs paramètres Hough du plus strict au plus souple
    int minDim = min(img.cols, img.rows);
    vector<Point2f> bestResult;

    struct HoughParams { int vote; int minLen; int maxGap; };
    vector<HoughParams> attempts = {
        { 80, minDim/4, 20 },
        { 60, minDim/6, 30 },
        { 40, minDim/8, 40 },
        { 30, minDim/10, 50 },
    };

    for (auto& p : attempts) {
        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI/180, p.vote, p.minLen, p.maxGap);
        if (lines.size() < 4) continue;

        // Séparer horizontales et verticales
        vector<Vec4i> horiz, vert;
        for (auto& l : lines) {
            double angle = atan2(abs(l[3]-l[1]), abs(l[2]-l[0])) * 180.0 / CV_PI;
            if (angle < 25)      horiz.push_back(l);
            else if (angle > 65) vert.push_back(l);
        }
        if (horiz.size() < 2 || vert.size() < 2) continue;

        // Fusionner les lignes parallèles proches
        horiz = mergeCloseLines(horiz, true);
        vert  = mergeCloseLines(vert,  false);
        if (horiz.size() < 2 || vert.size() < 2) continue;

        // Choisir les 2 lignes les plus longues ET les plus éloignées
        // → trier par position puis prendre front/back
        sort(horiz.begin(), horiz.end(), [](const Vec4i& a, const Vec4i& b){
            return (a[1]+a[3]) < (b[1]+b[3]);
        });
        sort(vert.begin(), vert.end(), [](const Vec4i& a, const Vec4i& b){
            return (a[0]+a[2]) < (b[0]+b[2]);
        });

        // Parmi les lignes du haut et du bas, prendre la plus longue
        Vec4i top    = horiz.front();
        Vec4i bottom = horiz.back();
        Vec4i left   = vert.front();
        Vec4i right  = vert.back();

        // Si plusieurs lignes proches des extrêmes, prendre la plus longue
        auto longest = [](vector<Vec4i>& v, int idx) -> Vec4i {
            Vec4i best = v[idx];
            int ref = idx == 0 ? (v[0][1]+v[0][3])/2 : (v.back()[1]+v.back()[3])/2;
            for (auto& l : v) {
                int pos = (l[1]+l[3])/2;
                if (abs(pos - ref) < 60 && lineLen(l) > lineLen(best))
                    best = l;
            }
            return best;
        };
        auto longestV = [](vector<Vec4i>& v, int idx) -> Vec4i {
            Vec4i best = v[idx];
            int ref = idx == 0 ? (v[0][0]+v[0][2])/2 : (v.back()[0]+v.back()[2])/2;
            for (auto& l : v) {
                int pos = (l[0]+l[2])/2;
                if (abs(pos - ref) < 60 && lineLen(l) > lineLen(best))
                    best = l;
            }
            return best;
        };
        top    = longest(horiz, 0);
        bottom = longest(horiz, (int)horiz.size()-1);
        left   = longestV(vert, 0);
        right  = longestV(vert, (int)vert.size()-1);

        // Calculer les 4 intersections
        auto L = [](const Vec4i& l) -> pair<Point2f,Point2f> {
            return { {(float)l[0],(float)l[1]}, {(float)l[2],(float)l[3]} };
        };
        auto [t1,t2] = L(top);  auto [b1,b2] = L(bottom);
        auto [l1,l2] = L(left); auto [r1,r2] = L(right);

        Point2f tl, tr, br, bl;
        if (!lineIntersect(t1,t2,l1,l2,tl)) continue;
        if (!lineIntersect(t1,t2,r1,r2,tr)) continue;
        if (!lineIntersect(b1,b2,r1,r2,br)) continue;
        if (!lineIntersect(b1,b2,l1,l2,bl)) continue;

        // Valider : les coins doivent être proches de l'image
        float margin = 0.25f * max(img.cols, img.rows);
        auto ok = [&](Point2f pt) {
            return pt.x > -margin && pt.x < img.cols+margin &&
                   pt.y > -margin && pt.y < img.rows+margin;
        };
        if (!ok(tl)||!ok(tr)||!ok(br)||!ok(bl)) continue;

        // Valider : l'aire du quad doit être raisonnable
        vector<Point> quad = { Point(tl), Point(tr), Point(br), Point(bl) };
        double area = contourArea(quad);
        if (area < 0.05 * img.cols * img.rows) continue;

        bestResult = sortCorners({tl, tr, br, bl});
        break; // Premier résultat valide = on s'arrête
    }

    return bestResult;
}

// ─────────────────────────────────────────────────────────────
//  Wrapper avec fallback automatique
// ─────────────────────────────────────────────────────────────
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
        cout << "  Détection échouée → coins par défaut (marge 10%).\n";
        float mx = img.cols * 0.10f, my = img.rows * 0.10f;
        float W  = img.cols - 1,     H  = img.rows - 1;
        corners  = { {mx,my}, {W-mx,my}, {W-mx,H-my}, {mx,H-my} };
    }

    return corners;
}
