/**
 * Devoir 2 - Transformation de Perspective
 * 
 * Programme C++ avec OpenCV pour :
 * 1. Détecter automatiquement les 4 coins d'un document
 * 2. Afficher le résultat de la détection
 * 3. Permettre l'ajustement interactif des points
 * 4. Appliquer la transformation de perspective
 * 
 * Deux méthodes selon le type d'image :
 *   - Fond simple  : détection par contours (Canny + findContours)
 *   - Fond texturé : détection par transformée de Hough (lignes)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// ─────────────────────────────────────────────────────────────
//  Structures et variables globales pour l'interaction souris
// ─────────────────────────────────────────────────────────────

struct State {
    vector<Point2f> corners;   // 4 coins du document
    int             dragging;  // indice du coin en cours de déplacement (-1 = aucun)
    Mat             display;   // image affichée dans la fenêtre interactive
};

static State g_state;

// ─────────────────────────────────────────────────────────────
//  Utilitaires géométriques
// ─────────────────────────────────────────────────────────────

// Trier les 4 coins dans l'ordre : haut-gauche, haut-droite, bas-droite, bas-gauche
vector<Point2f> sortCorners(vector<Point2f> pts)
{
    // Centroïde
    Point2f center(0, 0);
    for (auto& p : pts) center += p;
    center *= (1.0f / pts.size());

    vector<Point2f> tl, tr, br, bl;
    for (auto& p : pts) {
        if (p.x < center.x && p.y < center.y) tl.push_back(p);
        else if (p.x > center.x && p.y < center.y) tr.push_back(p);
        else if (p.x > center.x && p.y > center.y) br.push_back(p);
        else bl.push_back(p);
    }

    // Prendre le meilleur candidat dans chaque quadrant
    auto best = [](vector<Point2f>& v, Point2f ref) -> Point2f {
        if (v.empty()) return ref;
        return *min_element(v.begin(), v.end(), [&](const Point2f& a, const Point2f& b){
            return norm(a - ref) < norm(b - ref);
        });
    };
    return { best(tl, {0,0}),
             best(tr, {(float)1e9,0}),
             best(br, {(float)1e9,(float)1e9}),
             best(bl, {0,(float)1e9}) };
}

// Distance point-à-point
float dist(Point2f a, Point2f b) { return (float)norm(a - b); }

// ─────────────────────────────────────────────────────────────
//  Méthode 1 : Fond SIMPLE — détection par contours (Canny)
//
//  Justification : sur un fond uni et contrasté, le document
//  génère un fort gradient en bordure. Canny + findContours
//  isole facilement le plus grand quadrilatère.
// ─────────────────────────────────────────────────────────────

vector<Point2f> detectSimpleBackground(const Mat& img)
{
    Mat gray, blur, edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 0);

    // Seuil adaptatif de Canny (médiane × facteurs)
    double med = 0;
    {
        Mat sorted; blur.reshape(1,1).copyTo(sorted);
        cv::sort(sorted, sorted, SORT_EVERY_ROW);
        med = sorted.at<uchar>(0, sorted.cols / 2);
    }
    double lo = max(0.0, 0.66 * med);
    double hi = min(255.0, 1.33 * med);
    Canny(blur, edges, lo, hi);

    // Fermeture morphologique pour boucher les trous
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);

    // Trouver le plus grand contour quadrilatère
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(),
         [](const vector<Point>& a, const vector<Point>& b){
             return contourArea(a) > contourArea(b);
         });

    for (auto& c : contours) {
        double eps = 0.02 * arcLength(c, true);
        vector<Point> approx;
        approxPolyDP(c, approx, eps, true);

        if (approx.size() == 4 && isContourConvex(approx)
            && contourArea(approx) > 10000) {
            vector<Point2f> pts;
            for (auto& p : approx) pts.push_back(Point2f(p));
            return sortCorners(pts);
        }
    }
    return {}; // échec
}

// ─────────────────────────────────────────────────────────────
//  Méthode 2 : Fond TEXTURÉ — détection par lignes de Hough
//
//  Justification : sur un fond complexe, Canny seul produit
//  trop de bruit. La transformée de Hough probabiliste détecte
//  les longues lignes droites formant les bords du document,
//  plus robuste aux textures parasites.
// ─────────────────────────────────────────────────────────────

// Intersection de deux segments infinis (droites)
bool lineIntersect(Point2f p1, Point2f p2, Point2f p3, Point2f p4, Point2f& out)
{
    float d = (p1.x-p2.x)*(p3.y-p4.y) - (p1.y-p2.y)*(p3.x-p4.x);
    if (fabs(d) < 1e-6) return false;
    float t = ((p1.x-p3.x)*(p3.y-p4.y) - (p1.y-p3.y)*(p3.x-p4.x)) / d;
    out.x = p1.x + t*(p2.x-p1.x);
    out.y = p1.y + t*(p2.y-p1.y);
    return true;
}

vector<Point2f> detectTexturedBackground(const Mat& img)
{
    Mat gray, blur, edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(7, 7), 0);
    Canny(blur, edges, 50, 150);

    // Hough probabiliste : on garde les longues lignes
    int minLen = max(img.cols, img.rows) / 5;
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 80, minLen, 20);

    if (lines.size() < 4) return {};

    // Séparer lignes horizontales et verticales (±30°)
    vector<Vec4i> horiz, vert;
    for (auto& l : lines) {
        double angle = atan2(l[3]-l[1], l[2]-l[0]) * 180.0 / CV_PI;
        if (fabs(angle) < 30 || fabs(angle) > 150) horiz.push_back(l);
        else vert.push_back(l);
    }
    if (horiz.size() < 2 || vert.size() < 2) return {};

    // Garder les 2 lignes horizontales les plus éloignées (haut/bas)
    auto byY = [](const Vec4i& a, const Vec4i& b){
        return (a[1]+a[3])/2 < (b[1]+b[3])/2;
    };
    sort(horiz.begin(), horiz.end(), byY);
    Vec4i top    = horiz.front();
    Vec4i bottom = horiz.back();

    auto byX = [](const Vec4i& a, const Vec4i& b){
        return (a[0]+a[2])/2 < (b[0]+b[2])/2;
    };
    sort(vert.begin(), vert.end(), byX);
    Vec4i left  = vert.front();
    Vec4i right = vert.back();

    // Calculer les 4 intersections
    auto L2pts = [](const Vec4i& l) -> pair<Point2f,Point2f> {
        return { {(float)l[0],(float)l[1]}, {(float)l[2],(float)l[3]} };
    };
    auto [t1,t2] = L2pts(top);
    auto [b1,b2] = L2pts(bottom);
    auto [l1,l2] = L2pts(left);
    auto [r1,r2] = L2pts(right);

    Point2f tl, tr, br, bl;
    if (!lineIntersect(t1,t2,l1,l2,tl)) return {};
    if (!lineIntersect(t1,t2,r1,r2,tr)) return {};
    if (!lineIntersect(b1,b2,r1,r2,br)) return {};
    if (!lineIntersect(b1,b2,l1,l2,bl)) return {};

    return sortCorners({tl, tr, br, bl});
}

// ─────────────────────────────────────────────────────────────
//  Dessin des coins sur l'image
// ─────────────────────────────────────────────────────────────

void drawCorners(Mat& img, const vector<Point2f>& pts, bool labels = true)
{
    if (pts.size() != 4) return;
    Scalar color(0, 255, 0);
    Scalar lineColor(0, 200, 255);

    // Quadrilatère
    for (int i = 0; i < 4; i++)
        line(img, pts[i], pts[(i+1)%4], lineColor, 2);

    // Cercles et étiquettes
    const char* names[] = {"TL","TR","BR","BL"};
    for (int i = 0; i < 4; i++) {
        circle(img, pts[i], 10, color, -1);
        circle(img, pts[i], 12, Scalar(0,0,0), 2);
        if (labels)
            putText(img, names[i], pts[i] + Point2f(14, -8),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,255), 2);
    }
}

// ─────────────────────────────────────────────────────────────
//  Callback souris pour l'ajustement interactif
// ─────────────────────────────────────────────────────────────

void onMouse(int event, int x, int y, int /*flags*/, void* /*userdata*/)
{
    Point2f cur(x, y);

    if (event == EVENT_LBUTTONDOWN) {
        // Chercher le coin le plus proche (seuil 20px)
        float best = 20.0f;
        g_state.dragging = -1;
        for (int i = 0; i < 4; i++) {
            float d = dist(g_state.corners[i], cur);
            if (d < best) { best = d; g_state.dragging = i; }
        }
    }
    else if (event == EVENT_MOUSEMOVE && g_state.dragging >= 0) {
        g_state.corners[g_state.dragging] = cur;
        // Redessiner
        Mat tmp = g_state.display.clone();
        drawCorners(tmp, g_state.corners);
        putText(tmp, "Ajustez les coins. [ENTREE] = appliquer  [R] = reset  [Q] = quitter",
                Point(10,25), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,255,0), 1);
        imshow("Ajustement interactif", tmp);
    }
    else if (event == EVENT_LBUTTONUP) {
        g_state.dragging = -1;
    }
}

// ─────────────────────────────────────────────────────────────
//  Transformation de perspective
// ─────────────────────────────────────────────────────────────

Mat applyPerspective(const Mat& src, const vector<Point2f>& corners)
{
    // Calculer la taille de sortie à partir des longueurs des côtés
    float wTop  = dist(corners[0], corners[1]);
    float wBot  = dist(corners[3], corners[2]);
    float hLeft = dist(corners[0], corners[3]);
    float hRight= dist(corners[1], corners[2]);

    int W = (int)max(wTop, wBot);
    int H = (int)max(hLeft, hRight);

    vector<Point2f> dst = {
        {0.f,     0.f    },
        {(float)W-1, 0.f },
        {(float)W-1, (float)H-1},
        {0.f,     (float)H-1}
    };

    Mat M = getPerspectiveTransform(corners, dst);
    Mat warped;
    warpPerspective(src, warped, M, Size(W, H));
    return warped;
}

// ─────────────────────────────────────────────────────────────
//  Évaluer la complexité du fond (variance locale)
// ─────────────────────────────────────────────────────────────

bool isTexturedBackground(const Mat& img)
{
    Mat gray, lap;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);
    Scalar mean, stddev;
    meanStdDev(lap, mean, stddev);
    // Variance élevée → fond texturé
    double variance = stddev[0] * stddev[0];
    cout << "  Variance Laplacien = " << variance << endl;
    return variance > 500.0;  // seuil empirique
}

// ─────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    // ── Lecture de l'image ──────────────────────────────────
    string imgPath;
    if (argc >= 2) {
        imgPath = argv[1];
    } else {
        cout << "Usage: " << argv[0] << " <image_path> [simple|texture]\n";
        cout << "Chemin de l'image : ";
        cin >> imgPath;
    }

    Mat img = imread(imgPath);
    if (img.empty()) {
        cerr << "Erreur : impossible de lire \"" << imgPath << "\"\n";
        return 1;
    }

    // Redimensionner si trop grande (pour l'affichage)
    int maxDim = 900;
    float scale = 1.0f;
    if (img.cols > maxDim || img.rows > maxDim) {
        scale = (float)maxDim / max(img.cols, img.rows);
        resize(img, img, Size(), scale, scale);
    }

    // ── Choix de la méthode ─────────────────────────────────
    bool textured = false;
    if (argc >= 3) {
        string mode = argv[2];
        textured = (mode == "texture" || mode == "texturé" || mode == "textured");
    } else {
        cout << "\nDétection automatique du type de fond...\n";
        textured = isTexturedBackground(img);
        cout << "  → Fond détecté : " << (textured ? "TEXTURÉ" : "SIMPLE") << "\n";
    }

    // ── Détection des coins ─────────────────────────────────
    cout << "\nDétection des coins ("
         << (textured ? "Hough – fond texturé" : "Contours Canny – fond simple")
         << ")...\n";

    vector<Point2f> corners = textured
        ? detectTexturedBackground(img)
        : detectSimpleBackground(img);

    // Fallback : si l'une échoue, essayer l'autre
    if (corners.size() != 4) {
        cout << "  Méthode principale échouée, essai de la méthode alternative...\n";
        corners = textured
            ? detectSimpleBackground(img)
            : detectTexturedBackground(img);
    }

    // Fallback ultime : coins de l'image entière
    if (corners.size() != 4) {
        cout << "  Détection échouée — utilisation des coins de l'image.\n";
        float W = img.cols - 1, H = img.rows - 1;
        corners = { {0,0}, {W,0}, {W,H}, {0,H} };
    }

    // ── Affichage du résultat de détection ──────────────────
    Mat detected = img.clone();
    drawCorners(detected, corners);
    putText(detected, "Detection automatique - appuyez sur une touche",
            Point(10,25), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,255,0), 1);
    imshow("Detection des coins", detected);
    cout << "\nRésultat de la détection affiché. Appuyez sur une touche...\n";
    waitKey(0);
    destroyWindow("Detection des coins");

    // ── Ajustement interactif ───────────────────────────────
    g_state.corners  = corners;
    g_state.dragging = -1;
    g_state.display  = img.clone();

    Mat interactive = img.clone();
    drawCorners(interactive, corners);
    putText(interactive, "Ajustez les coins. [ENTREE] = appliquer  [R] = reset  [Q] = quitter",
            Point(10,25), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,255,0), 1);

    namedWindow("Ajustement interactif");
    setMouseCallback("Ajustement interactif", onMouse);
    imshow("Ajustement interactif", interactive);

    cout << "\nAjustement interactif :\n"
         << "  - Glissez les coins verts pour ajuster\n"
         << "  - [ENTREE] pour appliquer la transformation\n"
         << "  - [R] pour réinitialiser\n"
         << "  - [Q] pour quitter\n";

    vector<Point2f> savedCorners = corners;

    while (true) {
        int key = waitKey(30) & 0xFF;

        if (key == 13 || key == 10) { // ENTREE
            break;
        }
        else if (key == 'r' || key == 'R') {
            g_state.corners = savedCorners;
            Mat tmp = g_state.display.clone();
            drawCorners(tmp, g_state.corners);
            putText(tmp, "Ajustez les coins. [ENTREE] = appliquer  [R] = reset  [Q] = quitter",
                    Point(10,25), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255,255,0), 1);
            imshow("Ajustement interactif", tmp);
        }
        else if (key == 'q' || key == 'Q') {
            cout << "Annulé.\n";
            return 0;
        }
    }
    destroyWindow("Ajustement interactif");

    // ── Transformation de perspective ───────────────────────
    cout << "\nApplication de la transformation de perspective...\n";
    Mat warped = applyPerspective(img, g_state.corners);

    imshow("Resultat - vue de dessus", warped);
    cout << "Transformation terminée. Appuyez sur une touche pour sauvegarder.\n";
    waitKey(0);

    // Sauvegarder le résultat
    string outPath = imgPath.substr(0, imgPath.find_last_of('.')) + "_rectified.jpg";
    imwrite(outPath, warped);
    cout << "Image sauvegardée : " << outPath << "\n";

    waitKey(0);
    return 0;
}
