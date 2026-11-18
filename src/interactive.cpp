#include "interactive.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

// Marge (pixels) autour de l'image pour accéder aux coins hors-bords
static const int BORDER = 120;

// ── État interne ──────────────────────────────────────────────
namespace {
    struct MouseState {
        vector<Point2f> corners;  // coords dans l'image originale
        int    dragging = -1;
        Mat    baseImg;           // image originale (sans padding)
        int    W, H;              // taille image originale
    } g;
}

// ── Conversion coords image ↔ coords fenêtre (avec padding) ──
static Point2f imgToWin(Point2f p)  { return {p.x + BORDER, p.y + BORDER}; }
static Point2f winToImg(Point2f p)  { return {p.x - BORDER, p.y - BORDER}; }

// ── Dessin ────────────────────────────────────────────────────
void drawCorners(Mat& img, const vector<Point2f>& pts, bool labels)
{
    if (pts.size() != 4) return;

    for (int i = 0; i < 4; i++)
        line(img, pts[i], pts[(i+1)%4], Scalar(0, 200, 255), 2);

    const char* names[] = {"TL","TR","BR","BL"};
    for (int i = 0; i < 4; i++) {
        circle(img, pts[i], 10, Scalar(0, 255, 0), -1);
        circle(img, pts[i], 12, Scalar(0,  0,  0), 2);
        if (labels)
            putText(img, names[i], pts[i] + Point2f(14, -8),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
    }
}

static void redraw(const string& winName)
{
    // Créer image avec padding gris foncé
    Mat padded;
    copyMakeBorder(g.baseImg, padded,
                   BORDER, BORDER, BORDER, BORDER,
                   BORDER_CONSTANT, Scalar(50, 50, 50));

    // Convertir les coins en coordonnées fenêtre
    vector<Point2f> winPts;
    for (auto& p : g.corners) winPts.push_back(imgToWin(p));

    // Dessiner sur l'image paddée
    drawCorners(padded, winPts);

    // Dessiner une bordure blanche pour délimiter l'image réelle
    rectangle(padded,
              Point(BORDER, BORDER),
              Point(BORDER + g.W - 1, BORDER + g.H - 1),
              Scalar(180, 180, 180), 1);

    // Instructions
    putText(padded, "[ENTREE] appliquer  [R] reset  [Q] quitter",
            Point(BORDER, BORDER - 10),
            FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 255, 0), 1);
    putText(padded, "Glissez les coins verts (zone grise = hors image = OK)",
            Point(BORDER, BORDER - 30),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 200, 200), 1);

    imshow(winName, padded);
}

// ── Callback souris ───────────────────────────────────────────
static void onMouse(int event, int x, int y, int /*flags*/, void* data)
{
    const string& winName = *static_cast<string*>(data);

    // Position dans l'espace image (peut être négative ou hors image)
    Point2f imgPos = winToImg(Point2f(x, y));

    if (event == EVENT_LBUTTONDOWN) {
        float best = 25.0f;  // rayon de capture élargi
        g.dragging = -1;
        for (int i = 0; i < 4; i++) {
            float d = dist(g.corners[i], imgPos);
            if (d < best) { best = d; g.dragging = i; }
        }
    }
    else if (event == EVENT_MOUSEMOVE && g.dragging >= 0) {
        // Pas de clamp : on laisse sortir de l'image (zone padding)
        g.corners[g.dragging] = imgPos;
        redraw(winName);
    }
    else if (event == EVENT_LBUTTONUP) {
        g.dragging = -1;
    }
}

// ── Interface publique ────────────────────────────────────────
vector<Point2f> interactiveAdjust(const Mat& img, const vector<Point2f>& initial)
{
    const string winName = "Ajustement interactif";

    g.baseImg  = img.clone();
    g.corners  = initial;
    g.dragging = -1;
    g.W        = img.cols;
    g.H        = img.rows;

    vector<Point2f> savedCorners = initial;

    namedWindow(winName, WINDOW_AUTOSIZE);
    setMouseCallback(winName, onMouse, (void*)&winName);
    redraw(winName);

    cout << "\nAjustement interactif :\n"
         << "  Glissez les coins verts (zone grise = extensions hors image)\n"
         << "  [ENTREE] pour appliquer  |  [R] reset  |  [Q] quitter\n";

    while (true) {
        int key = waitKey(30) & 0xFF;
        if (key == 13 || key == 10) break;
        if (key == 'r' || key == 'R') {
            g.corners = savedCorners;
            redraw(winName);
        }
        if (key == 'q' || key == 'Q') {
            destroyWindow(winName);
            return {};
        }
    }

    destroyWindow(winName);
    return g.corners;
}
