#include "interactive.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

// ── État interne (accessible par le callback souris) ──────────

namespace {
    struct MouseState {
        vector<Point2f> corners;
        int             dragging = -1;
        Mat             baseImg;  // image sans dessins
    } g;
}

// ── Dessin ────────────────────────────────────────────────────

void drawCorners(Mat& img, const vector<Point2f>& pts, bool labels)
{
    if (pts.size() != 4) return;

    // Quadrilatère
    for (int i = 0; i < 4; i++)
        line(img, pts[i], pts[(i+1)%4], Scalar(0, 200, 255), 2);

    // Cercles et étiquettes
    const char* names[] = {"TL","TR","BR","BL"};
    for (int i = 0; i < 4; i++) {
        circle(img, pts[i], 10, Scalar(0, 255, 0), -1);
        circle(img, pts[i], 12, Scalar(0, 0, 0),   2);
        if (labels)
            putText(img, names[i], pts[i] + Point2f(14, -8),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
    }
}

static void redraw(const string& winName)
{
    Mat tmp = g.baseImg.clone();
    drawCorners(tmp, g.corners);
    putText(tmp, "[ENTREE] appliquer  [R] reset  [Q] quitter",
            Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 255, 0), 1);
    imshow(winName, tmp);
}

// ── Callback souris ───────────────────────────────────────────

static void onMouse(int event, int x, int y, int /*flags*/, void* data)
{
    const string& winName = *static_cast<string*>(data);
    Point2f cur(x, y);

    if (event == EVENT_LBUTTONDOWN) {
        float best = 20.0f;
        g.dragging = -1;
        for (int i = 0; i < 4; i++) {
            float d = dist(g.corners[i], cur);
            if (d < best) { best = d; g.dragging = i; }
        }
    }
    else if (event == EVENT_MOUSEMOVE && g.dragging >= 0) {
        g.corners[g.dragging] = cur;
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

    vector<Point2f> savedCorners = initial;

    namedWindow(winName);
    setMouseCallback(winName, onMouse, (void*)&winName);
    redraw(winName);

    cout << "\nAjustement interactif :\n"
         << "  Glissez les coins verts pour ajuster\n"
         << "  [ENTREE] pour appliquer  |  [R] reset  |  [Q] quitter\n";

    while (true) {
        int key = waitKey(30) & 0xFF;
        if (key == 13 || key == 10) break;          // ENTREE
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
