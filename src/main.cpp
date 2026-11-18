/**
 * Devoir 2 — Transformation de Perspective
 * main.cpp : point d'entrée uniquement
 */

#include <opencv2/opencv.hpp>
#include <iostream>

#include "detection.hpp"
#include "interactive.hpp"
#include "perspective.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // ── Lecture de l'image ──────────────────────────────────
    string imgPath;
    if (argc >= 2) {
        imgPath = argv[1];
    } else {
        cout << "Usage: " << argv[0] << " <image> [simple|texture]\n";
        cout << "Chemin de l'image : ";
        cin >> imgPath;
    }

    Mat img = imread(imgPath);
    if (img.empty()) {
        cerr << "Erreur : impossible de lire \"" << imgPath << "\"\n";
        return 1;
    }

    // Redimensionner si trop grande
    const int maxDim = 900;
    if (img.cols > maxDim || img.rows > maxDim) {
        float s = (float)maxDim / max(img.cols, img.rows);
        resize(img, img, Size(), s, s);
    }

    // ── Choix de la méthode ─────────────────────────────────
    bool textured;
    if (argc >= 3 && string(argv[2]) == "texture") {
        textured = true;
    } else if (argc >= 3 && string(argv[2]) == "simple") {
        textured = false;
    } else {
        cout << "\nDétection automatique du type de fond...\n";
        textured = isTexturedBackground(img);
        cout << "  → Fond : " << (textured ? "TEXTURÉ" : "SIMPLE") << "\n";
    }

    // ── Détection automatique des coins ─────────────────────
    cout << "\nDétection des coins...\n";
    vector<Point2f> corners = detectCorners(img, textured);

    // ── Affichage du résultat de détection ──────────────────
    Mat detected = img.clone();
    drawCorners(detected, corners);
    putText(detected, "Detection automatique - appuyez sur une touche",
            Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 255, 0), 1);
    imshow("Detection des coins", detected);
    waitKey(0);
    destroyWindow("Detection des coins");

    // ── Ajustement interactif ───────────────────────────────
    vector<Point2f> adjusted = interactiveAdjust(img, corners);
    if (adjusted.empty()) {
        cout << "Annulé.\n";
        return 0;
    }

    // ── Transformation de perspective ───────────────────────
    cout << "\nApplication de la transformation...\n";
    Mat warped = applyPerspective(img, adjusted);

    imshow("Resultat - vue de dessus", warped);
    waitKey(0);

    // Sauvegarde
    string outPath = imgPath.substr(0, imgPath.find_last_of('.')) + "_rectified.jpg";
    imwrite(outPath, warped);
    cout << "Image sauvegardée : " << outPath << "\n";

    waitKey(0);
    return 0;
}

