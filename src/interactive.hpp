#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Dessine le quadrilatère et les coins sur l'image
void drawCorners(cv::Mat& img, const std::vector<cv::Point2f>& pts, bool labels = true);

/**
 * Lance la fenêtre d'ajustement interactif.
 * L'utilisateur peut glisser les coins verts.
 * Retourne les coins ajustés après [ENTREE], ou vide si [Q].
 */
std::vector<cv::Point2f> interactiveAdjust(const cv::Mat& img,
                                           const std::vector<cv::Point2f>& initial);
