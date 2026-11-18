#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Retourne true si le fond est texturé (variance Laplacien élevée).
 * Justification : variance élevée → beaucoup de détails → fond complexe.
 */
bool isTexturedBackground(const cv::Mat& img);

/**
 * Méthode FOND SIMPLE : Canny + findContours + approxPolyDP
 * Justification : fort contraste doc/fond → gradient net sur les bords →
 * le plus grand quadrilatère convexe est le document.
 */
std::vector<cv::Point2f> detectSimpleBackground(const cv::Mat& img);

/**
 * Méthode FOND TEXTURÉ : HoughLinesP (longues lignes droites)
 * Justification : Canny seul donne trop de bruit sur fond complexe.
 * HoughLinesP filtre les longues lignes formant les bords du document.
 */
std::vector<cv::Point2f> detectTexturedBackground(const cv::Mat& img);

/**
 * Wrapper : choisit la bonne méthode selon le type de fond,
 * avec fallback sur l'autre méthode et sur les coins de l'image.
 */
std::vector<cv::Point2f> detectCorners(const cv::Mat& img, bool textured);

