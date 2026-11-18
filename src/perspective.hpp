#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Applique la transformation de perspective.
 * Les 4 coins (ordre TL, TR, BR, BL) sont mappés sur un rectangle.
 * La taille de sortie est calculée à partir des longueurs des côtés.
 */
cv::Mat applyPerspective(const cv::Mat& src,
                         const std::vector<cv::Point2f>& corners);
