#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Trie 4 points dans l'ordre : TL, TR, BR, BL
std::vector<cv::Point2f> sortCorners(std::vector<cv::Point2f> pts);

// Distance euclidienne entre deux points
float dist(cv::Point2f a, cv::Point2f b);

// Intersection de deux droites infinies (retourne false si parallèles)
bool lineIntersect(cv::Point2f p1, cv::Point2f p2,
                   cv::Point2f p3, cv::Point2f p4,
                   cv::Point2f& out);
