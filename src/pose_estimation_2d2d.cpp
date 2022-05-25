#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/features2d/features2d.hpp>

/**
 * @brief 根据匹配好的特征点集以及预先获取的相机内参获取相机外参，即旋转矩阵和平移向量
 *
 * @param keypoints_1 输入：特征点集1
 * @param keypoints_2 输入：特征点集2
 * @param matches 输入：图一图二的特征点匹配
 * @param R 输出：旋转矩阵
 * @param t 输出：平移向量
 */
void pose_estimation_2d2d(
    std::vector<cv::KeyPoint> keypoints_1,
    std::vector<cv::KeyPoint> keypoints_2,
    std::vector<cv::DMatch> matches,
    cv::Mat &R, cv::Mat &t)
{
    // 相机内参，这里用的是书上给的TUM Freiburg的内参，后面网购的标定板到了之后换成自己手机的内参
    // 520.9       0    325.1
    //     0   521.0    249.7
    //     0       0        1
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // 把匹配点从cv::KeyPoint转为cv::Point2f
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (auto &match : matches)
    {
        // std::cout << "match.queryIdx: " << match.queryIdx <<"  match.trainIdx:" << match.trainIdx << std::endl;
        points1.emplace_back(keypoints_1[match.queryIdx].pt);
        points2.emplace_back(keypoints_2[match.trainIdx].pt);
    }

    // 计算基础矩阵
    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    std::cout << "基础矩阵fundamental matrix: " << fundamental_matrix << std::endl;

    // 计算本质矩阵
    cv::Point2d principal_point(325.1, 249.7); // 光心，用的书上给的TUM标定值
    int focal_length = 521;
    cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point, cv::RANSAC);
    std::cout << "本质矩阵essential matrix: " << essential_matrix << std::endl;

    // 计算单应矩阵，最后两个参数: 迭代轮数 and 置信度
    cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 3, cv::noArray(), 2000, 0.99);

    // 从本质矩阵中恢复旋转和平移矩阵
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);

    std::cout<< "旋转矩阵R:" << R << std::endl;
    std::cout<< "平移向量t:" << t << std::endl;
}

/**
 * @brief 根据输入的两张图片获取特征点以及匹配信息
 * 
 * @param img_1 输入：图片矩阵1
 * @param img_2 输入：图片矩阵2
 * @param keypoints_1 输出：图一的特征点集
 * @param keypoints_2 输出：图二的特征点集
 * @param matches 输出：特征点的匹配
 */
void find_feature_matches(
    cv::Mat img_1,
    cv::Mat img_2,
    std::vector<cv::KeyPoint> &keypoints_1,
    std::vector<cv::KeyPoint> &keypoints_2,
    std::vector<cv::DMatch> &matches)
{
    // vector<KeyPoint> keypoints_1, keypoints_2; // 图1的特征点集， 图2的特征点集
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20); // 均为默认参数 500个特征点
    orb->detect(img_1, keypoints_1);
    orb->detect(img_2, keypoints_2);
    orb->compute(img_1, keypoints_1, descriptors_1);
    orb->compute(img_2, keypoints_2, descriptors_2);
    std::vector<cv::DMatch> total_matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, total_matches);

    // 第四步，匹配点对筛选
    double min_dist = 10000, max_dist = 0;
    for (auto &match : total_matches)
    {
        double dist = match.distance;
        min_dist = std::min(dist, min_dist);
        max_dist = std::max(dist, max_dist);
    }

    std::cout << "max dist: " << max_dist << "  min dist: " << min_dist << std::endl;

    // 筛选出优质匹配
    std::vector<cv::DMatch> good_matches;
    for (auto &match : total_matches)
    {
        if (match.distance <= std::max(2 * min_dist, 30.0))
            good_matches.emplace_back(match);
    }
    matches = good_matches;
}
/**
 * @brief 多目相机姿态估计，孙煜晗个人学习代码，参考：《视觉SLAM十四讲》
 *
 * @param argc 参数为3时正常运行
 * @param argv agrv[1]图片1文件路径，argv[2]图片2文件路径
 * @return int
 */
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "需要输入2张对应图片: pose_estimation_2d2d img1 img2";
        return 1;
    }
    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "找到了" << matches.size() << "组匹配点" << std::endl;
    cv::Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);
}