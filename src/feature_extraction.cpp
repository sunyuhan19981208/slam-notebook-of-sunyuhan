#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
/**
 * @brief 提取特征并匹配，孙煜晗个人学习代码，参考：《视觉SLAM十四讲》
 *
 * @param argc 参数为3时正常运行
 * @param argv agrv[1]图片1文件路径，argv[2]图片2文件路径
 * @return int
 */
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "需要输入2张对应图片: feature_extraction img1 img2";
        return 1;
    }
    Mat img_1 = imread(argv[1], IMREAD_COLOR); // 原文使用的CV_LOAD_IMAGE_COLOR，新版OPENCV不支持
    Mat img_2 = imread(argv[2], IMREAD_COLOR);
    // 初始化
    vector<KeyPoint> keypoints_1, keypoints_2; // 图1的特征点集， 图2的特征点集
    Mat descriptors_1, descriptors_2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20); // 均为默认参数 500个特征点

    // 第一步，检测Oriented FAST角点
    orb->detect(img_1, keypoints_1);
    orb->detect(img_2, keypoints_2);
    // 打印FAST角点坐标
    // for(auto &kp : keypoints_1) {
    //     cout<< "x: " << kp.pt.x <<"  y: "<< kp.pt.y<<endl;
    // }

    // 第二步，根据角点位置计算BRIEF描述子 
    orb->compute(img_1, keypoints_1, descriptors_1);
    orb->compute(img_2, keypoints_2, descriptors_2);
    // 500 * 32的描述矩阵
    // cout << "rows: " << descriptors_1.rows << "cols: " << descriptors_1.cols <<endl;

    // 特征点显示的结果
    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // imshow("ORB特征点", outimg1);
    // auto filename = "/Users/sunyuhan/Code/cpp/cv/result/orb.png";
    // imwrite(filename, outimg1);

    // 第三步，对两幅图像的描述子进行匹配，使用Hamming距离
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);

    // 第四步，匹配点对筛选
    double min_dist = 10000, max_dist = 0;
    for(auto &match : matches) {
        double dist = match.distance;
        min_dist = min(dist, min_dist);
        max_dist = max(dist, max_dist);
    }

    cout << "max dist: " << max_dist << "  min dist: "<<min_dist;

    // 筛选出优质匹配
    vector<DMatch> good_matches;
    for(auto &match : matches) {
        if(match.distance <= max(2 * min_dist, 30.0))
            good_matches.emplace_back(match);
    }
    
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);

    imwrite("/Users/sunyuhan/Code/cpp/cv/result/match.png", img_match);
    imwrite("/Users/sunyuhan/Code/cpp/cv/result/goodmatch.png", img_goodmatch);
}