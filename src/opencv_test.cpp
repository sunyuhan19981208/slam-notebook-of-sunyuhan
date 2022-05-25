#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
using namespace std;
using namespace cv;
void Rotate(Mat &&srcImage, Mat &destImage, double angle) {
    Point2f center(srcImage.cols / 2, srcImage.rows / 2);
    auto M = getRotationMatrix2D(center, angle, 1);
    warpAffine(srcImage, destImage, M, Size(srcImage.cols, srcImage.rows));
    circle(destImage, center, 2, Scalar(255, 0, 0));
}
int main()
{
    cv::Mat img_encode;
    img_encode = imread("/Users/sunyuhan/Desktop/Unknown.png", IMREAD_COLOR);
    cout << img_encode.rows << " " << img_encode.cols << endl;
    auto img_roi = img_encode({200,200,400,400}); 
    cv::Mat dest_img;
    ::Rotate(std::move(img_roi), dest_img, 30);
    // vector<uchar> data_encode;
    // imencode(".png", img_encode, data_encode);
    // string str_encode(data_encode.begin(), data_encode.end());
    //cout << str_encode << endl;

    // cv::Mat img_decode;
    // vector<uchar> data(str_encode.begin(), str_encode.end());
    // img_decode = imdecode(data, CV_LOAD_IMAGE_COLOR);
    // imshow("pic", img_encode);
    imwrite("./pic.png", dest_img);
    // cvWaitKey(10000);
    
    
    cout << "Hello" << endl;
}
