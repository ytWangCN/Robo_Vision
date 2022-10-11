#ifndef __DNN_
#define __DNN_

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class Dnn_NumDetect
{
public:
    Dnn_NumDetect(const string& path);
    // 运行前向传递以计算图层的输出
    Point2f forward(Mat& src);
private:
    dnn::Net Lenet5;

    // 加载onnx模型
    void loadModel(const string& path);
    // 矩阵归一化
    void Mat_Normalization(Mat &matrix);
};

#endif
