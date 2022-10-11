#include "Dnn.h"

/**
 * @brief   使用Opencv Dnn Module 读取ONNX模型
 * @note    如果OpenCV是使用Intel的推理引擎库编译的，则DNN_BACKEND_DEFAULT表示DNN_BACKEND_INFERENCE_ENGINE。 
 *          否则表示DNN_BACKEND_OPENCV。 
 */
Dnn_NumDetect::Dnn_NumDetect(const string &path)
{
    this->loadModel(path);
    //网络在支持的地方使用特定的计算后端
    this->Lenet5.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    //网络在特定的目标设备上进行计算
    this->Lenet5.setPreferableTarget(dnn::DNN_TARGET_CPU);
}

/**
 * @brief   加载ONNX模型
 */
void Dnn_NumDetect::loadModel(const string &path)
{
    this->Lenet5 = dnn::readNetFromONNX(path);
    CV_Assert(!this->Lenet5.empty());
}

/**
 * @brief   运行前向传递以计算图层的输出
 * @return  指定层的第一个输出的Blob。
 */
Point2f Dnn_NumDetect::forward(Mat &src)
{
    CV_Assert(!this->Lenet5.empty());
    // 设置输入
    Mat input;
    input = dnn::blobFromImage(src);
    this->Lenet5.setInput(input);

    Mat prob = this->Lenet5.forward();
    // cout << prob <<endl;
    // 矩阵归一化
    this->Mat_Normalization(prob);
    // cout << prob <<endl;
    Point classIdPoint;
    double confidence;
    //查找最大值和最小值
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    return Point2f(classId, confidence);
}

/**
 * @brief 对矩阵中的元素进行relu和归一化
 * @param matrix 输入矩阵
 */
void Dnn_NumDetect::Mat_Normalization(Mat &matrix)
{
    float sum = 0.f;
    for (auto it = matrix.begin<float>(); it != matrix.end<float>(); it++)
    {
        if((*it) > 0)
            sum += (*it);
        else
            (*it) = 0.f;
    }
    for (auto it = matrix.begin<float>(); it != matrix.end<float>(); it++)
    {
        (*it) /= sum;
    }
}
