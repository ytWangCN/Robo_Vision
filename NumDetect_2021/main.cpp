#include <torch/script.h> 
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "RMVideoCapture.h"
#include "LeNet5.h"
#include "libtorch.h"
#include "Dnn.h"
#include <iostream>

#undef UNICODE

using namespace std;
using namespace cv;

const char *keys = "{ help        |         |打印帮助信息}"
                   "{ input       |         |输入测试图片的路径}"
                   "{ cppLeNet5   |         |以libtorch构建lenet5网络进行训练和测试，最终生成pt模型}"
                   "{ torch       |         |以torch加载pt模型的方式运行}"
                   "{ dnn         |         |以opencv的Dnn加载ONNX模型方式运行}"
                   "{ camera      |         |开启相机}"
                   "{ expose      |         |曝光值}"
                   "{ test        |         |测试数据集}";

void test(int numclasses, int datanum);

int main(int argc, const char *argv[])
{
  /**
   * @note 用libtorch调用pytorch生成的模型转换后的pt模型
   */
  CommandLineParser parser(argc, argv, keys);

  if (parser.has("help")) // **打印帮助信息**
  {
    parser.printMessage();
    return 0;
  }

  double start_time, end_time, time;

  Mat src, dst;
  Ptr<VideoCapture> capture;

  torch::jit::script::Module module = torch_loadModel("../model/NumDetect.pt");
  Dnn_NumDetect dnnDetect("../model/NumDetect.onnx");

  if (parser.has("camera"))
  {
    int expose = parser.get<int>("expose");
    capture = (new RMVideoCapture());
    capture->set(CAP_PROP_EXPOSURE, expose); //设置曝光
    while (capture->read(src))
    {
      start_time = static_cast<double>(getTickCount()); // 获取开始执行时间
      resize(src, dst, Size(28, 28), 0, 0);
      Point2f result = dnnDetect.forward(dst);
      end_time = static_cast<double>(getTickCount());
      time = (end_time - start_time) / getTickFrequency() * 1000;
      cout << "每帧运行时间为: " << time << " ms" << endl;
      cout << "result:\t" << result << endl;
      imshow("Origin", src);
      if (waitKey(10) == 27)
      {
        if (waitKey(0) == 27)
        {
          return -1;
        }
      }
    }
  }
  else if (parser.has("test"))
  {
    test(5, 2000);
    return -1;
  }
  else
    src = imread(parser.get<string>("input"), IMREAD_GRAYSCALE); //读取灰度图

  start_time = static_cast<double>(getTickCount()); // 获取开始执行时间

  /**
   * @note 用libtorch调用C++直接定义的模型
   */
  if (parser.has("cppLeNet5"))
  {
    LeNet5 net(10, 2);
    string modelAddress = "../model/net.pt";

    // 训练模型
    net.Train();

    // 保存模型
    torch::serialize::OutputArchive archive;
    net.SaveModel(archive, modelAddress);

    // 测试模型
    net.Test(modelAddress);
  }
  /**
   * @note 用pytorch训练出pth模型后转换成pt模型，在torch加载
   */
  else if (parser.has("torch"))
  {
    resize(src, dst, Size(28, 28), 0, 0);
    auto result = torchForward(module, dst);
    cout << "检测结果为：" << result << endl;
  }
  /**
   * @note 用Opencv的dnn module 加载onnx模型
   */
  else if (parser.has("dnn"))
  {
    resize(src, dst, Size(100, 100), 0, 0);
    Point2f result = dnnDetect.forward(dst);
    cout << result << endl;
  }
  else
  {
    parser.printMessage();
    return 0;
  }

  end_time = static_cast<double>(getTickCount());
  time = (end_time - start_time) / getTickFrequency() * 1000;
  cout << "每帧运行时间为: " << time << " ms" << endl;

  imshow("origin", dst);

  if (waitKey(0))
    return 0;
}

void test(int numclasses, int datanum)
{
  Dnn_NumDetect dnnDetect("../model/NumDetect.onnx");
  Mat src;
  string filename;
  double start_time, end_time, time = 0.f;
  double sumTime = 0.f;
  int correct = 0;
  for (int i = 1; i <= numclasses; i++)
  {
    for (int j = 0; j < datanum; j++)
    {
      filename = "../../TestData/Numbers/" + to_string(i) + "/" + to_string(j) + ".jpg";
      src = imread(filename, IMREAD_GRAYSCALE);
      start_time = static_cast<double>(getTickCount()); // 获取开始执行时间
      resize(src, src, Size(28, 28), 0, 0);
      if (dnnDetect.forward(src).x == i)
        correct++;
      end_time = static_cast<double>(getTickCount());
      time = (end_time - start_time) / getTickFrequency() * 1000;
      sumTime += time;
    }
  }
  cout << "平均运行时间为：\t" << sumTime / (float)(numclasses * datanum) << " ms" << endl;
  cout << "正确率为:\t" << (float)correct / (float)(numclasses * datanum) << endl;
}        