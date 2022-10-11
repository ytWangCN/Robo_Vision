#ifndef __LENET5_
#define __LENET5_

#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct LeNet5 : torch::nn::Module
{
    LeNet5(){};
    LeNet5(int num_class, int padding);

    torch::Tensor forward(torch::Tensor X);

    torch::nn::Conv2d C1{nullptr};
    torch::nn::Conv2d C2{nullptr};

    torch::nn::Linear FC1{nullptr};
    torch::nn::Linear FC2{nullptr};
    torch::nn::Linear FC3{nullptr};

    void Train();
    void Test(const string& address);
    void SaveModel(torch::serialize::OutputArchive& archive, const string& path);
};


#endif
