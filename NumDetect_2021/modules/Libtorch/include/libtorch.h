#ifndef __TORCH_
#define __TORCH_

#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


Mat PreTreating(const Mat &src);

torch::jit::script::Module torch_loadModel(const string &path);

int torchForward(torch::jit::script::Module& module, const Mat& src);


#endif


