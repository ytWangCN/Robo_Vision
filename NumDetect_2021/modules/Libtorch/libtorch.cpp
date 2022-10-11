#include "libtorch.h"


/**
 * @brief 使用libtorch加载pt模型
 * @note  加载错误会输出错误信息
 */
torch::jit::script::Module torch_loadModel(const string &path)
{
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        return torch::jit::load(path);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
    }
}

/**
 * @brief  使用torch将图片传入模型中进行预测
 * @return 返回预测结果
 */
int torchForward(torch::jit::script::Module &module, const Mat &src)
{
    std::vector<int64_t> sizes = {1, 1, src.rows, src.cols};
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor tensor_image = torch::from_blob(src.data, at::IntList(sizes), options); //将opencv的图像数据转为Tensor张量数据
    tensor_image = tensor_image.toType(at::kFloat);                                    //转为浮点型张量数据
    at::Tensor result = module.forward({tensor_image}).toTensor();

    auto max_result = result.max(1, true);
    int max_index = std::get<1>(max_result).item<int>();

    return max_index;
}

/**
 * @brief 预处理 (追求速度时可以不用)
 */
Mat PreTreating(const Mat &src)
{
    Mat dst;

    cvtColor(src, dst, COLOR_RGB2GRAY);
    //高斯模糊
    GaussianBlur(dst, dst, Size(5, 5), 0);
    //二值化
    threshold(src, dst, 0, 255, THRESH_BINARY | THRESH_OTSU);
    Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));
    dilate(dst, dst, element);
    return dst;
}
