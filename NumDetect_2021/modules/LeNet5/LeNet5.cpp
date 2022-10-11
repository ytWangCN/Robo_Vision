#include "LeNet5.h"

const float LEARNING_RATE = 0.01;
const int EPOCHES = 20;

/**
 * @brief LeNet5 构造函数
 *        共七层
 * @note  
 */
LeNet5::LeNet5(int num_class, int padding)
{
    this->C1 = register_module("C1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, {5, 5}).padding(padding)));
    this->C2 = register_module("C2", torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, {5, 5})));
    this->FC1 = register_module("FC1", torch::nn::Linear(torch::nn::LinearOptions(400, 120)));
    this->FC2 = register_module("FC2", torch::nn::Linear(torch::nn::LinearOptions(120, 84)));
    this->FC3 = register_module("FC3", torch::nn::Linear(torch::nn::LinearOptions(84, num_class)));
}

/**
 * @brief 前馈函数，计算权重
 */
torch::Tensor LeNet5::forward(torch::Tensor X)
{
    //卷积池化层
    X = this->C1->forward(X);
    X = torch::nn::functional::relu(X);

    //最大池化
    X = torch::max_pool2d(X, {2, 2}, 2);

    X = this->C2->forward(X);
    X = torch::nn::functional::relu(X);

    //最大池化
    X = torch::max_pool2d(X, {2, 2}, 2);

    //全连接层
    X = X.view({X.size(0), -1});
    X = FC1->forward(X);
    X = torch::nn::functional::relu(X);

    X = FC2->forward(X);
    X = torch::nn::functional::relu(X);

    X = FC3->forward(X);
    return X;
}

/**
 * @brief 训练模型
 */
void LeNet5::Train()
{
    auto dataset = torch::data::datasets::MNIST("../data/")
                       .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                       .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(std::move(dataset));

    printf("Load MNIST handwriting database successfully!\n");

    auto criterion = torch::nn::CrossEntropyLoss();
    auto optimizer = torch::optim::SGD(this->parameters(), torch::optim::SGDOptions(0.005).momentum(0.9));

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < EPOCHES; ++i)
    {
        int count = 0;
        auto running_loss = 0.0;
        for (torch::data::Example<> &batch : *data_loader)
        {
            auto inputs = batch.data;
            auto labels = batch.target;

            //将梯度清零
            optimizer.zero_grad();

            //前向传播
            auto output = this->forward(inputs);
            //计算损失函数
            auto loss = criterion(output, labels);
            cout << "calculating Loss……" << endl;
            //反向传播
            loss.backward();
            cout << "backward ……" << endl;
            optimizer.step();
            cout << "Steping ……" << endl;

            running_loss += loss.item().toFloat();
        }
        printf("Finsh %d epoch, Loss: %6f", i + 1, running_loss);
        auto end = std::chrono::system_clock::now();
    }
}

/**
 * @brief 测试模型
 */
void LeNet5::Test(const string& address)
{
    auto start = std::chrono::system_clock::now();
    auto dataset = torch::data::datasets::MNIST("../data/")
                       .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                       .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(std::move(dataset));

    printf("Load MNIST handwriting database successfully!\n");

    torch::serialize::InputArchive archive;
    archive.load_from(address);

    this->load(archive);

    int total = 0, correct = 0;
    for (torch::data::Example<> &batch : *data_loader)
    {
        // 用训练好的网络处理测试数据
        auto outputs = this->forward(batch.data);
        // 得到预测值，0 ~ 9
        auto predicted = torch::max(outputs, 1);
        // 获取标签数据， 0 ~ 9
        auto labels = batch.target;
        cout << "Testing of Num" << total << endl;
        // 比较预测结果和实际结果，并更新统计结果
        if (labels[0].item<int>() == std::get<1>(predicted).item<int>())
            correct++;
        total++;
    }

    auto end = std::chrono::system_clock::now();

    // printf("Total test items: %d, passed test items: %d, pass rate: %.3f%%, cost %lld msec.\n",
    //        total, correct, correct * 100.f / total,
    //        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

/**
 * @brief 保存模型
 * @param archive libtorch中模型输出类
 * @param path 保存路径
 */
void LeNet5::SaveModel(torch::serialize::OutputArchive& archive, const string &path)
{
    this->save(archive);
    archive.save_to(path);
}

