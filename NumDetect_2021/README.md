# 使用LeNet5进行数字识别，并通过OpenVINO进行加速

### 使用说明

+ 环境依赖**pytorch、libtorch**，在第一级目录下运行**update.sh**
    
  ```
  ./update.sh
  ```

+ 使用OpenVINO加速推理

  ```shell
  cd pytorch
  python3 PTHConvertToONNX.py

  cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites
  sudo ./install_prerequisites_onnx.sh

  sudo python3 mo_onnx.py --input_model=/wyt/NumDetect_2021/model/NumDetect.onnx

  sudo mv NumDetect.* /wyt/NumDetect_2021/model/
  # 注意更换路径
  ```


### 文件结构

.  
├── CMakeLists.txt  
├── main.cpp  
├── model  
│   ├── NumDetect.onnx  
│   ├── NumDetect.pt  
│   └── Vision_NumDetect.pth  
├── modules  
│   ├── DNN  
│   │   ├── CMakeLists.txt  
│   │   ├── Dnn.cpp  
│   │   └── include  
│   │       └── Dnn.h  
│   ├── LeNet5  
│   │   ├── CMakeLists.txt  
│   │   ├── include  
│   │   │   └── LeNet5.h  
│   │   └── LeNet5.cpp  
│   └── Libtorch  
│       ├── CmakeLists.txt  
│       ├── CMakeLists.txt  
│       ├── include  
│       │   └── libtorch.h  
│       └── libtorch.cpp  
├── pytorch  
│   ├── CreateTXT.py  
│   ├── LeNet5.py  
│   ├── Mydatasets.py  
│   ├── PTHConvertToONNX.py  
│   ├── __pycache__  
│   │   ├── LeNet5.cpython-36.pyc  
│   │   └── Mydatasets.cpython-36.pyc  
│   ├── TestNum.txt  
│   ├── Test.py  
│   ├── TrainNum.txt  
│   └── Train.py  
└── README.md  

10 directories, 26 files

### 迭代过程

+ v1.0
  采用pytorch和libtorch两种方法进行模型训练
  * libtorch的效果不如pytorch理想，最终采用使用pytorch训练模型并转为pt模型，再以libtorch调用
  * 使用MNIST数据集测试,在没有预处理的情况下，处理速度约为20ms

+ v1.1
  创建记录img和label的txt文件，加载装甲板数字的数据集，预处理后传入LeNet5训练
  * 准确率可达97%+，处理速度15-20ms

+ v2.0
  使用dnn module加载onnx模型
  * onnx的优点在于处理速度快，仅需1~2ms

  **@Todo** 考虑使用**Mixup**

### 开发者

特别感谢杨泽霖（scut.bigeyoung@qq.com）对串口模块的贡献
Copyright South China Tiger(c) 2021

