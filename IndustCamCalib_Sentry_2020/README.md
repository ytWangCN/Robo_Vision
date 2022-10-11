# MindVision工业相机标定
用opencv库对工业相机进行标定，标定程序opencv官方本来就有，但是mindvision的相机打开要用的其官方的API，所以opencv官方的程序不可以直接使用，这份代码结合了mindvison的API和更改了部分opencv官方代码，编译后可直接用于标定mindvision的相机。

### 环境与依赖
>ubuntu 18.04

>opencv 4.1

>[mindvision 相机的SDK2.0版本](http://www.mindvision.com.cn/uploadfiles/SDK/linuxSDK_V2.1.0.17.tar.gz)

### 编译
>mkdir&&cd build

>cmake ..

### 程序执行
使用前要知道标定板的规格以及每个格子的实际长度，具体要输入的参数可以用下面这句话查询(直接运行即可)
> ./calibrate

下面这句代码为示例，意味着标定板尺寸为11*12（长宽各减一格），边长为10cm
>./calibrate --w=10 --h=11 --s=10
