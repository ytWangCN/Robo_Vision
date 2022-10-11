#include "RMVideoCapture.h"
#include <opencv2/opencv.hpp>
#include "Logging.h"

using namespace cv;
using namespace std;

bool RMVideoCapture::set(int propId, double value)
{
    switch (propId)
    {
    case CAP_PROP_EXPOSURE:
        exposure_time = value;
        CameraSetAeState(hCamera, false);              //关闭自动曝光
        CameraSetExposureTime(hCamera, exposure_time); //曝光时间设置
        return true;
    case CAP_PROP_GAIN:
        gain = value;
        CameraSetAnalogGain(hCamera, gain);
        return true;
    case CAP_PROP_AUTO_WB:
        CameraSetWbMode(hCamera, false);
        CameraSetPresetClrTemp(hCamera, 3);
        return true;
    default:
        DEBUG_ERROR_(__FILE__ << ", line" << __LINE__ << ": 尝试设置未定义变量。");
        return false;
    }
}

double RMVideoCapture::get(int propId) const
{
    switch (propId)
    {
    case CAP_PROP_EXPOSURE:
        return exposure_time;
    case CAP_PROP_GAIN:
        return gain;
    default:
        DEBUG_ERROR_(__FILE__ << ", line" << __LINE__ << ": 尝试获取未定义变量。");
        return 0;
    }
}

bool RMVideoCapture::open()
{
    iCameraCounts = 1;
    iStatus = -1;
    iplImage = nullptr;
    CameraSdkInit(1);

    //枚举设备，并建立设备列表
    iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
    DEBUG_INFO_("相机枚举状态：" << CameraGetErrorString(iStatus));

    DEBUG_INFO_("相机数量：" << iCameraCounts);
    //没有连接设备
    if (iCameraCounts == 0)
    {
        ERROR_("找不到相机。");
        return false;
    }

    //相机初始化。初始化成功后，才能调用任何其他相机相关的操作接口
    iStatus = CameraInit(&tCameraEnumList, -1, -1, &hCamera);

    //初始化失败
    DEBUG_INFO_("相机初始化状态：" << CameraGetErrorString(iStatus));
    if (iStatus != CAMERA_STATUS_SUCCESS)
    {
        ERROR_("相机初始化失败。");
        return false;
    }

    //获得相机的特性描述结构体。该结构体中包含了相机可设置的各种参数的范围信息。决定了相关函数的参数
    CameraGetCapability(hCamera, &tCapability);
    if (!g_pRgbBuffer)
        g_pRgbBuffer = new BYTE[tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3];
    CameraPlay(hCamera);

    if (tCapability.sIspCapacity.bMonoSensor)
    {
        channel = 1;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
    }
    else
    {
        channel = 3;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
    }

    /*让SDK进入工作模式，开始接收来自相机发送的图像
    数据。如果当前相机是触发模式，则需要接收到
    触发帧以后才会更新图像。    */

    return true;
}

bool RMVideoCapture::retrieve(OutputArray image, int flag)
{
    CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);
    // if (iplImage)
    // {
    //     cvReleaseImageHeader(&iplImage);
    // }
    // iplImage = cvCreateImageHeader(cvSize(sFrameInfo.iWidth, sFrameInfo.iHeight), IPL_DEPTH_8U, channel);
    // cvSetData(iplImage, g_pRgbBuffer, sFrameInfo.iWidth * channel); //此处只是设置指针，无图像块数据拷贝，不需担心转换效率
    // image.assign(cv::cvarrToMat(iplImage));                         //这里只是进行指针转换，将IplImage转换成Mat类型
    // CameraReleaseImageBuffer(hCamera, pbyBuffer);
    if (g_pRgbBuffer)
    {
        image.assign(cv::Mat(
                cvSize(sFrameInfo.iWidth,sFrameInfo.iHeight), 
                sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
                g_pRgbBuffer
                ));
        CameraReleaseImageBuffer(hCamera, pbyBuffer);
        return true;
    }
    else
    {
        image.assign(cv::Mat());
        return false;
    }
};

bool RMVideoCapture::reconnect()
{
    release();
    sleep(1);
    open();
    set(CAP_PROP_EXPOSURE, exposure_time);
    set(CAP_PROP_GAIN, gain);
    set(CAP_PROP_AUTO_WB, 0);
    std::cout << "reconnect" << std::endl;
    return true;
};