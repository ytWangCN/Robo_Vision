#ifndef _RMVIDEOCAPTURE_H
#define _RMVIDEOCAPTURE_H
#include <stdio.h>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/types_c.h>
#include <unistd.h>
#include "CameraApi.h"


using namespace cv;

class RMVideoCapture : public VideoCapture
{
public:
	RMVideoCapture()
	{
		open();
	};
	virtual ~RMVideoCapture() override
	{
		release();
		if (g_pRgbBuffer)
			delete[] g_pRgbBuffer;
	};

	virtual bool set(int propId, double value) override;
	virtual double get(int propId) const override;
	virtual bool open();

	virtual bool isOpened() const override { return iStatus == CAMERA_STATUS_SUCCESS; };
	virtual void release() override { CameraUnInit(hCamera); };
	virtual bool grab() override
	{
		return CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS;
	};

	//这个函数与OpenCV源码一致，实际上并没有重写。
	virtual bool read(OutputArray image) override
	{
		if (grab())
		{
			retrieve(image);
		}
		else
		{
			reconnect();
			image.release();
		}
		return !image.empty();
	};
	virtual bool retrieve(OutputArray image, int flag = 0) override;

	//这个函数与OpenCV源码一致，实际上并没有重写。
	virtual RMVideoCapture &operator>>(cv::Mat &image) override
	{
		read(image);
		return *this;
	};

	bool reconnect();

private:
	CameraHandle hCamera;
	BYTE *g_pRgbBuffer = nullptr;
	BYTE *pbyBuffer = nullptr;
	IplImage *iplImage = nullptr;
	int iCameraCounts = 0;
	CameraSdkStatus iStatus;
	tSdkCameraDevInfo tCameraEnumList;
	tSdkCameraCapbility tCapability; //设备描述信息
	tSdkFrameHead sFrameInfo;		 //图像信息，包括了图像的大小
	int channel;
	/**曝光时间和白平衡的参数缓存*/
	double exposure_time;
	int gain;

	using VideoCapture::open;	//通过私有化明确告诉编译器不使用VideoCapture的open方法
};

#endif
