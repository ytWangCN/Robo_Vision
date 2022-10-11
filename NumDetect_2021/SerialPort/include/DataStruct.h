#ifndef __DATASTRUCT_
#define __DATASTRUCT_

#include <bits/types.h>
#include <memory>
#include <opencv2/opencv.hpp>

/**
 * @brief 单云台协议结构体
 */
#pragma pack(1)
struct DataStruct
{
    //帧头 -- 决定策略
    uint8_t Flag;

    //射击模式
    uint8_t shoot_mode = 0;
    //射击频率
    uint8_t shoot_rate = 0;

    //云台角度
    float yaw;
    float pitch;
    
    uint8_t End = 0;
};
#pragma pack()

#endif // !__DATASTRUCT_
