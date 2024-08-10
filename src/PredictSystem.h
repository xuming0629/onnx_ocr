/*
* @FileName      : PredictSystem.h
* @Time          : 2024-08-03 10:00:00
* @Author        : XuMing
* @Email         : 920972751@qq.com
* @description   : PredictSystem text regions in images using ONNX Runtime.
*/


#ifndef PREDICTSYSTEM_H
#define PREDICTSYSTEM_H

#include <string>
#include <opencv2/opencv.hpp>

class PredictSystem {
public:
    PredictSystem();
    ~PredictSystem();

    void LoadModel(const std::string& det_model, const std::string& rec_model, const std::string& rec_char_dict, const std::string& cls_model);

    std::string ocr(cv::Mat img);

private:
    class Impl; // 前向声明内部实现类
    std::unique_ptr<Impl> pImpl; // 使用 Pimpl 方式隐藏实现
};

#endif // PREDICTSYSTEM_H
