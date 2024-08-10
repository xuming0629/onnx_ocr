#ifndef PREDICTSYSTEM_H
#define PREDICTSYSTEM_H

#include <string>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {
    EXPORT void LoadModel(const char* det_model, const char* rec_model, const char* rec_char_dict, const char* cls_model);
    EXPORT const char* ocr(const char* image_path);
}

#endif // PREDICTSYSTEM_H
