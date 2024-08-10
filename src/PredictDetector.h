/*
* @FileName      : PredictDetector.h
* @Time          : 2024-08-03 10:00:00
* @Author        : XuMing
* @Email         : 920972751@qq.com
* @description   : Detector text regions in images using ONNX Runtime.
*/


#ifndef PREDICTDETECTOR_H
#define PREDICTDETECTOR_H

#include "PredictBase.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <array>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

/**
 * @brief Derived class for text detection using ONNX Runtime.
 * 
 * This class provides specific functionalities for detecting text regions in images.
 */
class PredictDetector : public PredictBase {
public:
    /**
     * @brief Construct a new PredictDetector object.
     * 
     * @param model_path Path to the ONNX model file.
     * @param device Target device for model inference ("cpu", "cuda", "auto").
     * @param db_thresh Threshold for binary segmentation.
     * @param box_thresh Threshold for text box filtering.
     * @param max_candidates Maximum number of candidates to consider.
     * @param unclip_ratio Ratio for unclipping the detected text box.
     */
    PredictDetector(const std::string& model_path, 
                    const std::string& device,
                    float db_thresh = 0.3f,
                    float box_thresh = 0.6f,
                    int max_candidates = 1000,
                    float unclip_ratio = 1.5f);

    /**
     * @brief Destroy the PredictDetector object.
     */
    ~PredictDetector() override = default;

    /**
     * @brief Get the ONNX session model.
     * 
     * @return std::unique_ptr<Ort::Session>& Reference to the unique pointer of the session model.
     */
    std::unique_ptr<Ort::Session>& GetSessionModel() override;

    /**
     * @brief Get the input node names of the ONNX model.
     * 
     * @return std::vector<std::string> List of input node names.
     */
    std::vector<std::string> GetInputNames() override;

    /**
     * @brief Get the output node names of the ONNX model.
     * 
     * @return std::vector<std::string> List of output node names.
     */
    std::vector<std::string> GetOutputNames() override;

    /**
     * @brief Run the prediction on the input image and detect text regions.
     * 
     * @param src_img Input image for text detection.
     * @return std::vector<std::vector<cv::Point2f>> Detected text regions as vectors of points.
     */
    std::vector<std::vector<cv::Point2f>> Predict(cv::Mat& src_img);

    /**
     * @brief Crop and rotate the detected text region from the image.
     * 
     * @param frame The original image from which to crop.
     * @param vertices The vertices of the text region to crop.
     * @return cv::Mat The cropped and rotated text region.
     */
    cv::Mat get_rotate_crop_image(const cv::Mat& frame, const std::vector<cv::Point2f>& vertices);

private:
    /**
     * @brief Preprocess the input image (resize, normalization, etc.).
     * 
     * @param input_image The input image to preprocess.
     * @return cv::Mat The preprocessed image.
     */
    cv::Mat Preprocess(const cv::Mat& input_image);

    /**
     * @brief Normalize the preprocessed image for inference.
     * 
     * @param img The preprocessed image to normalize.
     */
    void Normalize(cv::Mat& img); 

    /**
     * @brief Postprocess the output of the model to extract text regions.
     * 
     * @param outputs The raw outputs from the ONNX model.
     * @return std::vector<std::vector<cv::Point2f>> The postprocessed text regions.
     */
    std::vector<std::vector<cv::Point2f>> Postprocess(std::vector<Ort::Value>& outputs);

    /**
     * @brief Calculate the contour score of a detected region.
     * 
     * @param binary The binary image output from the model.
     * @param contour The contour of the detected region.
     * @return float The score of the contour.
     */
    float contourScore(const cv::Mat& binary, const std::vector<cv::Point>& contour);

    /**
     * @brief Unclip the detected text region to expand its bounding box.
     * 
     * @param inPoly The original detected polygon.
     * @param outPoly The unclipped polygon (output).
     */
    void unclip(const std::vector<cv::Point2f>& inPoly, std::vector<cv::Point2f>& outPoly);

private:
    float db_thresh_;  ///< Threshold for binary segmentation.
    float box_thresh_;  ///< Threshold for text box filtering.
    int max_candidates_;  ///< Maximum number of candidates to consider.
    float unclip_ratio_;  ///< Ratio for unclipping the detected text box.

    int src_height_;  ///< Height of the source image.
    int src_width_;  ///< Width of the source image.

    Ort::AllocatorWithDefaultOptions allocator_;  ///< ONNX Runtime memory allocator.
    std::unique_ptr<Ort::Session> session_model_;  ///< ONNX Runtime session model.

    int short_size_;  ///< Short edge size for image resizing.
    std::vector<float> input_image_;  ///< Preprocessed and normalized image data.
    cv::Mat dstimg_;  ///< The preprocessed image.
    std::vector<float> mean_values_ = {0.485f, 0.456f, 0.406f};  ///< Mean values for normalization.
    std::vector<float> norm_values_ = {0.229f, 0.224f, 0.225f};  ///< Standard deviation values for normalization.
    
    const int longSideThresh = 3;  ///< Threshold for the long side of the bounding box.
};

#endif // PREDICTDETECTOR_H
