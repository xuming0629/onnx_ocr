#ifndef PREDICTCLASSIFY_H
#define PREDICTCLASSIFY_H

#include "PredictBase.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <array>
#include <vector>

/**
 * @brief Class for image classification using ONNX Runtime.
 * 
 * This class handles the loading of the classification model, preprocessing of input images,
 * running inference, and postprocessing to obtain classification results.
 */
class PredictClassifier : public PredictBase {
public:
    /**
     * @brief Construct a new PredictClassifier object.
     * 
     * Initializes the model and prepares the ONNX session.
     * 
     * @param model_path Path to the ONNX model file.
     * @param device Target device for model inference ("cpu", "cuda", "auto").
     */
    PredictClassifier(const std::string& model_path, const std::string& device);

    /**
     * @brief Get the ONNX session model, loading it if necessary.
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
     * @brief Get the output shapes of the ONNX model.
     * 
     * @return std::vector<std::vector<int64_t>> List of output shapes.
     */
    std::vector<std::vector<int64_t>> GetOutputShapes();

    /**
     * @brief Preprocess the input image (resize, normalization, etc.).
     * 
     * @param image The input image to preprocess.
     * @return cv::Mat The preprocessed image.
     */
    cv::Mat Preprocess(const cv::Mat& image);

    /**
     * @brief Normalize the preprocessed image for inference.
     * 
     * @param img The preprocessed image to normalize.
     */
    void Normalize(cv::Mat& img);

    /**
     * @brief Run the prediction on the input image to classify it.
     * 
     * @param img Input image for classification.
     * @return int Predicted class label.
     */
    int Predict(cv::Mat& img);

    /**
     * @brief Postprocess the output of the model to determine the predicted class.
     * 
     * @param outputs The raw outputs from the ONNX model.
     * @return int Predicted class label index.
     */
    int Postprocess(std::vector<Ort::Value>& outputs);

private:
    const int label_list[2] = { 0, 180 };  ///< List of class labels (0 and 180 degrees).
    int width_ = 192;  ///< Width of the input image for the model.
    int height_ = 48;  ///< Height of the input image for the model.
    int num_out;  ///< Number of output nodes.
    std::vector<float> input_image_;  ///< Preprocessed and normalized image data.
    Ort::AllocatorWithDefaultOptions allocator_;  ///< ONNX Runtime memory allocator.
    std::unique_ptr<Ort::Session> session_model_;  ///< ONNX Runtime session model.
};

#endif // PREDICTCLASSIFY_H
