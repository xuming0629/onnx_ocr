/*
* @FileName      : PredictRecognizer.h
* @Time          : 2024-08-03 10:00:00
* @Author        : XuMing
* @Email         : 920972751@qq.com
* @description   : Recognizer text regions in images using ONNX Runtime.
*/

#ifndef PREDICTRECOGNIZER_H
#define PREDICTRECOGNIZER_H

#include "PredictBase.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Class for text recognition using ONNX Runtime.
 * 
 * This class handles the loading of the recognition model, preprocessing of input images,
 * running inference, and postprocessing to extract recognized text.
 */
class PredictRecognizer : public PredictBase {
public:
    /**
     * @brief Construct a new PredictRecognizer object.
     * 
     * Initializes the model, loads the character dictionary, and sets up the session.
     * 
     * @param model_path Path to the ONNX model file.
     * @param device Target device for model inference ("cpu", "cuda", "auto").
     * @param rec_char_dic Path to the character dictionary file.
     */
    PredictRecognizer(const std::string& model_path, const std::string& device, const std::string& rec_char_dic);

    /**
     * @brief Get the ONNX session model, loading it if necessary.
     * 
     * @return std::unique_ptr<Ort::Session>& Reference to the unique pointer of the session model.
     */
    std::unique_ptr<Ort::Session>& GetSessionModel() override;

    /**
     * @brief Load the character dictionary for text recognition.
     * 
     * @param filename Path to the character dictionary file.
     */
    void LoadAlphabet(const std::string& filename);

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
     * @brief Run the prediction on the input image to recognize text.
     * 
     * @param image Input image for text recognition.
     * @return std::string Recognized text.
     */
    std::string Predict(const cv::Mat& image);

    /**
     * @brief Postprocess the output of the model to extract recognized text.
     * 
     * @param outputs The raw outputs from the ONNX model.
     * @return std::string Recognized text.
     */
    std::string Postprocess(std::vector<Ort::Value>& outputs);

private:
    int width_;  ///< Width of the input image for the model.
    int height_;  ///< Height of the input image for the model.
    std::vector<float> input_image_;  ///< Preprocessed and normalized image data.
    std::vector<std::string> alphabet_;  ///< Loaded alphabet from the character dictionary.
    std::vector<int> preb_label_;  ///< Predicted labels after inference.
    int names_len;  ///< Length of the alphabet.

    Ort::AllocatorWithDefaultOptions allocator_;  ///< ONNX Runtime memory allocator.
    std::unique_ptr<Ort::Session> session_model_;  ///< ONNX Runtime session model.
};

#endif // PREDICTRECOGNIZER_H
