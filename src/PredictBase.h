#ifndef PREDICTBASE_H
#define PREDICTBASE_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

/**
 * @brief Base class for ONNX model prediction.
 * 
 * This class provides the base functionality for loading and managing an ONNX model 
 * and offers an interface for derived classes to implement specific prediction logic.
 */
class PredictBase {
public:
    /**
     * @brief Construct a new PredictBase object.
     * 
     * @param model_path Path to the ONNX model file.
     * @param device Target device for model inference ("cpu", "cuda", "auto").
     */
    PredictBase(const std::string& model_path, const std::string& device);

    /**
     * @brief Destroy the PredictBase object and release resources.
     */
    virtual ~PredictBase();

    /**
     * @brief Get the ONNX session model.
     * 
     * This function should be implemented by derived classes to return the model session.
     * 
     * @return std::unique_ptr<Ort::Session>& Reference to the unique pointer of the session model.
     */
    virtual std::unique_ptr<Ort::Session>& GetSessionModel() = 0;

    /**
     * @brief Get the input node names of the ONNX model.
     * 
     * This function should be implemented by derived classes to return the names of input nodes.
     * 
     * @return std::vector<std::string> List of input node names.
     */
    virtual std::vector<std::string> GetInputNames() = 0;

    /**
     * @brief Get the output node names of the ONNX model.
     * 
     * This function should be implemented by derived classes to return the names of output nodes.
     * 
     * @return std::vector<std::string> List of output node names.
     */
    virtual std::vector<std::string> GetOutputNames() = 0;

protected:
    std::unique_ptr<Ort::Session> session_model;  ///< Unique pointer to the ONNX runtime session model.
    std::string onnx_model;  ///< Path to the ONNX model file.
    std::string device;  ///< Target device for model inference.
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"};  ///< ONNX runtime environment.
    Ort::SessionOptions session_options;  ///< Session options for the ONNX runtime.
    std::vector<const char*> providers;  ///< Execution providers selected for inference.

    /**
     * @brief Select the appropriate execution providers based on the target device.
     * 
     * @param device The target device ("cpu", "cuda", "auto").
     * @return std::vector<const char*> List of selected execution providers.
     */
    std::vector<const char*> SelectDevice(const std::string& device);

    /**
     * @brief Get the session options for the ONNX runtime.
     * 
     * @return Ort::SessionOptions Configured session options.
     */
    Ort::SessionOptions GetSessionOptions();

    /**
     * @brief Check if a CUDA execution provider is available.
     * 
     * @return true If CUDA is available.
     * @return false If CUDA is not available.
     */
    bool IsCudaAvailable();
};

#endif // PREDICTBASE_H
