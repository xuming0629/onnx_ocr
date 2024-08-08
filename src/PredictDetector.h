#ifndef PREDICTDETECTOR_H
#define PREDICTDETECTOR_H

#include "PredictBase.h"

class PredictDetector : public PredictBase {
public:
    PredictDetector(const std::string& model_path, const std::string& device) 
        : PredictBase(model_path, device) {}

    std::unique_ptr<Ort::Session>& GetSessionModel() override {
        if (!session_model) {
            session_model = std::make_unique<Ort::Session>(env, onnx_model.c_str(), session_options);
            std::cout << "Model loaded detector model successfully on " << device << std::endl;
        }
        return session_model;
    }

    std::vector<std::string> GetInputNames() {
        if (!session_model) {
            throw std::runtime_error("Session model is not initialized.");
        }

        std::vector<std::string> input_names;
        size_t numInputNodes = session_model->GetInputCount();
        for (size_t i = 0; i < numInputNodes; ++i) {
            auto inputName = session_model->GetInputNameAllocated(i, allocator).get();
            
            input_names.push_back(inputName);
            //allocator.Free(inputName);
        }
        return input_names;
    }
    private:
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session_model;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;



};

#endif // PREDICTDETECTOR_H
