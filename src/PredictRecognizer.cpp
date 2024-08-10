#ifndef PREDICTRECOGNIZE_H
#define PREDICTRECOGNIZE_H

#include "PredictBase.h"

class PredictRecognize : public PredictBase {
public:
    PredictRecognize(const std::string& model_path, const std::string& device) 
        : PredictBase(model_path, device) {}

    std::unique_ptr<Ort::Session>& GetSessionModel() override {
        if (!session_model) {
            session_model = std::make_unique<Ort::Session>(env, onnx_model.c_str(), session_options);
            std::cout << "Model loaded successfully on " << device << std::endl;
        }
        return session_model;
    }

    

private:
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session_model;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;


};

#endif // PREDICTRECOGNIZE_H
