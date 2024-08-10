#include "PredictDetector.h"

/**
 * @brief Construct a new PredictDetector object.
 * 
 * Initializes the model with specified parameters and prepares the ONNX session.
 * 
 * @param model_path Path to the ONNX model file.
 * @param device Target device for model inference ("cpu", "cuda", "auto").
 * @param db_thresh Threshold for binary segmentation.
 * @param box_thresh Threshold for text box filtering.
 * @param max_candidates Maximum number of candidates to consider.
 * @param unclip_ratio Ratio for unclipping the detected text box.
 */
PredictDetector::PredictDetector(const std::string& model_path, 
                                 const std::string& device,
                                 float db_thresh,
                                 float box_thresh,
                                 int max_candidates,
                                 float unclip_ratio)
    : PredictBase(model_path, device),  // 调用基类构造函数
      db_thresh_(db_thresh),
      box_thresh_(box_thresh),
      max_candidates_(max_candidates),
      unclip_ratio_(unclip_ratio),
      src_height_(0),
      src_width_(0),
      short_size_(640)  // 默认短边大小
{
    // 在构造函数主体中初始化 session_model_
    session_model_ = std::move(GetSessionModel());
}

/**
 * @brief Get the ONNX session model, loading it if necessary.
 * 
 * @return std::unique_ptr<Ort::Session>& Reference to the unique pointer of the session model.
 */
std::unique_ptr<Ort::Session>& PredictDetector::GetSessionModel() {
    if (!session_model_) {
        session_model_ = std::make_unique<Ort::Session>(env, onnx_model.c_str(), session_options);
        std::cout << "Model loaded detector model successfully on " << device << std::endl;
    }
    return session_model_;
}

/**
 * @brief Get the input node names of the ONNX model.
 * 
 * @return std::vector<std::string> List of input node names.
 */
std::vector<std::string> PredictDetector::GetInputNames() {
    if (!session_model_) {
        throw std::runtime_error("Session model is not initialized.");
    }

    std::vector<std::string> input_names;
    size_t num_input_nodes = session_model_->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto input_name = session_model_->GetInputNameAllocated(i, allocator_);
        input_names.push_back(input_name.get());
    }
    return input_names;
}

/**
 * @brief Get the output node names of the ONNX model.
 * 
 * @return std::vector<std::string> List of output node names.
 */
std::vector<std::string> PredictDetector::GetOutputNames() {
    if (!session_model_) {
        throw std::runtime_error("Session model is not initialized.");
    }

    std::vector<std::string> output_names;
    size_t num_output_nodes = session_model_->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto output_name = session_model_->GetOutputNameAllocated(i, allocator_);
        output_names.push_back(output_name.get());
    }
    return output_names;
}

/**
 * @brief Run the prediction on the input image and detect text regions.
 * 
 * @param src_img Input image for text detection.
 * @return std::vector<std::vector<cv::Point2f>> Detected text regions as vectors of points.
 */
std::vector<std::vector<cv::Point2f>> PredictDetector::Predict(cv::Mat& src_img) {
    if (!session_model_) {
        throw std::runtime_error("Session model is not initialized.");
    }

    src_height_ = src_img.rows;
    src_width_ = src_img.cols;
    dstimg_ = Preprocess(src_img);

    // Check if preprocessing was successful
    if (dstimg_.empty()) {
        throw std::runtime_error("Preprocessed image is empty.");
    }

    Normalize(dstimg_);

    // Assuming `input_image_` is populated within Preprocess or Normalize functions
    // Make sure `input_image_` is the correct size (1x3xHxW)
    std::array<int64_t, 4> input_shape{ 1, 3, dstimg_.rows, dstimg_.cols };
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape.data(), input_shape.size());

    // Retrieve input and output node names
    auto input_names = GetInputNames();
    auto output_names = GetOutputNames();

    std::vector<const char*> input_names_cstr;
    for (const auto& name : input_names) {
        input_names_cstr.push_back(name.c_str());
    }

    std::vector<const char*> output_names_cstr;
    for (const auto& name : output_names) {
        output_names_cstr.push_back(name.c_str());
    }

    // Run the model with the input tensor
    std::vector<Ort::Value> ort_outputs = session_model_->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), &input_tensor, 1, output_names_cstr.data(), output_names_cstr.size());

    // Post-process the output to extract results
    std::vector<std::vector<cv::Point2f>> results = Postprocess(ort_outputs);
    return results;
}

/**
 * @brief Postprocess the output of the model to extract text regions.
 * 
 * @param outputs The raw outputs from the ONNX model.
 * @return std::vector<std::vector<cv::Point2f>> The postprocessed text regions.
 */
std::vector<std::vector<cv::Point2f>> PredictDetector::Postprocess(std::vector<Ort::Value>& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error("Output from the model is empty.");
    }

    const float* floatArray = outputs[0].GetTensorMutableData<float>();
    int outputCount = 1;
    const auto& output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    for (int dim : output_shape) {
        outputCount *= dim;
    }

    cv::Mat binary(dstimg_.rows, dstimg_.cols, CV_32FC1);
    std::memcpy(binary.data, floatArray, outputCount * sizeof(float));

    cv::Mat bitmap;
    cv::threshold(binary, bitmap, db_thresh_, 255, cv::THRESH_BINARY);

    float scale_height = static_cast<float>(src_height_) / binary.rows;
    float scale_width = static_cast<float>(src_width_) / binary.cols;

    bitmap.convertTo(bitmap, CV_8UC1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    size_t num_candidate = std::min(contours.size(), static_cast<size_t>(max_candidates_ > 0 ? max_candidates_ : INT_MAX));

    std::vector<std::vector<cv::Point2f>> results;
    for (size_t i = 0; i < num_candidate; i++) {
        std::vector<cv::Point>& contour = contours[i];
        // Calculate text contour score
        if (contourScore(binary, contour) < box_thresh_) {
            continue;
        }

        // Rescale the contour
        std::vector<cv::Point> contour_scaled;
        contour_scaled.reserve(contour.size());
        for (const auto& pt : contour) {
            contour_scaled.emplace_back(static_cast<int>(pt.x * scale_width), static_cast<int>(pt.y * scale_height));
        }

        // Unclip and refine the bounding box
        cv::RotatedRect box = cv::minAreaRect(contour_scaled);
        float longSide = std::max(box.size.width, box.size.height);
        if (longSide < longSideThresh) {
            continue;
        }

        const float angle_threshold = 60.0f;
        if (box.size.width < box.size.height || std::fabs(box.angle) >= angle_threshold) {
            std::swap(box.size.width, box.size.height);
            box.angle = box.angle < 0 ? box.angle + 90 : box.angle - 90;
        }

        // Get the rectangle points
        cv::Point2f vertex[4];
        box.points(vertex);

        std::vector<cv::Point2f> approx(vertex, vertex + 4);
        std::vector<cv::Point2f> polygon;
        unclip(approx, polygon);

        // Recheck the polygon size after unclipping
        box = cv::minAreaRect(polygon);
        longSide = std::max(box.size.width, box.size.height);
        if (longSide < longSideThresh + 2) {
            continue;
        }

        results.push_back(polygon);
    }

    return results;
}

/**
 * @brief Unclip the detected text region to expand its bounding box.
 * 
 * @param inPoly The original detected polygon.
 * @param outPoly The unclipped polygon (output).
 */
void PredictDetector::unclip(const std::vector<cv::Point2f>& inPoly, std::vector<cv::Point2f>& outPoly) {
    // Calculate the area and perimeter of the polygon
    float area = cv::contourArea(inPoly);
    float perimeter = cv::arcLength(inPoly, true);
    float distance = area * unclip_ratio_ / perimeter;

    size_t numPoints = inPoly.size();
    std::vector<std::vector<cv::Point2f>> newLines;

    for (size_t i = 0; i < numPoints; i++) {
        std::vector<cv::Point2f> newLine;

        // Handle index wrap-around
        cv::Point2f pt1 = inPoly[i];
        cv::Point2f pt2 = inPoly[(i + numPoints - 1) % numPoints]; // Corrected index calculation

        // Vector from pt2 to pt1
        cv::Point2f vec = pt1 - pt2;

        // Normalize the vector and scale by the unclipping distance
        float normVec = cv::norm(vec);
        cv::Point2f offsetVec = cv::Point2f(vec.y * distance / normVec, -vec.x * distance / normVec);

        // Add new points offset by the unclipping distance
        newLine.push_back(pt1 + offsetVec);
        newLine.push_back(pt2 + offsetVec);
        newLines.push_back(newLine);
    }

    size_t numLines = newLines.size();
    for (size_t i = 0; i < numLines; i++) {
        cv::Point2f a = newLines[i][0];
        cv::Point2f b = newLines[i][1];
        cv::Point2f c = newLines[(i + 1) % numLines][0];
        cv::Point2f d = newLines[(i + 1) % numLines][1];
        cv::Point2f intersectionPoint;

        // Calculate vectors for angle between lines
        cv::Point2f vec1 = b - a;
        cv::Point2f vec2 = d - c;

        // Calculate the cosine of the angle between vec1 and vec2
        float cosAngle = (vec1.x * vec2.x + vec1.y * vec2.y) / (cv::norm(vec1) * cv::norm(vec2));

        // If the lines are almost parallel
        if (fabs(cosAngle) > 0.7) {
            intersectionPoint.x = (b.x + c.x) * 0.5f;
            intersectionPoint.y = (b.y + c.y) * 0.5f;
        } else {
            // Otherwise, calculate intersection of the lines
            float denominator = a.x * (d.y - c.y) + b.x * (c.y - d.y) +
                                d.x * (b.y - a.y) + c.x * (a.y - b.y);
            float numerator = a.x * (d.y - c.y) + c.x * (a.y - d.y) + d.x * (c.y - a.y);
            float s = numerator / denominator;

            intersectionPoint.x = a.x + s * (b.x - a.x);
            intersectionPoint.y = a.y + s * (b.y - a.y);
        }
        
        // Store the unclipped point
        outPoly.push_back(intersectionPoint);
    }
}

/**
 * @brief Crop and rotate the detected text region from the image.
 * 
 * @param frame The original image from which to crop.
 * @param vertices The vertices of the text region to crop.
 * @return cv::Mat The cropped and rotated text region.
 */
cv::Mat PredictDetector::get_rotate_crop_image(const cv::Mat& frame, const std::vector<cv::Point2f>& vertices) {
    // Calculate the bounding rectangle for the provided vertices
    cv::Rect rect = cv::boundingRect(vertices);
    
    // Crop the region of interest from the frame
    cv::Mat crop_img = frame(rect);

    // Define the size of the output image
    const cv::Size output_size(rect.width, rect.height);

    // Define the target vertices in the cropped image
    std::vector<cv::Point2f> targetVertices{
        cv::Point2f(0, output_size.height),     // Bottom-left
        cv::Point2f(0, 0),                      // Top-left
        cv::Point2f(output_size.width, 0),      // Top-right
        cv::Point2f(output_size.width, output_size.height) // Bottom-right
    };

    // Adjust the original vertices to be relative to the top-left corner of the bounding rectangle
    std::vector<cv::Point2f> adjustedVertices(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        adjustedVertices[i] = cv::Point2f(
            vertices[i].x - rect.x,  // Adjust x-coordinate relative to the bounding rectangle
            vertices[i].y - rect.y   // Adjust y-coordinate relative to the bounding rectangle
        );
    }

    // Get the perspective transformation matrix
    cv::Mat rotationMatrix = cv::getPerspectiveTransform(adjustedVertices, targetVertices);

    // Apply the perspective transformation
    cv::Mat result;
    cv::warpPerspective(crop_img, result, rotationMatrix, output_size, cv::BORDER_REPLICATE);

    return result;
}

/**
 * @brief Calculate the contour score of a detected region.
 * 
 * @param binary The binary image output from the model.
 * @param contour The contour of the detected region.
 * @return float The score of the contour.
 */
float PredictDetector::contourScore(const cv::Mat& binary, const std::vector<cv::Point>& contour) {
    cv::Rect rect = cv::boundingRect(contour);
    int xmin = std::max(rect.x, 0);
    int xmax = std::min(rect.x + rect.width, binary.cols - 1);
    int ymin = std::max(rect.y, 0);
    int ymax = std::min(rect.y + rect.height, binary.rows - 1);

    cv::Mat bin_roi = binary(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    std::vector<cv::Point> roi_contour;
    for (const auto& pt : contour) {
        roi_contour.emplace_back(pt.x - xmin, pt.y - ymin);
    }
    std::vector<std::vector<cv::Point>> roi_contours = { roi_contour };
    cv::fillPoly(mask, roi_contours, cv::Scalar(1));
    float score = cv::mean(bin_roi, mask).val[0];
    return score;
}

/**
 * @brief Preprocess the input image (resize, normalization, etc.).
 * 
 * @param input_image The input image to preprocess.
 * @return cv::Mat The preprocessed image.
 */
cv::Mat PredictDetector::Preprocess(const cv::Mat& input_image) {
    if (input_image.empty()) {
        throw std::runtime_error("Input image is empty.");
    }

    // Convert BGR to RGB
    cv::Mat rgb_image;
    cv::cvtColor(input_image, rgb_image, cv::COLOR_BGR2RGB);

    int original_height = input_image.rows;
    int original_width = input_image.cols;

    float scale_height = 1.0f;
    float scale_width = 1.0f;

    if (original_height < original_width) {
        scale_height = static_cast<float>(short_size_) / static_cast<float>(original_height);
        float target_width = static_cast<float>(original_width) * scale_height;
        target_width = target_width - static_cast<int>(target_width) % 32;
        target_width = std::max(32.0f, target_width);
        scale_width = target_width / static_cast<float>(original_width);
    } else {
        scale_width = static_cast<float>(short_size_) / static_cast<float>(original_width);
        float target_height = static_cast<float>(original_height) * scale_width;
        target_height = target_height - static_cast<int>(target_height) % 32;
        target_height = std::max(32.0f, target_height);
        scale_height = target_height / static_cast<float>(original_height);
    }

    cv::Mat resized_image;
    cv::resize(rgb_image, resized_image, cv::Size(static_cast<int>(scale_width * original_width), static_cast<int>(scale_height * original_height)), 0, 0, cv::INTER_LINEAR);

    return resized_image;
}

/**
 * @brief Normalize the preprocessed image for inference.
 * 
 * @param img The preprocessed image to normalize.
 */
void PredictDetector::Normalize(cv::Mat& img) {
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();

    // Ensure that input_image_ is the correct size
    input_image_.resize(rows * cols * channels);

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float pixel = img.ptr<uchar>(i)[j * channels + c];
                input_image_[c * rows * cols + i * cols + j] = (pixel / 255.0f - mean_values_[c]) / norm_values_[c];
            }
        }
    }
}
