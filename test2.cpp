#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // Load class names (Adjust this based on your model's classes)
    std::vector<std::string> classNames;
    std::ifstream classFile("coco.names"); // Modify to your class names file if necessary
    std::string line;
    while (std::getline(classFile, line)) {
        classNames.push_back(line);
    }

    // Load YOLOv5 ONNX model
    cv::dnn::Net net = cv::dnn::readNetFromONNX("yolop-640-640.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);  // Change to DNN_TARGET_OPENCL for GPU

    // Open video capture
    cv::VideoCapture cap(0); // Change to video file path if needed
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera" << std::endl;
        return -1;
    }

    // Define parameters for NMS
    float confidenceThreshold = 0.25; // Adjust as needed
    float nmsThreshold = 0.45; // Adjust as needed

    while (true) {
        cv::Mat frame;
        cap >> frame;  // Capture frame
        if (frame.empty()) {
            break;
        }

        // Create a 4D blob from the frame
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // Run forward pass to get output from YOLO
        std::vector<cv::Mat> outs;
        net.forward(outs);

        // Vectors to store the results
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // Process the detections
        for (const auto& output : outs) {
            float* data = (float*)output.data;
            for (int j = 0; j < output.rows; ++j, data += output.cols) {
                float confidence = data[4];
                if (confidence > confidenceThreshold) {  // Confidence threshold
                    int classId = -1;
                    float maxClassScore = -1;
                    for (int k = 5; k < output.cols; ++k) {
                        if (data[k] > maxClassScore) {
                            maxClassScore = data[k];
                            classId = k - 5;
                        }
                    }

                    // Assuming lane class ID is 0, adjust as per your model's classes
                    if (classId == 0 && maxClassScore > confidenceThreshold) {
                        int centerX = static_cast<int>(data[0] * frame.cols);
                        int centerY = static_cast<int>(data[1] * frame.rows);
                        int width = static_cast<int>(data[2] * frame.cols);
                        int height = static_cast<int>(data[3] * frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        // Save the box, confidence, and class ID
                        boxes.push_back(cv::Rect(left, top, width, height));
                        confidences.push_back(confidence);
                        classIds.push_back(classId);
                    }
                }
            }
        }

        // Apply Non-Maximum Suppression to remove redundant boxes
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

        // Draw the final boxes after NMS
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 3);

            std::string label = classNames[classIds[idx]] + ": " + cv::format("%.2f", confidences[idx]);
            cv::putText(frame, label, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        }

        // Show frame with detected lanes
        cv::imshow("YOLOv5 Lane Detection", frame);

        if (cv::waitKey(1) == 27) {  // Exit on 'ESC' key press
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}