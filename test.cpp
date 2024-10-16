#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // Load class names
    std::vector<std::string> classNames;
    std::ifstream classFile("coco.names");
    std::string line;
    while (std::getline(classFile, line)) {
        classNames.push_back(line);
    }

    // Load YOLO model
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("yolov3.cfg", "yolov3.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);  // You can change to DNN_TARGET_OPENCL if using GPU

    // Open video capture
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera" << std::endl;
        return -1;
    }

    // Define parameters for NMS
    float confidenceThreshold = 0.001;
    float nmsThreshold = 0.01;  // NMS threshold for filtering

    while (true) {
        cv::Mat frame;
        cap >> frame;  // Capture frame
        if (frame.empty()) {
            break;
        }

        // Create a 4D blob from the frame
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // Run forward pass to get output from YOLO
        std::vector<cv::Mat> outs;
        std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
        net.forward(outs, outNames);

        // Vectors to store the results
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // Process the detections
        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                float confidence = data[4];
                if (confidence > confidenceThreshold) {  // Confidence threshold
                    int classId = -1;
                    float maxClassScore = -1;
                    for (int k = 5; k < outs[i].cols; ++k) {
                        if (data[k] > maxClassScore) {
                            maxClassScore = data[k];
                            classId = k - 5;
                        }
                    }

                    // Only process if the detected object is a stop sign (class ID 11 in COCO dataset)
                    if (classId == 11 && maxClassScore > confidenceThreshold) {
                        int centerX = (int)(data[0] * frame.cols);
                        int centerY = (int)(data[1] * frame.rows);
                        int width = (int)(data[2] * frame.cols);
                        int height = (int)(data[3] * frame.rows);
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

        // Show frame with detected stop signs
        cv::imshow("YOLOv3 Stop Sign Detection", frame);

        if (cv::waitKey(1) == 27) {  // Exit on 'ESC' key press
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}