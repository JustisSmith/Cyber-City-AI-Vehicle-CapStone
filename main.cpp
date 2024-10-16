#include "yolov5.hpp"
#include <opencv2/opencv.hpp>
#include <csignal>

static volatile bool keep_running = true;

void keyboard_handler(int sig) {
    // Handle keyboard interrupt
    if (sig == SIGINT)
        keep_running = false;
}

int main(int argc, char** argv) {
    signal(SIGINT, keyboard_handler);

    // Model initialization
    std::string wts_name = "yolop.wts";  // Path to the weights file
    std::string engine_name = "yolop.engine";  // Path to the engine file

    // Deserialize the engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cout << "Building engine..." << std::endl;
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream);  // Build the model
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        std::cout << "Engine has been built and saved to file." << std::endl;
    }

    // Load the model
    char* trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // Prepare data for CPU inference
    static float det_out[BATCH_SIZE * OUTPUT_SIZE];
    static int seg_out[BATCH_SIZE * IMG_H * IMG_W];
    static int lane_out[BATCH_SIZE * IMG_H * IMG_W];

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Prepare buffers (CPU)
    void* buffers[4];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int output_det_index = engine->getBindingIndex(OUTPUT_DET_NAME);
    const int output_seg_index = engine->getBindingIndex(OUTPUT_SEG_NAME);
    const int output_lane_index = engine->getBindingIndex(OUTPUT_LANE_NAME);

    // Allocate CPU buffers
    float* inputBuffer = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    buffers[output_det_index] = det_out;
    buffers[output_seg_index] = seg_out;
    buffers[output_lane_index] = lane_out;

    // Open webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera" << std::endl;
        return -1;
    }

    char key = ' ';
    while (keep_running && key != 'q') {
        cv::Mat frame;
        cap >> frame;  // Capture frame from webcam
        if (frame.empty()) continue;  // Skip empty frames

        // Preprocess the frame (resize and normalize)
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(INPUT_W, INPUT_H));  // Resize to model input size
        resized_frame.convertTo(resized_frame, CV_32F, 1.0 / 255.0);  // Normalize to [0, 1]
        std::memcpy(inputBuffer, resized_frame.data, sizeof(float) * INPUT_H * INPUT_W * 3);  // Copy to input buffer

        // Run inference
        auto start = std::chrono::system_clock::now();
        context->execute(BATCH_SIZE, buffers);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // Postprocess the results
        std::vector<Yolo::Detection> batch_res;
        nms(batch_res, det_out, CONF_THRESH, NMS_THRESH);

        // Show results (implement visualization function to draw detections)
        visualization(frame, batch_res, key);
    }

    // Clean up
    cap.release();
    delete[] inputBuffer;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}