// Demo for the C++ TensorRT traffic-light detector.
//
// Usage:
//   tl_demo --source video.mp4 --model best.engine [--conf 0.25] [--imgsz 1280]
//   tl_demo --source 0 --model best.engine --no-show --save out.mp4

#include "trt_pipeline.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

// BGR colors keyed to class_id.
const cv::Scalar kColors[] = {
    {0, 0, 255},    {0, 255, 255}, {0, 255, 0},   {0, 0, 180},
    {0, 180, 0},    {128, 0, 255}, {128, 255, 0},
};

void drawDetections(cv::Mat& frame, const std::vector<tl::Detection>& dets, float fps) {
    for (const auto& d : dets) {
        cv::Scalar color =
            (d.class_id >= 0 && d.class_id < static_cast<int>(sizeof(kColors) / sizeof(kColors[0])))
                ? kColors[d.class_id]
                : cv::Scalar{255, 255, 255};
        cv::Point p1(static_cast<int>(d.x1), static_cast<int>(d.y1));
        cv::Point p2(static_cast<int>(d.x2), static_cast<int>(d.y2));
        cv::rectangle(frame, p1, p2, color, 2);

        std::ostringstream label;
        label.precision(2);
        label << d.class_name() << " " << std::fixed << d.confidence;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, {p1.x, p1.y - ts.height - 4}, {p1.x + ts.width, p1.y}, color, cv::FILLED);
        cv::putText(frame, label.str(), {p1.x, p1.y - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    {255, 255, 255}, 1);
    }
    if (fps > 0.f) {
        std::ostringstream s;
        s.precision(1);
        s << "FPS: " << std::fixed << fps;
        cv::putText(frame, s.str(), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);
    }
}

struct Args {
    std::string source;
    std::string model;
    float conf = 0.25f;
    int imgsz = 1280;
    bool show = true;
    std::string save;
};

void printUsage() {
    std::cerr << "Usage: tl_demo --source <video|camera_idx> --model <path.engine> "
                 "[--conf 0.25] [--imgsz 1280] [--no-show] [--save out.mp4]\n";
}

float parseFloat(const char* s, const char* flag) {
    try {
        size_t pos = 0;
        float v = std::stof(s, &pos);
        if (pos != std::strlen(s)) throw std::invalid_argument("trailing chars");
        return v;
    } catch (const std::exception& e) {
        std::cerr << flag << ": cannot parse '" << s << "' as float (" << e.what() << ")\n";
        std::exit(2);
    }
}

int parseInt(const char* s, const char* flag) {
    try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != std::strlen(s)) throw std::invalid_argument("trailing chars");
        return v;
    } catch (const std::exception& e) {
        std::cerr << flag << ": cannot parse '" << s << "' as int (" << e.what() << ")\n";
        std::exit(2);
    }
}

bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--source")      a.source = next("--source");
        else if (k == "--model")  a.model  = next("--model");
        else if (k == "--conf")   a.conf   = parseFloat(next("--conf"), "--conf");
        else if (k == "--imgsz")  a.imgsz  = parseInt(next("--imgsz"), "--imgsz");
        else if (k == "--save")   a.save   = next("--save");
        else if (k == "--no-show") a.show  = false;
        else if (k == "-h" || k == "--help") { printUsage(); std::exit(0); }
        else { std::cerr << "unknown arg: " << k << "\n"; return false; }
    }
    return !a.source.empty() && !a.model.empty();
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage();
        return 2;
    }

    cv::VideoCapture cap;
    try {
        int cam_idx = std::stoi(args.source);
        cap.open(cam_idx);
    } catch (const std::exception&) {
        cap.open(args.source);
    }
    if (!cap.isOpened()) {
        std::cerr << "Cannot open source: " << args.source << "\n";
        return 1;
    }

    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double input_fps = cap.get(cv::CAP_PROP_FPS);
    if (input_fps <= 0) input_fps = 30.0;

    cv::VideoWriter writer;
    if (!args.save.empty()) {
        writer.open(args.save, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), input_fps, {w, h});
        if (!writer.isOpened()) std::cerr << "warning: failed to open writer for " << args.save << "\n";
    }

    std::unique_ptr<tl::TRTDetector> detector;
    try {
        detector.reset(new tl::TRTDetector(args.model, args.conf, args.imgsz));
    } catch (const std::exception& e) {
        std::cerr << "failed to init detector: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Input: " << args.source << " (" << w << "x" << h << " @ "
              << input_fps << " fps)\n"
              << "Model: conf=" << args.conf << " imgsz=" << detector->imgsz() << "\n"
              << "Press 'q' to quit\n";

    cv::Mat frame;
    int frame_count = 0;
    double total_s = 0.0;
    float fps = 0.f;

    while (cap.read(frame)) {
        auto t0 = std::chrono::steady_clock::now();
        auto dets = detector->detect(frame);
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        total_s += dt;
        ++frame_count;
        fps = static_cast<float>(frame_count / total_s);

        if (frame_count % 30 == 0) {
            std::cout << "Frame " << frame_count << ": " << dets.size()
                      << " detections, " << (dt * 1000.0) << " ms ("
                      << fps << " FPS avg)\n";
        }

        if (args.show || writer.isOpened()) {
            cv::Mat vis = frame.clone();
            drawDetections(vis, dets, fps);
            if (writer.isOpened()) writer.write(vis);
            if (args.show) {
                cv::imshow("Traffic Light Detection", vis);
                int key = cv::waitKey(1) & 0xFF;
                if (key == 'q') break;
            }
        }
    }

    if (frame_count > 0) {
        double avg_ms = total_s / frame_count * 1000.0;
        std::cout << "\nSummary: " << frame_count << " frames, avg " << avg_ms
                  << " ms/frame (" << fps << " FPS)\n";
    }
    return 0;
}
