#include <net.h>
#include <gpu.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

int main() {
    ncnn::create_gpu_instance();
    {
        ncnn::Net upscaler;
        
        upscaler.opt.use_vulkan_compute = true;
        upscaler.opt.use_fp16_packed = true;
        upscaler.opt.use_fp16_storage = true;
        upscaler.opt.use_fp16_arithmetic = true;
        upscaler.opt.num_threads = 4;

        upscaler.set_vulkan_device(0);

        if (upscaler.load_param("lightning_v3.ncnn.param") != 0 ||
            upscaler.load_model("lightning_v3.ncnn.bin") != 0) {
            std::cerr << "Error: Model files not found." << std::endl;
            return -1;
        }

        cv::VideoCapture cap("test.mp4");
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open test.mp4" << std::endl;
            return -1;
        }

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

        cv::VideoWriter writer("testUP.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                                       fps, cv::Size(width * 2, height * 2));
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not open writer for testUP.mp4" << std::endl;
            return -1;
        }
        
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        const float denorm_vals[3] = {255.f, 255.f, 255.f};

        cv::Mat frame;
        cv::Mat out_frame(height * 2, width * 2, CV_8UC3);

        int frames_processed = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_log_time = start_time;

        std::cout << "Starting inference...\n";

        while (cap.read(frame)) {
            ncnn::Mat in = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, width, height);
            in.substract_mean_normalize(0, norm_vals);

            ncnn::Extractor ex = upscaler.create_extractor();
            ex.input("in0", in);

            ncnn::Mat out;
            ex.extract("out0", out);

            out.substract_mean_normalize(0, denorm_vals);
            out.to_pixels(out_frame.data, ncnn::Mat::PIXEL_RGB2BGR);
            
            writer.write(out_frame); 
            
            frames_processed++;
            
            // Calculate and print live FPS every 10 frames
            if (frames_processed % 10 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> interval_diff = current_time - last_log_time;
                
                double current_fps = 10.0 / interval_diff.count();
                
                std::cout << "\rFrames: " << frames_processed << "/" << total_frames 
                          << " | Live FPS: " << std::fixed << std::setprecision(2) << current_fps 
                          << "   " << std::flush; // Extra spaces clear trailing characters
                          
                last_log_time = current_time;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_diff = end_time - start_time;
        
        std::cout << "\n\nInference finished. Cleaning up..." << std::endl;
        std::cout << "Overall Avg FPS: " << frames_processed / total_diff.count() << std::endl;
    } 

    ncnn::destroy_gpu_instance();
    return 0;
}
