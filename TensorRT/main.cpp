
#pragma execution_character_set("utf-8")

#include "NVIDTRT.hpp"

#if defined(_WIN32)
#	include <Windows.h>
#   include <wingdi.h>
#	include <Shlwapi.h>
#	pragma comment(lib, "shlwapi.lib")
#   pragma comment(lib, "ole32.lib")
#   pragma comment(lib, "gdi32.lib")
#	undef min
#	undef max
#else
#	include <dirent.h>
#	include <sys/types.h>
#	include <sys/stat.h>
#	include <unistd.h>
#   include <stdarg.h>
#endif
using namespace std;
static const char* labels[] = { "guoba", "riqi"};
#pragma comment(lib, "./NVID.lib")


static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}



int main(){

    int device_id = 0;
    string model = "NVIDs";
    auto type = NVID::Type::Nvid;
    auto mode = NVID::Mode::FP32;
    string onnx_file = cv::format("%s.onnx", model.c_str());
    string model_file = cv::format("%s.%s.trtmodel", model.c_str(), NVID::mode_string(mode));
    NVID::set_device(device_id);

    if (!exists(model_file) && !NVID::compile(mode, type, 6, onnx_file, model_file, 1 << 30, "inference"))
    {
        printf("Compile failed\n");
        return 0;
    }

    float confidence_threshold = 0.4f;
    float nms_threshold = 0.5f;

    auto yolo = NVID::create_infer(model_file, type, device_id, confidence_threshold, nms_threshold);
    if (yolo == nullptr)
    {
        printf("Yolo is nullptr\n");
        return 0;
    }


    std::vector<cv::String> files_;
    cv::glob("./images/*.png", files_, false);
    std::vector<std::string> files(files_.begin(), files_.end());

    for (int i = 0; i < files.size(); i++)
    {
       
        cv::Mat img = cv::imread(files[i]);

        clock_t begin, end;
        begin = clock();

        auto objs = yolo->commit(img).get();
        end = clock();
        for (auto& obj : objs) 
        {
           if (obj.confidence > 0.5)
           {
               uint8_t b, g, r;
               cv::rectangle(img, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(0, 0, 255), 1);

       
               auto name = labels[obj.class_label];
               cout<< name <<endl;
               auto caption = cv::format("%s %.2f", name, obj.confidence);
               int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
               cv::rectangle(img, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
               cv::putText(img, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 1, 1);
           }

        }
        
        std::cout << "infetence time = " << end - begin << "ms;      " << objs.size() << "box" << endl;
        cv::Mat resizeImg;
        cv::resize(img, resizeImg, cv::Size(1280, 900));
        cv::imshow("images", resizeImg);
        cv::waitKey(0);

    }

    return 0;
}