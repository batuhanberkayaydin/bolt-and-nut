#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include "yolo-fastestv2.h"

yoloFastestv2 yoloF2;

#include "jsonConfig.hpp"
#include "track.hpp"

std::vector<struct Bbox> sortBboxes;
std::vector<BoundingBox> boxesSort;
std::vector<TrackingBox> detFrameData;
std::shared_ptr<TRACK> tracker;
int frameId_ = 0;
static const char* class_names[] = {
    "bolt", "nut"
};

/**
 * @brief Hungarian and Kalman Filter Tracker ( More Simple- Simple Object Realtime Tracking(SORT) )
 * 
 * @param img 
 * @param boxes 
 */
void hkfTracker(cv::Mat& img, const std::vector<TargetBox>& boxes, float detectionScoreThresh){
std::cout << "Test1 " << std::endl;
  for(size_t i = 0; i < boxes.size(); i++) {

    if ((boxes[i].score) > detectionScoreThresh)
    {
	std::cout << "Test2 " << std::endl;
        struct Bbox bboxS;
        bboxS.score = 1.0;
        bboxS.x1 = boxes[i].x1;
        bboxS.y1 = boxes[i].y1;
        bboxS.w = boxes[i].x2 - boxes[i].x1;
        bboxS.h = boxes[i].y2 - boxes[i].y1;
        sortBboxes.push_back(bboxS);
        bboxS = {};
        
        char text[256];
        sprintf(text, "%s", class_names[boxes[i].cate]);
        int baseLine = 0;
        
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(cv::Point(boxes[i].x1, boxes[i].y1), cv::Size(label_size.width, label_size.height + baseLine)),
        cv::Scalar(255, 255, 255), -1);
        cv::putText(img, text, cv::Point(boxes[i].x1, boxes[i].y1 + label_size.height),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        

    }
  }
std::cout << "Test3 " << std::endl;
  for (std::vector<struct Bbox>::iterator it = sortBboxes.begin(); it != sortBboxes.end(); it++) {
std::cout << "Test4 " << std::endl;
    boxesSort.push_back(BoundingBox(*it));
  }

  for (unsigned int i = 0; i < boxesSort.size(); ++i) {
std::cout << "Test5 " << std::endl;
    TrackingBox cur_box;
    cur_box.box = boxesSort[i].rect;
    cur_box.id = i;
    cur_box.frame = frameId_;
    detFrameData.push_back(cur_box);
    cur_box = {};
  }
  ++frameId_;

  std::vector<TrackingBox> tracking_results = tracker->update(detFrameData);

  for (TrackingBox it : tracking_results) {
std::cout << "Test6 " << std::endl;
    cv::Rect object(it.box.x, it.box.y, it.box.width, it.box.height);
    cv::rectangle(img, object, tracker->randColor[it.id % 255], 2);
    cv::putText(img,
      std::to_string(it.id),
      cv::Point2f(it.box.x -10, it.box.y),
      cv::FONT_HERSHEY_PLAIN,
      2,
      tracker->randColor[it.id % 255]);

    int centerX = object.x + 0.5 * (object.width - 1);
    int centerY = object.y + 0.5 * (object.height - 1);

  
  }
  
  sortBboxes.clear();
  boxesSort.clear();
  detFrameData.clear();
  tracking_results.clear();
}



int main(int argc, char** argv)
{
    configParams getConfigParams;
    nlohmann::json confParams = getConfigParams.getParams(); 
    std::string videoPath = confParams["videoPath"];
    std::string modelBinFile = confParams["modelBinFile"];
    std::string modelParamFile = confParams["modelParamFile"];
    bool printProcessMs = confParams["printProcessMs"];
    float confThresh = confParams["confThresh"];
    float iouThresh = confParams["iouThresh"];
    bool showImage = confParams["showImage"];
    bool isRecordingEnable = confParams["videoRecord"]["isRecordingEnable"];
    bool isAvi = confParams["videoRecord"]["isAvi"];
    bool isH264 = confParams["videoRecord"]["isH264"];   
    int trackingMaxAge = confParams["tracker"]["maxAge"];   
    tracker = std::shared_ptr<TRACK>(new TRACK(trackingMaxAge, iouThresh));

    float f;
    float FPS[16];
    int i,Fcnt=0;
    cv::Mat frame;
    //some timing
    std::chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    yoloF2.init(true); //we use the GPU of the Jetson Nano

    yoloF2.loadModel("lasttrain130-sim-opt.param","lasttrain130-sim-opt.bin");

    cv::VideoCapture cap("test.mp4");
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Unable to open the camera" << std::endl;
        return 0;
    }

    std::cout << "Start grabbing, press ESC on Live window to terminate" << std::endl;
	cap >> frame;
    cv::VideoWriter writeVideo;
    if (isRecordingEnable) {

        if (isH264){
            writeVideo.open("result.h264", cv::VideoWriter::fourcc('H','2','6','4'), 30, cv::Size(frame.rows,frame.cols),true);

        }
        else {
            writeVideo.open("result.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(frame.rows,frame.cols),true);

        }
    } 
	while(1){
//        frame=cv::imread("000139.jpg");  //need to refresh frame before dnn class detection
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: Unable to grab from the camera" << std::endl;
            break;
        }

        Tbegin = std::chrono::steady_clock::now();

        std::vector<TargetBox> boxes;
        yoloF2.detection(frame, boxes);
        //draw_objects(frame, boxes);
	if(boxes.size() > 0)
		hkfTracker(frame, boxes, confThresh);
        Tend = std::chrono::steady_clock::now();

        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(frame, cv::format("JETSON NANO FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

        //show outputstd::cerr << "ERROR: Unable to grab from the camera" << std::endl;
        cv::imshow("Jetson Nano",frame);
        if (isRecordingEnable)
            writeVideo.write(frame);  
        //cv::imwrite("test.jpg",frame); 
        char esc = cv::waitKey(5); 
        if(esc == 27) break;
	}
    cap.release();
    writeVideo.release();
    return 0;
}
