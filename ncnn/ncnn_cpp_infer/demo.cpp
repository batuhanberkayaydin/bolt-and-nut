#include "yolo-fastestv2.h"
#include "jsonConfig.hpp"
#include "track.hpp"

std::vector<struct Bbox> sortBboxes;
std::vector<BoundingBox> boxesSort;
std::vector<TrackingBox> detFrameData;
std::shared_ptr<TRACK> tracker;
int frameId_;
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

  for(size_t i = 0; i < boxes.size(); i++) {

    if ((boxes[i].score) > detectionScoreThresh)
    {

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

  for (vector<struct Bbox>::iterator it = sortBboxes.begin(); it != sortBboxes.end(); it++) {

    boxesSort.push_back(BoundingBox(*it));
  }

  for (unsigned int i = 0; i < boxesSort.size(); ++i) {

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


int main()
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


    
    yoloFastestv2 api;
    api.init(false); // If you want to use vulcan set true
    api.loadModel(modelParamFile.c_str(),
                  modelBinFile.c_str());

    cv::Mat currentFrame; 
    cv::VideoCapture cap;
    cap.open(videoPath);
    cap >> currentFrame;

    cv::VideoWriter writeVideo;
    if (isRecordingEnable) {

        if (isH264){
            writeVideo.open("result.h264", cv::VideoWriter::fourcc('H','2','6','4'), 30, cv::Size(currentFrame.rows,currentFrame.cols),true);

        }
        else {
            writeVideo.open("result.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(currentFrame.rows,currentFrame.cols),true);

        }
    } 
    while (true)
    {
        double start = cv::getTickCount();

        cap >> currentFrame;
        std::vector<TargetBox> boxes;
        api.detection(currentFrame, boxes);

        for (int i = 0; i < boxes.size(); i++) {

            hkfTracker(currentFrame, boxes, confThresh);
            
            double end = cv::getTickCount();
            std::string totalTime = std::to_string((end - start) / cv::getTickFrequency() * 1000.) + " ms.";
            if (printProcessMs){

                std::cout << "Total Time: " << totalTime << std::endl;
            }
        }
        if (showImage){

            cv::imshow("Result", currentFrame);
            cv::waitKey(1);
        }
        if (isRecordingEnable)
            writeVideo.write(currentFrame);   
    }
                  
    cap.release();
    writeVideo.release();
    return 0;
}
