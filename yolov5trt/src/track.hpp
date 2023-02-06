#ifndef __TRACK_HPP__
#define __TRACK_HPP__

#include <set>
#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#define mydataFmt float

struct Bbox
{
    float score;
    int x1;
    int y1;
    int w;
    int h;
    std::string className;
};

struct BoundingBox
{
    cv::Rect_<float> rect;
    BoundingBox(const Bbox &box)
    {
        rect = cv::Rect_<float>(box.x1, box.y1, box.w, box.h);
    }
};

typedef struct TrackingBox
{
    int frame;
    int id;
    cv::Rect_<float> box;
    std::string className;
} TrackingBox;

struct TRACK
{
// global variables for counting
#define CNUM 255 // max num. of people per frame
    int total_frames = 0;
    double total_time = 0.0;

    cv::Scalar_<int> randColor[CNUM];

    int frame_count = 0;
    int max_age = 1;
    int min_hits = 1;
    double iouThreshold = 0.3;
    std::vector<KalmanTracker> trackers;

    // variables used in the for-loop
    std::vector<cv::Rect_<float>> predictedBoxes;
    std::vector<std::vector<double>> iouMatrix;
    std::vector<int> assignment;
    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    std::vector<cv::Point> matchedPairs;
    std::vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    double cycle_time = 0.0;
    int64 start_time = 0;

    TRACK(int max_age)
    {
        this->max_age = max_age;
        KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
        cv::RNG rng(0xFFFFFFFF);
        for (int i = 0; i < CNUM; i++)
            rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    }

    // Computes IOU between two bounding boxes
    double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
    {
        float in = (bb_test & bb_gt).area();
        float un = bb_test.area() + bb_gt.area() - in;

        if (un < DBL_EPSILON)
            return 0;

        return (double)(in / un);
    }

    std::vector<TrackingBox> update(const std::vector<TrackingBox> &detFrameData)
    {
        total_frames++;
        frame_count++;
        //cout << frame_count << endl;

        // I used to count running time using clock(), but found it seems to conflict with cv::cvWaitkey(),
        // when they both exists, clock() can not get right result. Now I use cv::getTickCount() instead.
        start_time = cv::getTickCount();

        if (trackers.size() == 0) // the first frame met
        {
            // initialize kalman trackers using first detections.
            for (unsigned int i = 0; i < detFrameData.size(); i++)
            {
                KalmanTracker trk = KalmanTracker(detFrameData[i].box);
                trackers.push_back(trk);
            }
            return std::vector<TrackingBox>();
        }

        ///////////////////////////////////////
        // 3.1. get predicted locations from existing trackers.
        predictedBoxes.clear();

        for (auto it = trackers.begin(); it != trackers.end();)
        {
            cv::Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0)
            {
                predictedBoxes.push_back(pBox);
                it++;
            }
            else
            {
                it = trackers.erase(it);
                //cerr << "Box invalid at frame: " << frame_count << endl;
            }
        }

        ///////////////////////////////////////
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
        // dets : detFrameData[fi]
        trkNum = predictedBoxes.size();
        detNum = detFrameData.size();

        iouMatrix.clear();
        iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

        for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < detNum; j++)
            {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[j].box);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        assignment.clear();
        HungAlgo.Solve(iouMatrix, assignment);

        // find matches, unmatched_detections and unmatched_predictions
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum) //	there are unmatched detections
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (detNum < trkNum) // there are unmatched trajectory/predictions
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        }

        // filter out matched with low IOU
        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
            {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
                matchedPairs.push_back(cv::Point(i, assignment[i]));
        }

        ///////////////////////////////////////
        // 3.3. updating trackers

        // update matched trackers with assigned detections.
        // each prediction is corresponding to a tracker
        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); i++)
        {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers[trkIdx].update(detFrameData[detIdx].box);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections)
        {
            KalmanTracker tracker = KalmanTracker(detFrameData[umd].box);
            trackers.push_back(tracker);
        }

        // get trackers' output
        frameTrackingResult.clear();
        for (auto it = trackers.begin(); it != trackers.end();)
        {
            if (((*it).m_time_since_update < 1) &&
                ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
            {
                TrackingBox res;
                res.box = (*it).lastRect;
                res.id = (*it).m_id + 1;
                res.frame = frame_count;
                frameTrackingResult.push_back(res);
                it++;
            }
            else
                it++;

            // remove dead tracklet
            if (it != trackers.end() && (*it).m_time_since_update > max_age)
                it = trackers.erase(it);
        }

        cycle_time = (double)(cv::getTickCount() - start_time);
        total_time += cycle_time / cv::getTickFrequency();

        return frameTrackingResult;
    }
};

#endif