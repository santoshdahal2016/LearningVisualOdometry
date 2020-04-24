/*************************************************************************
	> File Name: OpticalFlow.cpp
	> Author: Santosh Dahal
	> Mail: dahalsantosh2018@gmail.com
 ************************************************************************/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void featureDetection(const Mat &img_1, vector<Point2f> &points) {

    goodFeaturesToTrack(img_1, points, 500, 0.3, 7, Mat(), 7, 3, 0, 0.04);
//    vector<KeyPoint> keypoints;
//    FAST(img_1, keypoints, 20, true);
//    KeyPoint::convert(keypoints, points, vector<int>());
}

void featureTracking(const Mat &img_1, const Mat &img_2, vector<Point2f> &points1f, vector<Point2f> &points2f,
                     vector<uchar> &status) {

    //this function automatically gets rid of points for which tracking fails

    vector<float> err;
    Size winSize = Size(15, 15);
    TermCriteria termCriteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 0.03);

    calcOpticalFlowPyrLK(img_1, img_2, points1f, points2f, status, err, winSize, 3, termCriteria, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        Point2f pt = points2f.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points1f.erase(points1f.begin() + (i - indexCorrection));
            points2f.erase(points2f.begin() + (i - indexCorrection));
            indexCorrection++;
        }

    }

}


int main(int argc, char **argv) {


    cv::Mat current_frame, current_frameGrey, previous_frame, previous_frameGrey;
    vector<Point2f> points1, points2;

    cv::VideoCapture cap(2);


    cap >> previous_frame;
    cvtColor(previous_frame, previous_frameGrey, COLOR_BGR2GRAY);

    Mat mask = cv::Mat::zeros(previous_frame.size() , 16) ;

    cout<<mask.size()<<endl;

    while (true) {

        cap >> current_frame;

        cvtColor(current_frame, current_frameGrey, COLOR_BGR2GRAY);

        vector<uchar> status;

        featureDetection(previous_frameGrey, points1);


        featureTracking(previous_frame, current_frame, points1, points2, status);



        for (int i = 0; i < points1.size(); i++) {
                        line(mask, Point2f(points1[i].x, points1[i].y),
                 Point2f(points2[i].x,points2[i].y), Scalar(0, 255, 100), 5, LINE_8);

            circle(current_frame, Point2f(points1[i].x, points1[i].y), 3, Scalar(0,255,0), -1, 8);

        }

        previous_frame = current_frame.clone();
        previous_frameGrey = current_frameGrey.clone();

        points1 = points2;
        points2.clear();


        namedWindow("Optical Flow");
        imshow("Optical Flow", current_frame+mask);
        waitKey(1);

    }
    return 0;
}


