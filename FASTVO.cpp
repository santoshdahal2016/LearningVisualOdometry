/*************************************************************************
	> File Name: FASTVO.cpp
	> Author: Santosh Dahal
	> Mail: dahalsantosh2018@gmail.com
 ************************************************************************/

#include<iostream>

#include <pangolin/pangolin.h>
#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cv.hpp>

#include <unistd.h>
#include <thread>
#include <chrono>

using namespace cv;
using namespace std;
using namespace Eigen;


vector<Eigen::Matrix<double, 3, 1>> trajectory;



class Extractor {

public:
    Ptr<Feature2D> orb = ORB::create(500);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    Mat last_descriptor ;
    std::vector<KeyPoint> last_keypoints;

    void extract(Mat &img , vector< Point2f> &src_points ,  vector< Point2f> &dst_points , bool &status){
        std::vector<KeyPoint> keypoints;

        status = false;

        Mat descriptors;
        std::vector< std::vector<DMatch> > matches;

        orb->detectAndCompute(img, Mat(), keypoints, descriptors);
        vector< Point2f> src_points1 , dst_points1;
        if(!last_descriptor.empty()){

            matcher->knnMatch( descriptors, last_descriptor, matches, 2 );

            const float ratio_thresh = 0.5f;

            for (size_t i = 0; i < matches.size(); i++)
            {
                if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
                {
                    src_points1.push_back(keypoints[matches[i][0].queryIdx].pt);
                    dst_points1.push_back(last_keypoints[matches[i][0].trainIdx].pt);
                }
            }

            cout<<src_points1.size()<<endl;
            Mat F , mask;
            F = findFundamentalMat(dst_points1, src_points1,RANSAC, 0.3/460, 0.99, mask);

            SVD decomp = SVD(F);

            cout<<decomp.w<<endl;


            for (int j = 0; j < mask.rows; ++j) {
                if(mask.at<double>(j,0) != 0){
                    src_points.push_back(src_points1.at(j));
                    dst_points.push_back(dst_points1.at(j));
                }
            }
            cout<<src_points.size()<<endl;
            status = true;

        }
        last_descriptor = descriptors;
        last_keypoints =  keypoints;
    }




};




void featureDetection(const Mat &img_1, vector<Point2f> &points) {
    goodFeaturesToTrack(img_1, points, 500, 0.09, 7, Mat(), 7, 3, 0, 0.04);
}


void featureTracking(const Mat &img_1, const Mat &img_2, vector<Point2f> &points1, vector<Point2f> &points2,
                     vector<uchar> &status) {

    //this function automatically gets rid of points for which tracking fails

    vector<float> err;
    Size winSize = Size(21, 21);
    TermCriteria termCriteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termCriteria, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }

    }

}

void DrawTrajectory() {

    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);

        glColor4f(0.1, 0.1, 0.1, 0.4f);
        glBegin(GL_LINES);
        for (GLfloat i = -2.5; i <= 2.5; i += 0.25) {
            glVertex3f(i, 2.5, 0);
            glVertex3f(i, -2.5, 0);
            glVertex3f(2.5, i, 0);
            glVertex3f(-2.5, i, 0);
        }

        glLineWidth(10);

        glColor3f(0.0, 1.0, 0.0);

        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 0.5);


        glColor3f(1.0, 0.0, 0.0);

        glVertex3f(0, 0, 0);
        glVertex3f(0, 0.5, 0);

        glColor3f(0.0, 0.0, 1.0);

        glVertex3f(0, 0, 0);
        glVertex3f(0.5, 0, 0);
        glEnd();

        // Draw the connection
        for (size_t i = 0; i < trajectory.size(); i++) {

            glPointSize(1.5 * 7);
            glBegin(GL_POINTS);
            auto p1 = trajectory[i];
            glColor4f(1.0, 0.0, 1.0, 1);
            glVertex3d(p1[0], p1[2], 0);
            glEnd();
        }

        glPointSize(1.5 * 10);
        glBegin(GL_POINTS);
        auto p1 = trajectory[trajectory.size() - 1];
        glColor4f(0.0, 1.0, 0.0, 1);
        glVertex3d(p1[0], p1[1], 0);
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}


int main(int argc, char **argv) {

    namedWindow("Feature Window");


    thread t1(DrawTrajectory);

    cv::Mat current_frame, current_frameGrey;

    Matrix3d R_f;
    Vector3d t_f;


    trajectory.push_back((Eigen::Matrix<double, 3, 1>(0, 0, 0)));

    cv::Point2d pp(0, 0);

    Mat E, R, t, mask;


    VideoCapture cap("../files/test.mp4");

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }


//    cv::VideoCapture cap(2);

    Extractor extractor1;

    cout << "Initializing" << endl;

    while (1) {
        vector<Point2f> points1, points2;

        bool status;

        cap >> current_frame;
        cvtColor(current_frame, current_frameGrey, COLOR_BGR2GRAY);

        extractor1.extract(current_frameGrey  , points1 ,points2,status);

        if(status){
            E = findFundamentalMat(points2, points1,RANSAC, 0.3/460, 0.99, mask);

            cv::Mat rot, trans;

            cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 200, 0,320, 0, 200, 240, 0, 0, 1);

            recoverPose(E, points2, points1,cameraMatrix, rot, trans,mask);

            Eigen::Matrix3d R_d;
            Eigen::Vector3d T;
            for (int i = 0; i < 3; i++) {
                T(i) = trans.at<double>(i, 0);
                for (int j = 0; j < 3; j++)
                    R_d(i, j) = rot.at<double>(i, j);
            }

            R_f = R_d.transpose();
            t_f =  T;


            break   ;
        }




    }

    trajectory.emplace_back(t_f);

    while (true) {

        vector<Point2f> points1, points2;

        bool status;

        cap >> current_frame;
        cvtColor(current_frame, current_frameGrey, COLOR_BGR2GRAY);

        extractor1.extract(current_frameGrey  , points1 ,points2,status);

        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 844.46113182, 0, 356.18030027, 0, 881.57023676, 208.94028916, 0, 0, 1);

        E = findFundamentalMat(points2, points1,RANSAC, 0.3/460, 0.999, mask);

        if (!E.empty() && E.cols == 3 && E.rows == 3) {

            recoverPose(E, points2, points1,cameraMatrix, R, t,mask);

            Eigen::Matrix3d R_d;
            Eigen::Vector3d T;
            for (int i = 0; i < 3; i++) {
                T(i) = t.at<double>(i, 0);
                for (int j = 0; j < 3; j++)
                    R_d(i, j) = R.at<double>(i, j);
            }



                R_f = R_d.transpose()*R_f;

                t_f =  -R_d.transpose()*T;

//                t_f = t_f + (R_f * t);
//                R_f = R * R_f;
                cout<<R<<endl;

                trajectory.emplace_back(T);


            for( size_t i = 0; i < points1.size(); i++ )
            {
                cv::circle( current_frame, Point2f(int(points1[i].x),int(points1[i].y)), 1, Scalar(0,0,255), 3, 8, 0 );
                cv::circle( current_frame, Point2f(int(points2[i].x),int(points2[i].y)), 1, Scalar(0,255,0), 3, 8, 0 );

                cv::line(current_frame,Point2f(int(points1[i].x),int(points1[i].y)),Point2f(int(points2[i].x),int(points2[i].y)),Scalar(255,0,0),2);
            }

        }
        imshow("Feature Window", current_frame);
        waitKey(1);

        cout << trajectory.size() << endl;
    }
    return 0;
}


